import datetime
import random
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import os
import torch
import errno
from collections import OrderedDict
from jsmin import jsmin
import json
from matplotlib.patches import Rectangle
import pandas as pd
from pandas import Series

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiTensorAugmentator:
    def __init__(self):
        self.transform_params = None

    def augment_batch(self, *tensors):
        batch_size = tensors[0].size(0)
        num_tensors = len(tensors)
        self.transform_params = []
        
        augmented = [[] for _ in range(num_tensors)]

        for i in range(batch_size):
            sample_tensors = [t[i] for t in tensors]
            sample_params = []
            
            for _ in range(4):
                k = torch.randint(0, 4, (1,)).item()
                flip_type = torch.randint(0, 3, (1,)).item()
                
                for t_idx, tensor in enumerate(sample_tensors):
                    # rotate
                    transformed = torch.rot90(tensor, k, dims=(-2, -1))
                    
                    # flip
                    if flip_type == 1:
                        transformed = transformed.flip(-1)  
                    elif flip_type == 2:
                        transformed = transformed.flip(-2)  
                    
                    # collect augs
                    augmented[t_idx].append(transformed)
                
                sample_params.append((k, flip_type))
            
            self.transform_params.append(sample_params)

        augmented_tensors = []
        for a in augmented:
            stacked = torch.stack(a)  # shape: [B*4, ... , H, W]
            reshaped = stacked.view(batch_size, 4, *stacked.shape[1:])
            augmented_tensors.append(reshaped)

        return tuple(augmented_tensors)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def load_config(config_name="configs", config_dirpath="", try_default=False):
    if os.path.splitext(config_name)[1] == ".json":
        config_filepath = os.path.join(config_dirpath, config_name)
    else:
        config_filepath = os.path.join(config_dirpath, config_name + ".json")

    try:
        with open(config_filepath, 'r') as f:
            minified = jsmin(f.read())
            # config = json.loads(minified)
            try:
                config = json.loads(minified)
            except json.decoder.JSONDecodeError as e:
                print("ERROR: Parsing configs failed:")
                print(e)
                print("Minified JSON causing the problem:")
                print(str(minified))
                exit()
        return config
    except FileNotFoundError:
        if config_name == "configs" and config_dirpath == "":
            print(
                "WARNING: the default configs file was not found....")
            return None
        elif try_default:
            print(
                "WARNING: configs file {} was not found, opening default configs file configs.defaults.json instead.".format(
                    config_filepath))
            return load_config()
        else:
            print(
                "WARNING: configs file {} was not found.".format(config_filepath))
            return None


def set_ddp(args, cfg, key='DDP'):
    args.dist_url = cfg[key]['dist_url']
    args.dist_backend = cfg[key]['dist_backend']
    args.multiprocessing_distributed = cfg[key]['multiprocessing_distributed']
    args.world_size = cfg[key]['world_size']
    args.rank = cfg[key]['rank']
    args.sync_bn = cfg[key]['sync_bn']
    return args

def build_roots(configs):
    record_root = os.path.join(configs['Paths']['RecordRoot'], configs['Paths']['records_filename'])
    work_dir = '{}/{}'.format(os.path.join(record_root, 'runs'), 'Train')
    vis_root = os.path.join(record_root, 'vis')
    model_dir = os.path.join(record_root, 'weights')

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return work_dir, vis_root, model_dir, record_root



def load_ckpt(weight_dir, model, dp_mode='DDP'):
    pre_weight = torch.load(weight_dir)
    new_pre_weight = OrderedDict()
    # pre_weight =torch.jit.load(resume)
    model_dict = model.state_dict()
    new_model_dict = OrderedDict()

    for k, v in pre_weight.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        new_pre_weight[new_k] = v
    for k, v in model_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        new_model_dict[new_k] = v

    pre_weight = new_pre_weight  # ["model_state"]
    pretrained_dict = {}
    t_n = 0
    v_n = 0
    unmatched_keys = []
    for k, v in pre_weight.items():
        t_n += 1
        if k in new_model_dict and v.shape == new_model_dict[k].shape:
            # k = 'module.' + k if 'module' not in k else k
            v_n += 1
            pretrained_dict[k] = v
            # print(k)
        else:
            unmatched_keys.append(k)
    # os._exit()
    if dp_mode == 'DDP':
        if dist.get_rank() == 0:
            print(f'{v_n}/{t_n} weights have been loaded!')
            print('Unmatched keys in state_dict:', unmatched_keys)
    else:
        print(f'{v_n}/{t_n} weights have been loaded!')
        print('Unmatched keys in state_dict:', unmatched_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed) 
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)



def save_assess_model(args, best_loss, current_loss, epoch, update_best=True):
    if update_best:
        best_loss['total_loss'] = current_loss[0].avg
        best_loss['IoU'] = current_loss[1].avg

    if args.configs['Model']['assess_model']['save']:
        path = os.path.join(args.weight_root, 'best.pkl')
        torch.save(args.assess_model.state_dict(), path)
        print('Saving Assess Model=>', path)
    
    return best_loss

def save_seg_model(model, args, best_loss, current_loss, epoch):
    best_loss['total_loss'] = current_loss[0].avg
    best_loss['IoU'] = current_loss[1].avg
    best_loss['boundary IoU'] = current_loss[2].avg

    path = os.path.join(args.model_dir, 'best.pkl')
    torch.save(model.state_dict(), path)
    print('Saving Model=>', path)

    return best_loss

def get_dataframe(legend_data, metrics=None):
    if metrics is None:
        metrics = {'Pixel Accuracy': [0], 'Precision': [0], 'Recall': [0], 'F1-score': [0], 'IoU': [0]}
    row_idx = []
    for d in legend_data:
        row_idx.append(d[1])
    row_idx.append('Mean')
    row_idx = Series(row_idx)
    metrics_table = pd.DataFrame(metrics, index=row_idx)
    # for k, v in metrics:
    #     metrics_table[k] = metrics_table[k].astype('object')
    return metrics_table


class LogSummary(object):

    def __init__(self, log_path):

        mkdirs(log_path)
        self.writer = SummaryWriter(log_path)

    def write_scalars(self, scalar_dict, n_iter, tag=None):

        for name, scalar in scalar_dict.items():
            if tag is not None:
                name = '/'.join([tag, name])
            self.writer.add_scalar(name, scalar, n_iter)

    def write_hist_parameters(self, net, n_iter):
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().detach().numpy(), n_iter)
            self.writer.add_histogram(name + '/grad', param.grad.clone().data.cpu().numpy(), n_iter)


def make_print_to_file(env):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    path = env.record_root

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    if env.configs['Experiment']['evaluate']:
        fileName = datetime.datetime.now().strftime('eval' + ' | ' + 'day' + '%Y-%m-%d %H:%M:%S')
    elif env.configs['Experiment']['test']:
        fileName = datetime.datetime.now().strftime('infer' + ' | ' + 'day' + '%Y-%m-%d %H:%M:%S')
    else:
        fileName = datetime.datetime.now().strftime('train' + ' | ' + '%Y-%m-%d %H:%M:%S')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_model = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, current_loss, best_score):

        # if current_loss[0].avg <= best_score['total_loss']:
        if current_loss[1].avg >= best_score['IoU'] :
            self.save_model = True
            self.counter = 0
        else:
            self.save_model = False
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True




