import argparse
import datetime
import shutil
import os
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import time
import sys
from functools import partial
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
import builtins
from Data.DataLoader import Dataset_Loader_DDP
from tools.setting_utils import *
from models.buildmodel import build_seg_model, build_assess_model
from solvers.SegAssess_ddp import run_train, run_val, run_eval
warnings.filterwarnings("ignore")

original_print = builtins.print

def is_rank_0():
    return dist.get_rank() == 0

# Define print function
def print_if_rank0(*args, **kwargs):
    if is_rank_0():
        original_print(*args, **kwargs)  # 使用原始的 print 函数


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument(
        '-c', '--configs',
        type=str,
        # default=r'./configs/config_Inria.json',
        default=r'./configs/config_CrowdAI.json',
        # default=r'./configs/config_DeepGlobe.json',
        # default=r'./configs/config_Massachusetts.json',
        # default=r'./configs/config_GID.json',  
        # default=r'./configs/config_BAQS.json',
        # default=r'./configs/config_WAQS.json',
        help='Name of the configs file, including the .json file extension.')
    
    parser.add_argument(
        '-eval_seg_model',
        type=str,
        default='ocrnet',
        help='All datasets support "deeplabv3+", "hrnet", "transunet", "unetformer", "ocrnet" . BAQS and WAQS additional support "raw"')
    
    parser.add_argument(
        '-eval',
        type=int,
        default=True,
        help='Whether to save inference results during evaluation')
    
    parser.add_argument(
        '-eval_save_out',
        type=int,
        default=False,
        help='Whether to save inference results during evaluation')
    
    args = parser.parse_args()
    assert args.configs is not None
    cfg = load_config(args.configs)
    args = set_ddp(args, cfg, key='DDP')
    args.configs = cfg
    args.image_size = args.configs['Data']['image_size']
    args.configs['Paths']['records_filename'] = f'{args.configs['Data']['dataset_name']}_AMS2{args.eval_seg_model}'
    if 'eval' in args.configs['Experiment']['tasks']:
        args.eval = True
    args.class_num = 1

    return args

def main():
    args = get_parser()
    setup_seed(20)
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))

def main_worker(gpu, args):    
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()

    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    dist.barrier()

    args.logger_root, args.vis_root, args.weight_root, args.record_root = build_roots(args.configs)
    args.logger = LogSummary(args.logger_root)

    # Replace print
    builtins.print = print_if_rank0
    
    make_print_to_file(args)

    args.augmentator = MultiTensorAugmentator()
    if not args.eval:
        args = build_seg_model(args)
    args = build_assess_model(args)

    args.early_stopping = EarlyStopping(5, verbose=True, path=args.weight_root)

    if args.configs['Experiment']['batch_size'] >= args.ngpus_per_node:
        args.configs['Experiment']['batch_size'] = int(args.configs['Experiment']['batch_size'] / args.ngpus_per_node)
    args.configs['Experiment']['num_workers'] = int(
        (args.configs['Experiment']['num_workers'] + args.ngpus_per_node - 1) / args.ngpus_per_node)
    args.workers = args.configs['Experiment']['num_workers']
    
    args.train_loader, args.val_loader = Dataset_Loader_DDP(args, args.configs)

    print(args.configs)
    if args.eval:
        run_eval(args, vis=True, save_outs=args.eval_save_out)
        dist.destroy_process_group()
        return
    if 'train-val' in args.configs['Experiment']['tasks']:
        scaler = torch.cuda.amp.GradScaler()
        best_loss = {'total_loss': 1000, 'IoU': 0,'early_stop': False}
        train_step = 0
        val_step = 0
        epoch_num = args.configs['Experiment']['epoch_num'] if not args.configs['Experiment']['evaluate'] else 2
        for epoch in range(1, epoch_num):
            start_time = datetime.datetime.now()
            print('Training Epoch:{}'.format(epoch))
            train_step = run_train(epoch, train_step, args,scaler)
            if epoch:
                print('Validating...')
                val_step, best_loss = run_val(epoch, val_step, best_loss, args)
            end_time = datetime.datetime.now()
            print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))
            if best_loss['early_stop']:
                print('Early Stop !')
                return
        dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()