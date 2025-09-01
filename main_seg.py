import os

import matplotlib.pyplot as plt
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import time
import datetime
import warnings
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
import torch.nn.functional as F
from Data.DataLoader import Dataset_Loader
from solvers.segmentation_trainer import run_train, run_val
from models.Segmentation.DeepLabv3plus.deeplabv3plus_resnet import resnet101_deeplabv3p as deeplabv3plus
from models.Segmentation.unetformer import UNetFormer
from models.Segmentation.hrnet import HRNet
from models.Segmentation.TransUnet.vit_seg_modeling import get_transunet
from models.Segmentation.OCRNet.OCRNet import OCRNet
from tools.setting_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

setup_seed(20)
parser = argparse.ArgumentParser(description='Params for label making')
parser.add_argument(
    '-c', '--configs_path',
    type=str,
    default=r'./configs/config_Inria.json',
    # default=r'./configs/config_CrowdAI.json',
    # default=r'./configs/config_DeepGlobe.json',
    # default=r'./configs/config_Massachusetts.json',
    # default=r'./configs/config_GID.json',  
    # default=r'./configs/config_BAQS.json',
    # default=r'./configs/config_WAQS.json',
    help='Name of the configs file, including the .json file extension.')

parser.add_argument(
    '-model_name',
    type=str,         
    default="deeplabv3+",
    help='Name of segmentation model.All datasets support "deeplabv3+", "hrnet", "transunet", "unetformer", "ocrnet"')

args = parser.parse_args()
assert args.configs_path is not None, "Argument --configs must be specified. Run 'python main.py --help' for help on arguments."
configs = load_config(args.configs_path)
args.configs = configs

args.work_dir, args.vis_root, args.model_dir, args.record_root = build_roots(configs)

args.logger = LogSummary(args.work_dir)

make_print_to_file(args)
args.early_stopping = EarlyStopping(5, verbose=True, path=args.model_dir)


args.train_loader, args.val_loader = Dataset_Loader(args, configs)

args.class_num = 1

args.aux_loss = False

args.model_name = args.model_name
print('Apply ' + args.model_name + ' Model')
if args.model_name == 'deeplabv3+':
    # network = DeepLabV3(aspp_dim=2048, num_classes=args.class_num+1)
    network = deeplabv3plus(in_channel=3, n_class=args.class_num, output_stride=8, pretrained=True)
elif args.model_name == 'transunet':
    network = get_transunet(in_channel=3, img_size=args.configs['Data']['image_size'], num_classes=args.class_num, pretrain=True)
elif args.model_name == 'ocrnet':
    network = OCRNet(num_classes=args.class_num)
    args.aux_loss = True
elif args.model_name == 'hrnet':
    # network = HRNetSeg(num_classes=args.class_num + 1)
    network = HRNet(num_classes=args.class_num,
                    pretrained='/root/autodl-tmp/Projects/checkpoints/hrnetv2_w48_imagenet_pretrained.pth')

elif args.model_name == 'unetformer':
    network = UNetFormer(num_classes=args.class_num, pretrained=True).to(device)
    args.aux_loss = True


net = nn.DataParallel(network)
net = net.to(device)

args.loss_fuc = nn.CrossEntropyLoss() if args.configs['Data']['class_num'] > 1 else nn.BCEWithLogitsLoss()
args.optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)

def main():
    scaler = torch.cuda.amp.GradScaler()
    print(configs)
    best_loss = {'total_loss': 1000, 'IoU': 0, 'boundary IoU': 0, 'early_stop': False}
    train_step = 0
    val_step = 0
    for epoch in range(1, args.configs['Experiment']['epoch_num']):
        start_time = datetime.datetime.now()
        print('Training Epoch:{}'.format(epoch))
        train_step = run_train(net, epoch, train_step, args, scaler)
        train_end_time = datetime.datetime.now()
        print('Validating...')
        val_step, best_loss = run_val(net, epoch, val_step, best_loss, args, scaler)
        end_time = datetime.datetime.now()
        print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))
        print(f'Training time = {train_end_time - start_time} | Validation time = {end_time - train_end_time}')
        if best_loss['early_stop']:
            print('Early Stop!')
            break


if __name__ == '__main__':
    main()
