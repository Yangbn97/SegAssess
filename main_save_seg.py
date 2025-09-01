import argparse
import datetime
import shutil
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import time
import sys
from functools import partial
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torchvision.transforms.v2 as transforms
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.multiprocessing import spawn
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR
import torch.cuda.amp as amp
import torch.nn.functional as F
import warnings
from Data.DataLoader import Dataset_Loader
from tools.setting_utils import *
from models.buildmodel import build_seg_model
from solvers.segmentation_trainer import save_train_seg, save_val_seg
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument(
        '-c', '--configs',
        type=str,           
        default=r'./configs/config_Inria.json',
        # default=r'./configs/config_CrowdAI.json',
        # default=r'./configs/config_DeepGlobe.json',
        # default=r'./configs/config_Massachusetts.json',
        # default=r'./configs/config_GID.json',  
        # default=r'./configs/config_BAQS.json',
        # default=r'./configs/config_WAQS.json',
        help='Name of the configs file, excluding the .json file extension.')
    
    parser.add_argument(
        '-model_names',
        nargs='+', 
        type=str,         
        default=["deeplabv3+", "hrnet", "transunet", "unetformer", "ocrnet"],
        help='Name of segmentation model.All datasets support "deeplabv3+", "hrnet", "transunet", "unetformer", "ocrnet"')
    
    args = parser.parse_args()
    assert args.configs is not None
    cfg = load_config(args.configs)
    args.configs = cfg
    args.image_size = args.configs['Data']['image_size']

    args.class_num = 1

    return args

def main():
    args = get_parser()
    args.configs['DDP']['flag'] = 0
    args.configs['Experiment']['evaluate'] = 0
    args.configs['Model']['seg_model']['name'] = args.model_names

    args = build_seg_model(args)

    args.workers = args.configs['Experiment']['num_workers']
    args.configs['Experiment']['batch_size'] = 1
    
    args.train_loader, args.val_loader = Dataset_Loader(args, args.configs)

    dataset_name = args.configs['Data']['dataset_name']
    print(f'--------------Processing {dataset_name}-----------------')

    start_time = datetime.datetime.now()

    # print('Start processing train dataset...')
    # save_train_seg(args, save_mode='mask')
   
    print('Start processing val dataset...')
    save_val_seg(args, save_mode='mask')
    end_time = datetime.datetime.now()
    print("Time Elapsed for Processing => {}".format(end_time - start_time))



if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()