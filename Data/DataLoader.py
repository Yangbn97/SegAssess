import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as data
import numpy as np
import torch
import random
from functools import partial
from Data.dataset_base import Dataset_base
from Data.dataset_offline import Dataset_AMS



def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Dataset_Loader(args, configs, is_seg=False):
    train_ROOT = configs['Paths']['TrainRoot']
    val_ROOT = configs['Paths']['ValRoot']
    train_loader = []
    val_loader = []

    drop_last_flag = False
    if not configs['Experiment']['evaluate']:
        drop_last_flag = True
        trainset = Dataset_base(train_ROOT, mode='train')
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=configs['Experiment']['batch_size'],
            shuffle=True,
            num_workers=configs['Experiment']['num_workers'], pin_memory=True, drop_last=drop_last_flag)

    if is_seg:
        valset = Dataset_base(val_ROOT, mode='valid')
    else:
        valset = Dataset_AMS(val_ROOT, seg_model=args.eval_seg_model)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=configs['Experiment']['batch_size'],
        shuffle=True,
        num_workers=configs['Experiment']['num_workers'],pin_memory=True, drop_last=drop_last_flag)

    return train_loader, val_loader

def Dataset_Loader_DDP(args, configs):
    train_ROOT = configs['Paths']['TrainRoot']
    val_ROOT = configs['Paths']['ValRoot']
    # NUM_POINTS = configs['Data']['NUM_POINTS']
    # dilate_pixels = configs['Model']['detection_model']['dilate_pixels']
    train_loader = []
    val_loader = []
    drop_last_flag = False if  configs['Experiment']['evaluate'] else True
       
    if not configs['Experiment']['evaluate']:
        trainset = Dataset_base(train_ROOT, mode='train')
    valset = Dataset_AMS(val_ROOT, seg_model=args.eval_seg_model)

    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=20)

    if not configs['Experiment']['evaluate']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=configs['Experiment']['batch_size'],
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=init_fn,
            sampler=train_sampler,
            drop_last=drop_last_flag)

    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=configs['Experiment']['batch_size'],
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn,
        sampler=val_sampler,
        drop_last=drop_last_flag)

    return train_loader, val_loader
