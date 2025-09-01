import os

import matplotlib.pyplot as plt
import sys

from torch.cuda.amp import autocast

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from  tqdm import tqdm
import time
from datetime import datetime, timedelta
import warnings
import argparse
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
import torch.nn.functional as F
from skimage.morphology import skeletonize
import json
import pandas as pd
from tools.visualize import *
from tools.metric_utils import *
from tools.setting_utils import save_seg_model
from tools.mask2coco import generate_anno

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

def To_BoundaryMask_from_BatchTensor(mask,mode='poly',dilate_pixels=5):
    mask_np = mask.clone().squeeze(1).detach().cpu().numpy()
    boundary_mask = np.zeros_like(mask_np)
    for i in range(mask.shape[0]):
        mask_tmp = mask_np[i]
        if mode == 'line' or mode == 'road':
            # kernel = np.ones((7, 7), np.uint8)
            # mask_tmp = cv2.dilate(np.uint8(mask_tmp), kernel)
            mask_tmp = skeletonize(mask_tmp)*1.0
            mask_tmp[mask_tmp > 0] = 255
            boundary_mask_tmp = cv2.GaussianBlur(mask_tmp, ksize=(dilate_pixels, dilate_pixels), sigmaX=1, sigmaY=1)
            boundary_mask_tmp[boundary_mask_tmp > 0] = 1
            boundary_mask[i] = boundary_mask_tmp
        else:
            padded = np.zeros((mask_tmp.shape[0] + 2, mask_tmp.shape[1] + 2), dtype=np.uint8)
            padded[1:mask_tmp.shape[0] + 1, 1:mask_tmp.shape[1] + 1] = mask_tmp
            dist = cv2.distanceTransform(src=padded, distanceType=cv2.DIST_L2, maskSize=5)
            dist[dist != 1] = 0
            dist[dist == 1] = 255
            boundary_mask_tmp = dist[1:mask_tmp.shape[0] + 1, 1:mask_tmp.shape[1] + 1]

            boundary_mask_tmp = cv2.GaussianBlur(boundary_mask_tmp, ksize=(dilate_pixels, dilate_pixels), sigmaX=1, sigmaY=1)
            boundary_mask_tmp[boundary_mask_tmp > 0] = 1

            boundary_mask[i] = boundary_mask_tmp

    boundary_mask = torch.from_numpy(boundary_mask).unsqueeze(1)
    boundary_mask = boundary_mask.to(mask.device)
    return boundary_mask

def run_train(model, epoch, train_step, args, scaler):
    model.train()
    epoch_time = timedelta(0)
    image_num = 0
    totalLosses = AverageMeter()
    mask_IoUs = AverageMeter()
    boundary_IoUs = AverageMeter()
    if args.class_num == 1:
        stats = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Boundary': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}
    else:
        stats, dataframe = initialize_stats(args.legend, args.class_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    countt = 0
    epochs = args.configs['Experiment']['epoch_num']
    # with tqdm(total=len(args.train_loader), desc=f'Train Epoch {epoch}/{epochs}', unit='img') as pbar:
    for i, batch in enumerate(args.train_loader):
        start = time.time()
        train_step += 1
        img = batch['images']
        ori_img = batch['ori_images']
        segmap_gt = batch['masks']
        boundary_mask_gt = batch['boundary_map']

        image = img.to(device)
        segmap_gt = segmap_gt.to(device)
        boundary_mask_gt = boundary_mask_gt.to(device)

        with autocast(False):         
            start_time = datetime.now()
            outputs = model(image)
            end_time = datetime.now()
            epoch_time += (end_time - start_time)
            image_num += image.size(0)
            # loss = env.detection_loss_function(outputs, pointmap.unsqueeze(1),attentionmap.unsqueeze(1))
            if args.class_num == 1:
                segmap_gt = segmap_gt.float()
            else:
                segmap_gt = segmap_gt.long()

            if args.model_name == 'unetformer':
                loss = args.loss_fuc(outputs[0], segmap_gt) + args.loss_fuc(outputs[1], segmap_gt)
                outputs = outputs[0]
            elif args.model_name == 'ocrnet':
                loss = 0.4 * args.loss_fuc(outputs["pred"], segmap_gt) + args.loss_fuc(outputs["pred"], segmap_gt)
                outputs = outputs["pred"]
            else:
                loss = args.loss_fuc(outputs, segmap_gt)


            # batch_out = {'heatmap_pred': F.sigmoid(outputs), 'outputs': [F.sigmoid(out) for out in sideouts]}


        # backward
        args.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(args.optimizer)
        scaler.update()


        if args.class_num == 1:
            mask_predict = F.sigmoid(outputs)
            # mask_predict = outputs

            mask_predict[mask_predict > args.configs['Model']['threshold']] = 1
            mask_predict[mask_predict <= args.configs['Model']['threshold']] = 0

            boundary_mask_pred = To_BoundaryMask_from_BatchTensor(mask_predict, mode=args.configs['Data']['class_shape'][0],
                                                                  dilate_pixels=args.configs['Model']['assess_model']['delta'])
            # mask_predict_simplified, boundary_mask_pred = To_SimiplifiedBoundaryMask_from_BatchTensor(mask_predict,
            #                                                                                mode=args.configs[
            #                                                                                    'mode'],
            #                                                                                dilate_pixels=
            #                                                                                args.configs[
            #                                                                                    'dilate_pixels'])


            stats_mask = performMetrics(boundary_mask_pred, segmap_gt, n_classes=2)
            mask_iou = stats_mask['IoU']

            stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_gt, n_classes=2)
            boundary_iou = stats_boundary['IoU']
            stats = update_stats(stats, stats_mask, key='Mask')
            stats = update_stats(stats, stats_boundary, key='Boundary')
            batch_out = {'mask_pred': mask_predict, 'mask_pred_simplified': mask_predict,
                         'boundary_mask_pred': boundary_mask_pred,
                         'probability_map': F.sigmoid(outputs)}

        else:
            mask_gt = batch['mask_acc'].to(device)
            mask_predict = F.softmax(outputs, dim=1)
            mask_predict = torch.argmax(mask_predict, dim=1)
            mask_predict_new = torch.zeros_like(outputs[:, 1:, :, :])
            boundary_mask_pred = torch.zeros_like(mask_predict_new)
            boundary_mask_pred = boundary_mask_pred.to(device)
            for cls_id in range(1, args.class_num+1):
                cls_mask = mask_predict.clone()
                cls_mask[cls_mask != cls_id] = 0
                cls_mask[cls_mask == cls_id] = 1
                cls_boundary_mask = To_BoundaryMask_from_BatchTensor(cls_mask,
                                                                     mode=args.configs['class_shape'][cls_id - 1],
                                                                     dilate_pixels=args.configs['dilate_pixels'])
                # cls_mask, cls_boundary_mask = To_SimiplifiedBoundaryMask_from_BatchTensor(cls_mask,
                #                                                                           mode=
                #                                                                           args.configs['class_shape'][
                #                                                                               cls_id - 1],
                #                                                                           dilate_pixels=args.configs[
                #                                                                               'dilate_pixels'])
                mask_predict_new[:, cls_id-1, :, :] = cls_mask
                boundary_mask_pred[:, cls_id-1, :, :] = cls_boundary_mask.squeeze(1)

            stats_mask = performMetrics_multiclass(mask_predict_new, mask_gt, args.legend, n_classes=args.class_num)
            mask_iou = stats_mask['Mean']['IoU']

            stats_boundary = performMetrics_multiclass(boundary_mask_pred, boundary_mask_gt, args.legend, n_classes=args.class_num)
            boundary_iou = stats_boundary['Mean']['IoU']
            stats = update_stats_mulcls(stats, stats_mask, key='Mask')
            stats = update_stats_mulcls(stats, stats_boundary, key='Boundary')

            batch_out = {'mask_pred': mask_predict_new,'boundary_mask_pred': boundary_mask_pred, 'probability_map': F.softmax(outputs, dim=1)}



        totalLosses.update(loss.item())
        mask_IoUs.update(mask_iou)
        boundary_IoUs.update(boundary_iou)

        # measure elapsed time
        batch_time.update(time.time() - start)


        if i ==0 and epoch == 1:
        # # if i + 1:
            print('Visualizing [{}/{} ({:.2f}%)]'.format(i + 1, len(args.train_loader),
                                                                100.0 * (i + 1) / len(args.train_loader)))
            countt += 1
            visualize_segmetation_out(batch, batch_out, epoch, vis_dir=args.vis_root, mode='train')



        # logger.write_hist_parameters(net=model,n_iter=train_step)

        if i % 10 == 0:
            print('Epoch: {} [{}/{} ({:.2f}%)] | trainLoss:{:.6f}'.format(
                epoch, i + 1, len(args.train_loader), 100.0 * (i + 1) / len(args.train_loader), loss.item()))

        # pbar.update(image.shape[0])

    curr_detection_lr = args.optimizer.state_dict()['param_groups'][0]['lr']
    args.logger.write_scalars({
        'TrainLoss': totalLosses.avg,
        'Mask IoU': mask_IoUs.avg,
        'Boundary IoU': boundary_IoUs.avg,
        'point_lr': curr_detection_lr,
    }, tag='train', n_iter=epoch)

    print('Train Epoch:{} | Loss: {:.4f}, Mask IoU: {:.4f}, boundary IoU: {:.4f}'.format(epoch, totalLosses.avg, mask_IoUs.avg, boundary_IoUs.avg))
    print('Train time: {} per image'.format(epoch_time / image_num))
    if args.class_num == 1:
        summary_stats(stats)
    else:
        summary_stats_with_dataframe(stats, dataframe)

    return train_step


def run_val(model,epoch, val_step, best_loss, args, scaler):
    model.eval()
    with torch.no_grad():
        epoch_time = timedelta(0)
        image_num = 0
        totalLosses = AverageMeter()
        mask_IoUs = AverageMeter()
        boundary_IoUs = AverageMeter()
        if args.class_num == 1:
            stats = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                     'Boundary': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}
        else:
            stats, dataframe = initialize_stats(args.legend, args.class_num)
        batch_time = AverageMeter()

        # if args.configs['mode'] != 'road':
        #     stats['Boundary']['MTA'] = []
        epochs = args.configs['Experiment']['epoch_num']
        # with tqdm(total=len(args.val_loader), desc=f'Valid Epoch {epoch}/{epochs}', unit='img') as pbar:
        for i, batch in enumerate(args.val_loader):
            start = time.time()
            val_step += 1
            img = batch['images']
            ori_img = batch['ori_images']
            segmap_gt = batch['masks']
            boundary_mask_gt = batch['boundary_map']

            image = img.to(device)
            segmap_gt = segmap_gt.to(device)
            boundary_mask_gt = boundary_mask_gt.to(device)

            with autocast(False):

                start_time = datetime.now()
                outputs = model(image)
                end_time = datetime.now()
                epoch_time += (end_time - start_time)
                image_num += image.size(0)
                # loss = env.detection_loss_function(outputs, pointmap.unsqueeze(1),attentionmap.unsqueeze(1))
                if args.class_num == 1:
                    segmap_gt = segmap_gt.float()
                else:
                    segmap_gt = segmap_gt.long()

                if args.model_name == 'unetformer':
                    loss = args.loss_fuc(outputs[0], segmap_gt) + args.loss_fuc(outputs[1], segmap_gt)
                    outputs = outputs[0]
                elif args.model_name == 'ocrnet':
                    loss = 0.4 * args.loss_fuc(outputs["pred"], segmap_gt) + args.loss_fuc(outputs["pred"], segmap_gt)
                    outputs = outputs["pred"]
                else:
                    loss = args.loss_fuc(outputs, segmap_gt)

                # batch_out = {'heatmap_pred': F.sigmoid(outputs), 'outputs': [F.sigmoid(out) for out in sideouts]}


            if args.class_num == 1:
                mask_predict = F.sigmoid(outputs)
                # mask_predict = outputs

                mask_predict[mask_predict > args.configs['Model']['threshold']] = 1
                mask_predict[mask_predict <= args.configs['Model']['threshold']] = 0

                boundary_mask_pred = To_BoundaryMask_from_BatchTensor(mask_predict, mode=args.configs['Data']['class_shape'][0],
                                                                  dilate_pixels=args.configs['Model']['assess_model']['delta'])
                # mask_predict_simplified, boundary_mask_pred = To_SimiplifiedBoundaryMask_from_BatchTensor(mask_predict, mode=args.configs['mode'],
                #                                                       dilate_pixels=args.configs['dilate_pixels'])

                stats_mask = performMetrics(mask_predict, segmap_gt, n_classes=2)
                mask_iou = stats_mask['IoU']

                stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_gt, n_classes=2)
                boundary_iou = stats_boundary['IoU']
                stats = update_stats(stats, stats_mask, key='Mask')
                stats = update_stats(stats, stats_boundary, key='Boundary')
                batch_out = {'mask_pred': mask_predict, 'mask_pred_simplified': mask_predict,
                             'boundary_mask_pred': boundary_mask_pred,
                             'probability_map': F.sigmoid(outputs)}

            else:
                mask_gt = batch['mask_acc'].to(device)
                mask_predict = F.softmax(outputs, dim=1)
                mask_predict = torch.argmax(mask_predict, dim=1)
                mask_predict_new = torch.zeros_like(outputs[:, 1:, :, :])
                boundary_mask_pred = torch.zeros_like(mask_predict_new)
                boundary_mask_pred = boundary_mask_pred.to(device)
                for cls_id in range(1, args.class_num + 1):
                    cls_mask = mask_predict.clone()
                    cls_mask[cls_mask != cls_id] = 0
                    cls_mask[cls_mask == cls_id] = 1
                    cls_boundary_mask = To_BoundaryMask_from_BatchTensor(cls_mask,
                                                                         mode=args.configs['class_shape'][cls_id - 1],
                                                                         dilate_pixels=args.configs['dilate_pixels'])
                    # cls_mask, cls_boundary_mask = To_SimiplifiedBoundaryMask_from_BatchTensor(cls_mask,
                    #                                                      mode=args.configs['class_shape'][cls_id - 1],
                    #                                                      dilate_pixels=args.configs['dilate_pixels'])
                    mask_predict_new[:, cls_id - 1, :, :] = cls_mask
                    boundary_mask_pred[:, cls_id - 1, :, :] = cls_boundary_mask.squeeze(1)

                stats_mask = performMetrics_multiclass(mask_predict_new, mask_gt, args.legend, n_classes=args.class_num)
                mask_iou = stats_mask['Mean']['IoU']

                stats_boundary = performMetrics_multiclass(boundary_mask_pred, boundary_mask_gt, args.legend,
                                                           n_classes=args.class_num)
                boundary_iou = stats_boundary['Mean']['IoU']
                stats = update_stats_mulcls(stats, stats_mask, key='Mask')
                stats = update_stats_mulcls(stats, stats_boundary, key='Boundary')

                batch_out = {'mask_pred': mask_predict_new, 'boundary_mask_pred': boundary_mask_pred,
                            'probability_map': F.softmax(outputs, dim=1)}


            totalLosses.update(loss.item())
            mask_IoUs.update(mask_iou)
            boundary_IoUs.update(boundary_iou)

            # measure elapsed time
            batch_time.update(time.time() - start)



            if i % 10 == 0:
                print(
                    'Evaluating [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader), 100.0 * (i + 1) / len(args.val_loader)))
            # pbar.update(image.shape[0])
        args.logger.write_scalars({
            'TrainLoss': totalLosses.avg,
            'Mask IoU': mask_IoUs.avg,
            'Boundary IoU':boundary_IoUs.avg,
        }, tag='val', n_iter=epoch)

        print('Valid Epoch:{} | Loss: {:.4f}, Mask IoU: {:.4f}, boundary IoU: {:.4f}'.format(epoch, totalLosses.avg, mask_IoUs.avg, boundary_IoUs.avg))
        print('Infer time: {} per image'.format(epoch_time / image_num))
        if args.class_num == 1:
            summary_stats(stats)
        else:
            summary_stats_with_dataframe(stats, dataframe)

        args.early_stopping([totalLosses,mask_IoUs,boundary_IoUs], best_loss)

        if args.early_stopping.save_model and not args.configs['Experiment']['evaluate']:
            visualize_segmetation_out(batch, batch_out, epoch, vis_dir=args.vis_root, mode='val')
            best_loss['total_loss'] = totalLosses.avg
            best_loss = save_seg_model(model, args, best_loss, [totalLosses,mask_IoUs,boundary_IoUs], epoch)
            pass
        if args.early_stopping.early_stop:
            best_loss['early_stop'] = True
        return val_step,best_loss


categories = [
    {'supercategory': 'none', 'id': 1, 'name': 'building'},
]

def save_train_seg(args, save_mode='coco'):
    model = args.deeplabv3plus
    for m_name in args.model_names:
        print('Processing {} seg...'.format(m_name))
        if m_name == 'transunet':
            model = args.transunet
        elif m_name == 'hrnet':
            model = args.hrnet
        elif m_name == 'unetformer':
            model = args.unetformer
        elif m_name == 'ocrnet':
            model = args.ocrnet
        else:
            pass
        images_info = []
        annotations = []
        label_save_dir = args.configs['Paths']['TrainRoot']
        ins_count = 0
        IoUs = AverageMeter()
        for i, batch in tqdm(enumerate(args.train_loader), total=len(args.train_loader), desc="Trainset"):
            start = time.time()
            
            image = batch['images'].to(device)
            mask_true = batch['masks'].to(device)
            name = batch['names'][0]   
            
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                model.eval()
                with torch.no_grad():
                    pred = model(image)

                mask_pred = F.sigmoid(pred)
                mask_pred[mask_pred > args.configs['Model']['threshold']] = 1
                mask_pred[mask_pred <= args.configs['Model']['threshold']] = 0
                             
                eps = 1e-7           
            # 前景IoU
            intersection_fg = (mask_true * mask_pred).sum(dim=(1,2,3))  # [B]
            union_fg = mask_true.sum(dim=(1,2,3)) + mask_pred.sum(dim=(1,2,3)) - intersection_fg  # [B]
            iou_fg = (intersection_fg + eps) / (union_fg + eps)  # [B]

            # 背景IoU（直接反转mask）
            intersection_bg = ((1 - mask_true) * (1 - mask_pred)).sum(dim=(1,2,3))  # [B]
            union_bg = (1 - mask_true).sum(dim=(1,2,3)) + (1 - mask_pred).sum(dim=(1,2,3)) - intersection_bg  # [B]
            iou_bg = (intersection_bg + eps) / (union_bg + eps)  # [B]

            # 计算最终平均IoU
            mean_iou = ((iou_fg + iou_bg) / 2).mean()  # 先求每个样本的前后景平均，再整体平均
            IoUs.update(100 *(mean_iou.item()))

            mask_pred = np.squeeze(mask_pred.detach().cpu().numpy().astype(np.uint8))
            mask_true = np.squeeze(mask_true.detach().cpu().numpy().astype(np.uint8))
            if save_mode == 'mask':
                # img = image.detach().cpu().numpy()[0] * 255
                # img = img.transpose([1,2,0]).astype(np.uint8)
                # plt.figure()
                # plt.subplot(131)
                # plt.imshow(img)
                # plt.subplot(132)
                # plt.imshow(mask_true)
                # plt.subplot(133)
                # plt.imshow(mask_pred)
                # plt.savefig('seg_check.png')

                # if args.model_names.index(m_name) == 0:
                #     gt_save_dir = os.path.join(label_save_dir, 'binary_map')
                #     os.system('mkdir -p {}'.format(gt_save_dir))
                #     gt_save_path = os.path.join(gt_save_dir, "{}.tif".format(name))
                #     cv2.imwrite(gt_save_path, mask_true*255)
                mask_save_dir = os.path.join(label_save_dir, 'seg_preds')
                mask_save_dir = os.path.join(mask_save_dir, m_name)
                os.system('mkdir -p {}'.format(mask_save_dir))
                mask_save_path = os.path.join(mask_save_dir, "{}.tif".format(name))
                cv2.imwrite(mask_save_path, mask_pred*255)     
                
            else:
                images_info, annotations, ins_count = generate_anno(mask_pred, images_info, annotations, name, ins_count, i)
        
        if save_mode == 'coco':
            coco_instance = {'images': images_info, 'annotations': annotations, 'categories': categories}

            os.system('mkdir -p {}'.format(label_save_dir))
            coco_save_path = os.path.join(label_save_dir, "coco_{}_pred.json".format(m_name))
            with open(coco_save_path, 'w') as f:
                json.dump(coco_instance, f)
        
        print("模型IoU精度", IoUs.avg)


def save_val_seg(args, save_mode='mask'):
    model = None
    for m_name in args.model_names:
        print('Processing {} seg...'.format(m_name))
        if m_name == 'transunet':
            model = args.transunet
        elif m_name == 'hrnet':
            model = args.hrnet
        elif m_name == 'unetformer':
            model = args.unetformer
        elif m_name == 'ocrnet':
            model = args.ocrnet
        else:
            model = args.deeplabv3plus
        images_info = []
        annotations = []
        label_save_dir = args.configs['Paths']['ValRoot']
        ins_count = 0
        for i, batch in tqdm(enumerate(args.val_loader), total=len(args.val_loader), desc="Valset"):
            start = time.time()
            image = batch['images'].to(device)
            mask_true = batch['masks'].to(device)
            name = batch['names'][0]    
            
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                model.eval()
                with torch.no_grad():
                    pred = model(image)
            if m_name == 'ocrnet':
                pred = pred['pred']
            elif m_name == 'unetformer':
                pred = pred[0]
            mask_pred = F.sigmoid(pred)
            mask_pred[mask_pred > args.configs['Model']['threshold']] = 1
            mask_pred[mask_pred <= args.configs['Model']['threshold']] = 0

            mask_pred = np.squeeze(mask_pred.detach().cpu().numpy().astype(np.uint8))

            if save_mode == 'mask':
                mask_save_dir = os.path.join(label_save_dir, 'seg_preds')
                mask_save_dir = os.path.join(mask_save_dir, m_name)
                os.system('mkdir -p {}'.format(mask_save_dir))
                mask_save_path = os.path.join(mask_save_dir, "{}.tif".format(name))
                cv2.imwrite(mask_save_path, mask_pred*255)
            else:
                images_info, annotations, ins_count = generate_anno(mask_pred, images_info, annotations, name, ins_count, i)
        
        if save_mode == 'coco':
            coco_instance = {'images': images_info, 'annotations': annotations, 'categories': categories}

            os.system('mkdir -p {}'.format(label_save_dir))
            coco_save_path = os.path.join(label_save_dir, "coco_{}_pred.json".format(m_name))
            with open(coco_save_path, 'w') as f:
                json.dump(coco_instance, f)
        