import os
import matplotlib.pyplot as plt
import sys
from torch.cuda.amp import autocast
import torch.distributed as dist
from torchvision import tv_tensors
# from models.buildmodel import save_model
import time
from datetime import datetime
import warnings
import argparse
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
from tools.setting_utils import *
from models.loss import HDNet_RCF_edge_criterion
from tools.visualize import *

from tools.metric_utils import *
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

legend_data = [
            [[255, 0, 0], 'True Postive', 'TP'],
            [[0, 255, 0], 'False Postive', 'FP'],
            [[0, 0, 255], 'True Negtive', 'TN'],
            [[0, 255, 255], 'False Negtive', 'FN']]

def run_train(epoch, train_step, args, scaler):
    totalLosses = AverageMeter()
    losses_CE = AverageMeter()
    losses_Edge = AverageMeter()
    losses_Refine = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    countt = 0

    stats_assess_4cls, dataframe = initialize_stats(legend_data, 4)
    stats_assess = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}

    for i, batch in enumerate(args.train_loader):
        start = time.time()
        train_step += 1
        names = batch['names']

        image = batch['images'].to(device)
        ori_img = batch['ori_images']
        mask_true = batch['masks'].to(device)
        edge_map_true = batch['edge_map'].to(device)
        
        # 数据增强
        image, mask_true, edge_map_true = args.augmentator.augment_batch(image, mask_true, edge_map_true)
        extended_name_list = [name for name in names for _ in range(4)]

        batch_out = {}
        batch_out['init_preds'] = {}
        # with autocast():
        
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                args.deeplabv3plus.eval()
                args.transunet.eval()
                args.hrnet.eval()
                args.unetformer.eval()
                # args.ocrnet.eval()
                with torch.no_grad():
                    pred1 = args.deeplabv3plus(image[:,0,...])
                    pred2 = args.transunet(image[:,1,...])
                    pred3 = args.hrnet(image[:,2,...])
                    pred4 = args.unetformer(image[:,3,...])[0]
                    # pred5 = args.ocrnet(image)
            
            init_pred = torch.cat([pred1,pred2,pred3,pred4],dim=1)
            mask_pred = F.sigmoid(init_pred)
            mask_pred[mask_pred > args.configs['Model']['threshold']] = 1
            mask_pred[mask_pred <= args.configs['Model']['threshold']] = 0

            mask_true = torch.cat([mask_true[:,0,...], mask_true[:,1,...], mask_true[:,2,...], mask_true[:,3,...]], dim=1)
            edge_map_true = torch.cat([edge_map_true[:,0,...], edge_map_true[:,1,...], edge_map_true[:,2,...], edge_map_true[:,3,...]], dim=1) 

            new_shape = [mask_pred.shape[0]*mask_pred.shape[1], 1, mask_pred.shape[2], mask_pred.shape[3]]
            mask_pred = mask_pred.reshape(new_shape)
            mask_true = mask_true.reshape(new_shape)
            edge_map_true = edge_map_true.reshape(new_shape)
            

            new_shape_image = [mask_pred.shape[0]*mask_pred.shape[1], 3, mask_pred.shape[2], mask_pred.shape[3]]
            image = image.reshape(new_shape_image)

            shuffle_indices = torch.randperm(mask_pred.shape[0])

            mask_pred = mask_pred[shuffle_indices]
            mask_true = mask_true[shuffle_indices]
            edge_map_true = edge_map_true[shuffle_indices]
            image = image[shuffle_indices]
            shuffled_name_list = [extended_name_list[i] for i in shuffle_indices]

            batch['images'] = image
            batch['masks'] = mask_true
            batch['names'] = shuffled_name_list


            batch_out['init_preds']['mask_pred'] = mask_pred.squeeze(1)
            
            mask_prompt = mask_pred

            pred_flat = mask_prompt.flatten()
            gt_flat = mask_true.flatten()

            tp = torch.logical_and(pred_flat == 1, gt_flat == 1)
            fp = torch.logical_and(pred_flat == 1, gt_flat == 0)
            tn = torch.logical_and(pred_flat == 0, gt_flat == 0)
            fn = torch.logical_and(pred_flat == 0, gt_flat == 1)

            result = torch.zeros_like(pred_flat)
            result[tp] = 0
            result[fp] = 1
            result[tn] = 2
            result[fn] = 3

            gt = result.reshape(mask_prompt.shape)

            gt_onehot = F.one_hot(gt.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()

            batch_out['assess_true_4cls'] = gt

            args.assess_model.train()
            preds, edge_pred = args.assess_model(x=image, dense_prompt=mask_prompt.float())
            pred = preds[-1]

            weight = torch.tensor([0.5, 5.0, 0.1, 5.0]).cuda() #Inira DeepLabv3+:[0.55, 5.95, 0.14, 5.33]; simulate:[0.59, 7.55, 0.13, 3.32]; CrowdAI simulate:[0.59, 8.48, 0.15, 3.29]
            loss_ce = F.cross_entropy(pred, gt.long().squeeze(1), weight=weight)
            loss_edge = HDNet_RCF_edge_criterion(edge_pred, edge_map_true)  

            predict = F.softmax(pred, dim=1)
            pos_pred = predict[:,slice(0,1),:,:] + predict[:, slice(3,4), :, :]
            neg_pred = predict[:, slice(1,2), :, :] + predict[:, slice(2,3), :, :]
            wrong_pred = predict[:, slice(1,2), :, :] + predict[:, slice(3,4), :, :]
            wrong_gt = gt.clone()
            wrong_gt[wrong_gt == 3] = 1
            wrong_gt[wrong_gt != 1] = 0
            refined_mask = mask_pred + predict[:, slice(3,4), :, :] - predict[:, slice(1,2), :, :]
            loss_rf = F.mse_loss(pos_pred, mask_true.float()) + F.mse_loss(neg_pred, (1-mask_true).float()) + F.mse_loss(refined_mask, mask_true.float())       

            loss = loss_ce + loss_edge + loss_rf

        # backward only local network
        args.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(args.optimizer)
        scaler.update()

        # args.optimizer_encoder.zero_grad()
        # loss.backward()
        # args.optimizer_detection.step()
        # args.optimizer_encoder.step()


        totalLosses.update(loss.item())
        losses_CE.update(loss_ce.item())
        losses_Edge.update(loss_edge.item())
        losses_Refine.update(loss_rf.item())
        # measure elapsed time
        batch_time.update(time.time() - start)

        #calculate metrics and record results
        # predict = F.softmax(pred, dim=1)
        predict = torch.argmax(predict, dim=1)
        batch_out['assess_preds_4cls'] = predict
        predict_onehot = F.one_hot(predict.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        stats_assess_curr = performMetrics_multiclass(predict_onehot, gt_onehot, legend_data, n_classes=4)
        stats_assess_4cls = update_stats_mulcls(stats_assess_4cls, stats_assess_curr, key='Mask')

        pred_binary = predict.clone()
        pred_binary[pred_binary == 3] = 1
        pred_binary[pred_binary != 1] = 0

        batch_out['assess_true'] = wrong_gt

        stats_assess_mask = performMetrics(pred_binary.squeeze(), wrong_gt.squeeze())
        mask_iou = stats_assess_mask['IoU']
        stats_assess = update_stats(stats_assess, stats_assess_mask, key='Mask')

        edge_predict = F.sigmoid(edge_pred)
        edge_predict[edge_predict > 0.5] = 1
        edge_predict[edge_predict <= 0.5] = 0
        batch_out['assess_preds'] = pred_binary
        batch_out['edge_pred'] = edge_predict.squeeze()
        
        # if i % int(len(args.train_loader) // 2) == 0:
        if i == 0 and epoch == 1:
            print('Visualizing [{}/{} ({:.2f}%)]'.format(i + 1, len(args.train_loader),
                                                            100.0 * (i + 1) / len(args.train_loader)))
            if dist.get_rank() == 0:
            
                visualize_Assess(batch, batch_out, legend=legend_data, vis_dir=args.vis_root, mode='train')

        if i % 10 == 0:
            print(
                'Epoch: {} [{}/{} ({:.2f}%)] | trainLoss:{:.6f}, CELoss:{:.6f}, EdgeLoss:{:.6f}, RefineLoss:{:.6f}'.format(
                    epoch, i + 1, len(args.train_loader), 100.0 * (i + 1) / len(args.train_loader), loss.item(), loss_ce.item(), loss_edge.item(), loss_rf.item()))

    
    print(
        'Train Epoch:{} | Loss: {:.4f}, CELoss:{:.4f}, EdgeLoss:{:.6f}, RefineLoss:{:.4f}'.format(
            epoch,
            totalLosses.avg, losses_CE.avg, losses_Edge.avg, losses_Refine.avg))

    summary_stats_with_dataframe(stats_assess_4cls, dataframe)
    summary_stats(stats_assess)
                    
    return train_step


def run_val(epoch, val_step, best_loss, args):
    with torch.no_grad():
        totalLosses = AverageMeter()
        losses_CE = AverageMeter()
        losses_Edge = AverageMeter()
        losses_Refine = AverageMeter()
        IoUs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        
        stats_assess_4cls, dataframe = initialize_stats(legend_data, 4)
        stats_assess = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}

        countv = 0


        for i, batch in enumerate(args.val_loader):
            val_step += 1
            names = batch['names']

            image = batch['images'].to(device)
            ori_img = batch['ori_images']
            mask_true = batch['masks'].to(device)
            unchecked_masks = batch['unchecked_masks'].to(device)
            edge_map_true = batch['edge_map'].to(device)

            batch_out = {}
            batch_out['init_preds'] = {}
            # with autocast():
            with torch.autocast(device_type='cuda', dtype=torch.float32):

                mask_pred = unchecked_masks

                batch_out['init_preds']['mask_pred'] = mask_pred.squeeze(1)

                mask_prompt = mask_pred

                pred_flat = mask_prompt.flatten()
                gt_flat = mask_true.flatten()

                tp = torch.logical_and(pred_flat == 1, gt_flat == 1)
                fp = torch.logical_and(pred_flat == 1, gt_flat == 0)
                tn = torch.logical_and(pred_flat == 0, gt_flat == 0)
                fn = torch.logical_and(pred_flat == 0, gt_flat == 1)

                result = torch.zeros_like(pred_flat)
                result[tp] = 0
                result[fp] = 1
                result[tn] = 2
                result[fn] = 3

                gt = result.reshape(mask_prompt.shape)
                gt_onehot = F.one_hot(gt.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()

                batch_out['assess_true_4cls'] = gt

                args.assess_model.eval()
                preds, edge_pred = args.assess_model(x=image, dense_prompt=mask_prompt.float())
                pred = preds[-1]

                weight = torch.tensor([0.5, 5.0, 0.1, 5.0]).cuda() #Inira DeepLabv3+:[0.55, 5.95, 0.14, 5.33]; simulate:[0.59, 7.55, 0.13, 3.32]; CrowdAI simulate:[0.59, 8.48, 0.15, 3.29], old ckpt weight:[0.63, 2.38, 0.16, 3.35]
                loss_ce = F.cross_entropy(pred, gt.long().squeeze(1), weight=weight)
                loss_edge = HDNet_RCF_edge_criterion(edge_pred, edge_map_true)  

                predict = F.softmax(pred, dim=1)
                pos_pred = predict[:,slice(0,1),:,:] + predict[:, slice(3,4), :, :]
                neg_pred = predict[:, slice(1,2), :, :] + predict[:, slice(2,3), :, :]
                wrong_pred = predict[:, slice(1,2), :, :] + predict[:, slice(3,4), :, :]
                wrong_gt = gt.clone()
                wrong_gt[wrong_gt == 3] = 1
                wrong_gt[wrong_gt != 1] = 0

                refined_mask = mask_pred + predict[:, slice(3,4), :, :] - predict[:, slice(1,2), :, :]
                batch_out['pos_pred'] = pos_pred
                batch_out['neg_pred'] = neg_pred
                batch_out['refine_mask'] = refined_mask
                loss_rf = F.mse_loss(pos_pred, mask_true.float()) + F.mse_loss(neg_pred, (1-mask_true).float()) + F.mse_loss(refined_mask, mask_true.float())       

                loss = loss_ce + loss_edge + loss_rf
                totalLosses.update(loss.item())
                losses_CE.update(loss_ce.item())
                losses_Edge.update(loss_edge.item())
                losses_Refine.update(loss_rf.item())

            #calculate metrics and record results
            predict = torch.argmax(predict, dim=1)
            batch_out['assess_preds_4cls'] = predict
            predict_sam = torch.argmax(F.softmax(preds[0], dim=1), dim=1)
            batch_out['assess_preds_sam'] = predict_sam
            predict_hq = torch.argmax(F.softmax(preds[1], dim=1), dim=1)
            batch_out['assess_preds_hq'] = predict_hq
            predict_coarse = torch.argmax(F.softmax(preds[2], dim=1), dim=1)
            batch_out['assess_preds_coarse'] = predict_coarse
            predict_onehot = F.one_hot(predict.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
            stats_assess_curr = performMetrics_multiclass(predict_onehot, gt_onehot, legend_data, n_classes=4)
            stats_assess_4cls = update_stats_mulcls(stats_assess_4cls, stats_assess_curr, key='Mask')

            mask_iou = (np.mean(stats_assess_4cls['Mask']['False Postive']['IoU']) + np.mean(stats_assess_4cls['Mask']['False Negtive']['IoU'])) / 2
            
            pred_binary = predict.clone()
            pred_binary[pred_binary == 3] = 1
            pred_binary[pred_binary != 1] = 0

            batch_out['assess_true'] = wrong_gt

            stats_assess_mask = performMetrics(pred_binary.squeeze(), wrong_gt.squeeze())
            IoUs.update(mask_iou)
            stats_assess = update_stats(stats_assess, stats_assess_mask, key='Mask')


            edge_predict = F.sigmoid(edge_pred)
            edge_predict[edge_predict > 0.5] = 1
            edge_predict[edge_predict <= 0.5] = 0
            batch_out['assess_preds'] = pred_binary
            batch_out['edge_pred'] = edge_predict.squeeze()
            countv += 1                                 
                    
            # if i % int(len(args.val_loader) // 2) == 0:
            if i % 10 == 0:
                print(
                        'Evaluating [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                            100.0 * (i + 1) / len(args.val_loader)))
        


        print(
        'Valid Epoch:{} | Loss: {:.4f}, CELoss:{:.4f}, FPLoss:{:.4f}, RefineLoss:{:.4f}'.format(
            epoch,
            totalLosses.avg, losses_CE.avg, losses_Edge.avg, losses_Refine.avg))
        
        summary_stats_with_dataframe(stats_assess_4cls, dataframe)
        summary_stats(stats_assess)

        if not args.configs['Experiment']['evaluate']:
            current_loss = [totalLosses, IoUs]  #

            args.early_stopping(current_loss, best_loss)
            print('Current FP/FN mIoU / Best IoU: {:.5f}/{:.5f}'.format(IoUs.avg, best_loss['IoU']))
            if args.early_stopping.save_model:
                print('Visualizing......')
                visualize_Assess(batch, batch_out, legend=legend_data, vis_dir=args.vis_root, mode='val')
                best_loss = save_assess_model(args, best_loss,
                                           [totalLosses, IoUs], epoch, update_best=True)
            if args.early_stopping.early_stop:
                best_loss['early_stop'] = True

            print('EarlyStopping counter: {} out of {}'.format(args.early_stopping.counter,
                                                                args.early_stopping.patience))
            if args.early_stopping.counter % 2 == 0 and args.early_stopping.counter > 0:
                print("lr from {} to {}".format(args.optimizer.state_dict()['param_groups'][0]['lr'],
                                                    args.optimizer.state_dict()['param_groups'][0]['lr'] * 0.5))
                for p in args.optimizer.param_groups:
                    p['lr'] *= 0.5

        return val_step, best_loss

def run_eval(args, vis=True, save_outs=False):
    with torch.no_grad():
        IoUs = AverageMeter()

        stats = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Boundary': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}

        
        stats_assess_4cls, dataframe = initialize_stats(legend_data, 4)
        stats_assess = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}

        for i, batch in enumerate(args.val_loader):
            names = batch['names']

            image = batch['images'].to(device)
            ori_img = batch['ori_images']
            mask_true = batch['masks'].to(device)
            unchecked_masks = batch['unchecked_masks'].to(device)

            batch_out = {}
            batch_out['init_preds'] = {}
            # with autocast():
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                mask_pred = unchecked_masks

                batch_out['init_preds']['mask_pred'] = mask_pred.squeeze(1)

                mask_prompt = mask_pred

                pred_flat = mask_prompt.flatten()
                gt_flat = mask_true.flatten()

                tp = torch.logical_and(pred_flat == 1, gt_flat == 1)
                fp = torch.logical_and(pred_flat == 1, gt_flat == 0)
                tn = torch.logical_and(pred_flat == 0, gt_flat == 0)
                fn = torch.logical_and(pred_flat == 0, gt_flat == 1)

                result = torch.zeros_like(pred_flat)
                result[tp] = 0
                result[fp] = 1
                result[tn] = 2
                result[fn] = 3

                gt = result.reshape(mask_prompt.shape)
                gt_onehot = F.one_hot(gt.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()

                batch_out['assess_true_4cls'] = gt
                args.assess_model.eval()
                preds, edge_pred = args.assess_model(x=image, dense_prompt=mask_prompt.float())
                pred = preds[-1]

                predict = F.softmax(pred, dim=1)
                pos_pred = predict[:,slice(0,1),:,:] + predict[:, slice(3,4), :, :]
                neg_pred = predict[:, slice(1,2), :, :] + predict[:, slice(2,3), :, :]
                wrong_gt = gt.clone()
                wrong_gt[wrong_gt == 3] = 1
                wrong_gt[wrong_gt != 1] = 0
                refined_mask = mask_pred + predict[:, slice(3,4), :, :] - predict[:, slice(1,2), :, :]
                batch_out['pos_pred'] = pos_pred
                batch_out['neg_pred'] = neg_pred
                batch_out['refine_mask'] = refined_mask
                

            #calculate metrics and record results
            predict = torch.argmax(predict, dim=1)
            batch_out['assess_preds_4cls'] = predict
            predict_sam = torch.argmax(F.softmax(preds[0], dim=1), dim=1)
            batch_out['assess_preds_sam'] = predict_sam
            predict_hq = torch.argmax(F.softmax(preds[1], dim=1), dim=1)
            batch_out['assess_preds_hq'] = predict_hq
            predict_coarse = torch.argmax(F.softmax(preds[2], dim=1), dim=1)
            batch_out['assess_preds_coarse'] = predict_coarse
            predict_onehot = F.one_hot(predict.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
            stats_assess_curr = performMetrics_multiclass(predict_onehot, gt_onehot, legend_data, n_classes=4)
            stats_assess_4cls = update_stats_mulcls(stats_assess_4cls, stats_assess_curr, key='Mask')

            mask_iou = (np.mean(stats_assess_4cls['Mask']['False Postive']['IoU']) + np.mean(stats_assess_4cls['Mask']['False Negtive']['IoU'])) / 2
            
            pred_binary = predict.clone()
            pred_binary[pred_binary == 3] = 1
            pred_binary[pred_binary != 1] = 0

            batch_out['assess_true'] = wrong_gt

            stats_assess_mask = performMetrics(pred_binary.squeeze(), wrong_gt.squeeze())
            IoUs.update(mask_iou)
            stats_assess = update_stats(stats_assess, stats_assess_mask, key='Mask')


            edge_predict = F.sigmoid(edge_pred)
            edge_predict[edge_predict > 0.5] = 1
            edge_predict[edge_predict <= 0.5] = 0
            batch_out['assess_preds'] = pred_binary
            batch_out['edge_pred'] = edge_predict.squeeze()                          
                    
            # if i % int(len(args.val_loader) // 2) == 0:
            if i % 10 == 0:
                print(
                    'Evaluating [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                            100.0 * (i + 1) / len(args.val_loader)))
            
            
            
            if save_outs:
                args.save_root = os.path.join(args.vis_root, "eval")
                print('Saving [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                                    100.0 * (i + 1) / len(args.val_loader)))
                mode = 'eval'
                save_Assess_infer(batch, batch_out, legend=legend_data, vis_dir=args.save_root, mode=mode)
            

        summary_stats_with_dataframe(stats_assess_4cls, dataframe)
        summary_stats(stats_assess)
        
        if vis:
            print('Visualizing......')
            visualize_Assess(batch, batch_out, legend=legend_data, vis_dir=args.vis_root, mode='eval')


def run_speedtest(args):
    with torch.no_grad():
        for i, batch in tqdm(enumerate(args.val_loader), total=len(args.val_loader), desc="Infer"):
            val_step += 1
            image = batch['images'].to(device)
            mask_true = batch['masks'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                args.assess_model.eval()
                preds, edge_preds = args.assess_model(x=image, nodes=None, node_scores=None, dense_prompt=mask_true.float())

