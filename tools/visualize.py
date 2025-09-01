import matplotlib
matplotlib.use('Agg')
import networkx as nx
import scipy
import torch
import numpy as np
import cv2
from PIL import Image
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage.draw import polygon2mask
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def Label2RGB(label, COLOR_MAP,start_idx=0):
    width, height = label.shape[0], label.shape[1]
    temp_mask = np.zeros(shape=(width, height, 3))
    for i, color in enumerate(COLOR_MAP):

        index = np.argwhere(label == (i + start_idx))
        for coord in index:
            temp_mask[coord[0], coord[1], :] = COLOR_MAP[i]
    return temp_mask

def visualize_Assess(batch, batch_out, legend=None, vis_dir=None, mode='train', title=None):
    vis_dir = os.path.join(vis_dir, mode)
    os.makedirs(vis_dir, exist_ok=True)

    if mode == 'eval':
        eval_dir = os.path.join(vis_dir, 'vis')
        os.makedirs(eval_dir, exist_ok=True)
        infer_dir = os.path.join(vis_dir, 'infer')
        os.makedirs(infer_dir, exist_ok=True)
        gt_dir = os.path.join(vis_dir, 'gt')
        os.makedirs(gt_dir, exist_ok=True)
        
    mask = batch['masks']
    ori_img = batch['ori_images']
    images = batch['images']
    names = batch['names']

    init_mask_pred = batch_out['init_preds']['mask_pred']
    
    assess_true = batch_out['assess_true']
    assess_true_4cls = batch_out['assess_true_4cls']
    assess_preds = batch_out['assess_preds']
    assess_preds_4cls = batch_out['assess_preds_4cls']
    edge_preds = batch_out['edge_pred']

    # BN = min(10, heatmap.shape[0])
    BN = mask.shape[0]
    for n in range(0, BN):
        mask_gt = mask.squeeze(1).detach().cpu().numpy()[n, ...]
        assess_gt = assess_true.squeeze(1).detach().cpu().numpy()[n, ...]
        assess_gt_4cls = assess_true_4cls.squeeze(1).detach().cpu().numpy()[n, ...].astype(np.uint8)

        if np.sum(mask_gt) == 0:
            continue
        
        # img = ori_img.detach().cpu().numpy()[n, ...].astype(np.uint8)
        image = images.detach().cpu().numpy()[n, ...] * 255
        img = image.transpose([1,2,0]).astype(np.uint8)

        init_mask_predict = init_mask_pred.detach().cpu().numpy()[n, ...].astype(np.uint8)

        contours_true, _ = cv2.findContours(mask_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(init_mask_predict, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (image * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        image_with_contours = image_bgr.copy()
        cv2.drawContours(image_with_contours, contours_true, -1, (255, 0, 255), 2) 
        cv2.drawContours(image_with_contours, contours_pred, -1, (0, 165, 255), 2) 

        assess_pred_index_map = assess_preds.detach().cpu().numpy()[n, ...].astype(np.uint8)
        assess_pred_index_map_4cls = assess_preds_4cls.detach().cpu().numpy()[n, ...].astype(np.uint8)
        edge_pred = edge_preds.detach().cpu().numpy()[n, ...].astype(np.uint8)
        color_mapping = np.array([l[0] for l in legend],dtype=np.uint8)
        assess_gt_4cls = color_mapping[assess_gt_4cls]
        assess_pred_index_map_4cls = color_mapping[assess_pred_index_map_4cls]
        if mode == 'eval':     
            gt_path = os.path.join(gt_dir, f'{names[n]}.tif')
            infer_path = os.path.join(infer_dir, f'{names[n]}.tif')
            img_gt = Image.fromarray(assess_gt_4cls)
            img_pred = Image.fromarray(assess_pred_index_map_4cls)
            img_gt.save(gt_path)
            img_pred.save(infer_path)
        
        color_list = np.array([l[0] for l in legend])/255
        color_list = ListedColormap(color_list)
        handles = [
            Rectangle((0, 0), 1, 1, color=tuple((v / 255 for v in c[0]))) for c in legend
        ]
        labels = [c[1] for c in legend]

        plt.figure()
        plt.subplot(231)
        plt.title('Image')
        plt.imshow(image_with_contours)
        plt.axis('off')
        plt.subplot(232)
        plt.title('Binary Assess True')
        plt.imshow(assess_gt)
        plt.axis('off')
        plt.subplot(233)
        plt.title('Binary Assess Pred')
        plt.imshow(assess_pred_index_map)
        plt.axis('off')
        plt.subplot(234)
        plt.title('4cls Assess True')
        plt.imshow(assess_gt_4cls)
        plt.axis('off')
        plt.subplot(235)
        plt.title('4cls Assess Pred')
        plt.imshow(assess_pred_index_map_4cls)
        plt.axis('off')
        plt.subplot(236)
        plt.title('Edge Pred')
        plt.imshow(edge_pred)
        plt.axis('off')
        plt.figlegend(handles, labels, ncol=3, loc='lower center')
        img_save_path = os.path.join(vis_dir, '{}_{}.png'.format(n, names[n]))
        if mode == 'eval':
            img_save_path = os.path.join(eval_dir, f'{names[n]}.png')
        plt.suptitle(title)
        plt.savefig(img_save_path, bbox_inches='tight')
        # plt.show()g_save_path, bbox_inches='tight')
        # plt.show()

def save_Assess_infer(batch, batch_out, legend=None, vis_dir=None, mode='train', title=None):
    os.makedirs(vis_dir, exist_ok=True)
    eval_dir = os.path.join(vis_dir, 'vis')
    os.makedirs(eval_dir, exist_ok=True)
    infer_dir = os.path.join(vis_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)
    gt_dir = os.path.join(vis_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    im_dir = os.path.join(vis_dir, 'im')
    os.makedirs(im_dir, exist_ok=True)
        
    mask = batch['masks']
    ori_img = batch['images']
    edge_map_true = batch['edge_map']
    names = batch['names']

    init_mask_pred = batch_out['init_preds']['mask_pred']
    
    assess_true_4cls = batch_out['assess_true_4cls']
    assess_preds_refined = batch_out['assess_preds_4cls']
    assess_preds_sam = batch_out['assess_preds_sam'] 
    assess_preds_hq = batch_out['assess_preds_hq']
    assess_preds_coarse = batch_out['assess_preds_coarse']

    # BN = min(10, heatmap.shape[0])
    BN = mask.shape[0]
    for n in range(0, BN):
        mask_gt = mask.squeeze(1).detach().cpu().numpy()[n, ...]
        edge_gt = edge_map_true.squeeze(1).detach().cpu().numpy()[n, ...]
        assess_gt_4cls = assess_true_4cls.squeeze(1).detach().cpu().numpy()[n, ...].astype(np.uint8)

        image = ori_img.detach().cpu().numpy()[n, ...] * 255
        img = image.transpose([1,2,0]).astype(np.uint8)

        init_mask_predict = init_mask_pred.detach().cpu().numpy()[n, ...].astype(np.uint8)
        
        contours_true, _ = cv2.findContours(mask_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(init_mask_predict, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (image * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        image_with_contours = image_bgr.copy()
        cv2.drawContours(image_with_contours, contours_true, -1, (255, 0, 255), 2)  
        cv2.drawContours(image_with_contours, contours_pred, -1, (0, 165, 255), 2)  

        assess_pred_refined = assess_preds_refined.detach().cpu().numpy()[n, ...].astype(np.uint8)
        assess_pred_sam = assess_preds_sam.detach().cpu().numpy()[n, ...].astype(np.uint8)
        assess_pred_hq = assess_preds_hq.detach().cpu().numpy()[n, ...].astype(np.uint8)
        assess_pred_coarse = assess_preds_coarse.detach().cpu().numpy()[n, ...].astype(np.uint8)

        color_mapping = np.array([l[0] for l in legend],dtype=np.uint8)
        assess_gt_4cls = color_mapping[assess_gt_4cls]
        assess_pred_refined = color_mapping[assess_pred_refined]
        assess_pred_sam = color_mapping[assess_pred_sam]
        assess_pred_hq = color_mapping[assess_pred_hq]
        assess_pred_coarse = color_mapping[assess_pred_coarse]
        img_mask_path = os.path.join(im_dir, f'im_{names[n]}.png')

        refined_path = os.path.join(infer_dir, f'{names[n]}.png')
        cv2.imwrite(img_mask_path, image_with_contours)
        img_refined = Image.fromarray(assess_pred_refined)
        img_refined.save(refined_path)



        mask_RGB = np.expand_dims(init_mask_predict, axis=-1)
        overlay = np.where(mask_RGB == 1, [0, 255, 0], img.copy())
   
        color_list = np.array([l[0] for l in legend])/255
        color_list = ListedColormap(color_list)
        handles = [
            Rectangle((0, 0), 1, 1, color=tuple((v / 255 for v in c[0]))) for c in legend
        ]
        labels = [c[1] for c in legend]

        plt.figure()
        plt.subplot(231)
        plt.title('Image')
        plt.imshow(overlay)
        plt.axis('off')
        plt.subplot(232)
        plt.title('Assess SAM')
        plt.imshow(assess_pred_sam)
        plt.axis('off')
        plt.subplot(233)
        plt.title('Assess HQ')
        plt.imshow(assess_pred_hq)
        plt.axis('off')
        plt.subplot(234)
        plt.title('Assess Coarse')
        plt.imshow(assess_pred_coarse)
        plt.axis('off')
        plt.subplot(235)
        plt.title('Assess Refined')
        plt.imshow(assess_pred_refined)
        plt.axis('off')
        plt.subplot(236)
        plt.title('Assess True')
        plt.imshow(assess_gt_4cls)
        plt.axis('off')
        plt.figlegend(handles, labels, ncol=3, loc='lower center')
        img_save_path = os.path.join(eval_dir, f'{names[n]}.png')
        plt.suptitle(title)
        plt.savefig(img_save_path, bbox_inches='tight')
        # plt.show()

def visualize_segmetation_out(batch, batch_out, epoch, vis_dir=None, mode='train'):
    vis_dir = os.path.join(vis_dir, mode)
    # vis_dir = os.path.join(r'D:\Water_Woodland_Extraction\SkelNet_v3\trainedModels\vis\local\road', mode)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
    infer_dir = os.path.join(vis_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    segmap_gt = batch['masks']
    boundary_mask_gt = batch['boundary_map']
    ori_img = batch['ori_images']
    names = batch['names']
    segmap_pred = batch_out['mask_pred']
    segmap_pred_simplified = batch_out['mask_pred_simplified']
    boundary_mask_pred = batch_out['boundary_mask_pred']
    prob_map = batch_out['probability_map']

    # BN = min(10, segmap_gt.shape[0])
    BN = segmap_gt.shape[0] if mode == 'eval' else min(5, segmap_gt.shape[0])
    for n in range(0, BN):
        mask_predict = segmap_pred.squeeze(1).cpu().detach().numpy()[n, ...]
        mask_pred_simplified = segmap_pred_simplified.squeeze(1).cpu().detach().numpy()[n, ...]
        mask_true = segmap_gt.squeeze(1).cpu().numpy()[n, ...]

        mask_predict_ = np.uint8(mask_predict) * 255
        mask_pred_simplified_ = np.uint8(mask_pred_simplified) * 255
        mask_true_ = np.uint8(mask_true) * 255

        if np.sum(mask_true_) == 0:
            continue

        # kernel = np.ones((7, 7), np.uint8)
        # mask_predict_ = cv2.dilate(mask_predict_, kernel)
        # mask_predict_ = cv2.morphologyEx(mask_predict_, cv2.MORPH_CLOSE, kernel)

        polys_pred, _ = cv2.findContours(mask_predict_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        polys_pred_simplified, _ = cv2.findContours(mask_predict_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polys_true, _ = cv2.findContours(mask_true_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        boundary_mask_predict = boundary_mask_pred.squeeze(1).cpu().detach().numpy()[n, ...]
        boundary_mask_true = boundary_mask_gt.squeeze(1).cpu().numpy()[n, ...]

        prob_map_pred = prob_map.squeeze(1).cpu().detach().numpy()[n, ...]

        image = ori_img.numpy()[n, ...].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.drawContours(image, polys_true, -1, (0, 0, 255), 2)
        image = cv2.drawContours(image, polys_pred, -1, (0, 0, 255), 1)
        image = cv2.drawContours(image, polys_pred_simplified, -1, (0, 255, 0), 1)

        # for pt in points_pred:
        #     cv2.circle(img_show_pred, (int(pt[1]), int(pt[0])), 2, (255, 0, 0), -1)
        # plt.imshow(img_show_pred)
        # plt.show()
        if mode == 'eval':
            infer_save_path = os.path.join(infer_dir, '{}.png'.format(names[n]))
            cv2.imwrite(infer_save_path, image)
        img_save_path = os.path.join(vis_dir, 'E{}_{}.png'.format(epoch, names[n]))
        plt.figure()
        plt.subplot(231)
        plt.imshow(image)
        plt.title('image')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(mask_true*255)
        plt.title('mask_true')
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(mask_predict_)
        plt.title('mask_pred')
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(boundary_mask_true*255)
        plt.title('boundary_true')
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(boundary_mask_predict*255)
        plt.title('boundary_pred')
        plt.subplot(236)
        plt.imshow(prob_map_pred)
        plt.title('prob_map')
        plt.axis('off')

        plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
        # plt.show()