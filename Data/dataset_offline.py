# -*- coding: UTF-8 -*-
import os
os.environ["ALBUMENTATIONS_DISABLE_UPGRADE_CHECK"] = "1"
import cv2
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as data
import cv2 as cv
import numpy as np
import torch
from skimage import morphology, measure
import random
from glob import glob
from skimage import io

import albumentations as A

def region2boundary(mask):
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded[1:mask.shape[0] + 1, 1:mask.shape[1] + 1] = mask
    dist = cv2.distanceTransform(src=padded, distanceType=cv2.DIST_L2, maskSize=5)
    dist[dist != 1] = 0
    dist[dist == 1] = 255
    boundary_mask = dist[1:mask.shape[0] + 1, 1:mask.shape[1] + 1]
    return boundary_mask
    # dist = cv2.distanceTransform(src=mask, distanceType=cv2.DIST_L2, maskSize=5)
    # dist[dist != 1] = 0
    # dist[dist == 1] = 255
    # return dist

def read_data(filepath):
    image_path = os.path.join(filepath, 'image')
    seg_label_path = os.path.join(filepath, 'label_checked')
    if not os.path.exists(seg_label_path):
        seg_label_path = os.path.join(filepath, 'binary_map')
    unchecked_seg_path = os.path.join(filepath, 'label_uncheck')
    seg_preds_path = os.path.join(filepath, 'seg_preds')
    mask_list = os.listdir(seg_label_path)
    # if mode=='train':
    #     image_list = random.sample(image_list, 100)
    # else:
    #     image_list = random.sample(image_list, 40)

    img_lists = []
    check_lab_lists = []
    for i in range(len(mask_list)):
        mask_id = mask_list[i]
        # image_id ='austin1_1_0.tif'

        image = os.path.join(image_path, mask_id)
        if not os.path.exists(image):
            image = os.path.join(image_path, mask_id[:-4]+'.jpg')
        
        check_label = os.path.join(seg_label_path, mask_id)

        img_lists.append(image)
        check_lab_lists.append(check_label)
    return img_lists, check_lab_lists, unchecked_seg_path, seg_preds_path

class Dataset_AMS(data.Dataset):
    def __init__(self, ROOT, seg_model='ocrnet'):
        imglists, check_label_lists, uncheck_seg_dir, seg_preds_dir = read_data(ROOT)
        self.img_list = imglists
        self.check_label_lists = check_label_lists
        self.uncheck_seg_dir = uncheck_seg_dir
        self.seg_preds_dir = seg_preds_dir
        self.seg_model = seg_model

    def __getitem__(self, index):
        img_path = self.img_list[index]
        check_mask_path = self.check_label_lists[index]
        basename = os.path.basename(check_mask_path)

        name = basename.split('.')[0]

        # img = cv.imread(img_path, cv.IMREAD_COLOR)
        img = io.imread(img_path)
        mask = cv.imread(check_mask_path, cv.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        ori_img = img.copy()

        if self.seg_model == 'raw':
            uncheck_mask_path = os.path.join(self.uncheck_seg_dir, basename)
        else:
            mask_dir = os.path.join(self.seg_preds_dir, self.seg_model)
            uncheck_mask_path = os.path.join(mask_dir, name+'.tif')

        unchecked_mask = cv.imread(uncheck_mask_path, cv.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        unchecked_mask[unchecked_mask > 0] = 1
        unchecked_mask[unchecked_mask <= 0] = 0

        tmp_boundary_mask = region2boundary(mask)
        tmp_boundary_mask[tmp_boundary_mask > 0] = 1
        edge_map = np.uint8(tmp_boundary_mask)
        edge_map[edge_map > 0] = 1
        boundary_mask = cv2.GaussianBlur(tmp_boundary_mask, ksize=(3, 3), sigmaX=1, sigmaY=1)
        boundary_mask[boundary_mask > 0] = 1
        boundary_mask = np.uint8(boundary_mask)

        # plt.subplot(221)
        # plt.imshow(ori_img)
        # plt.subplot(222)
        # plt.imshow(mask)
        # plt.subplot(223)
        # plt.imshow(unchecked_mask)
        # plt.subplot(224)
        # plt.imshow(edge_map)
        # plt.savefig('check_offline_aug.png')
        # plt.show()
        
        img = np.array(img, np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        mask = mask[None,...]
        seg_preds = unchecked_mask[None,...]
        edge_map = edge_map[None,...] 
        boundary_mask = boundary_mask[None,...]

        batch = {
            'ori_images': ori_img,
            'images': img,
            'edge_map': edge_map,
            'masks': mask,
            'unchecked_masks':seg_preds,
            'names': name
        }

        return batch

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    import os

    trainset = Dataset_AMS('/data/SemanticSegmentation/CrowdAI/valid', mode='train')
    # trainset = Dataset_Inria(r'G:\Datasets\BuildingDatasets\AerialImageDataset\cropped300\valid',mode='val',N=256)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2,
        shuffle=False,
        num_workers=0, pin_memory=True)
    for i, batch in enumerate(train_loader):
        print(i)



