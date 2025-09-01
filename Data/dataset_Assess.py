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

def read_data(filepath, mode='train'):
    image_path = os.path.join(filepath, 'image')
    seg_label_path = os.path.join(filepath, 'binary_map')
    unchecked_seg_path = os.path.join(filepath, 'seg_preds')
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
        check_label = os.path.join(seg_label_path, mask_id)

        image = os.path.join(image_path, mask_id)
        if not os.path.exists(image):
            image = os.path.join(image_path, mask_id[:-4]+'.jpg')
        
        img_lists.append(image)
        check_lab_lists.append(check_label)
    return img_lists, check_lab_lists, unchecked_seg_path

class Dataset_Assess(data.Dataset):
    def __init__(self, ROOT, mode='train', seg_models=['deeplabv3+', 'hrnet', 'transunet', 'unetformer', 'ocrnet']):
        imglists, check_label_lists, uncheck_seg_dir = read_data(ROOT, mode)
        self.img_list = imglists
        self.check_label_lists = check_label_lists
        self.uncheck_seg_dir = uncheck_seg_dir
        self.mode = mode
        if self.mode == 'train':
            self.seg_models = ['deeplabv3+', 'hrnet', 'transunet', 'unetformer']
        else:
            self.seg_models = ['ocrnet']

        # 动态生成additional_targets字典
        self.additional_targets = {f'seg_pred_{i}': 'mask' for i in range(len(seg_models))}
        
        # 空间变换（同步应用到image/mask/单个seg_pred）
        self.spatial_aug = A.Compose([
            A.Rotate(limit=(90, 270), steps=3, interpolation=cv2.INTER_NEAREST, 
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.1
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.1
            ),
        ], additional_targets=self.additional_targets)  # 动态配置所有seg_pred

        # 仅图像增强
        self.image_only_aug = A.Compose([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, 
                               val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ])

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

        if self.mode == 'train':
            # 读取所有模型的预测结果
            seg_preds = []
            for m_name in self.seg_models:
                mask_dir = os.path.join(self.uncheck_seg_dir, m_name)
                mask_path = os.path.join(mask_dir, name+'.tif')
                seg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                seg = (seg > 0).astype(np.uint8)  # 二值化处理
                seg_preds.append(seg)
                # plt.subplot(131)
                # plt.imshow(ori_img)
                # plt.subplot(132)
                # plt.imshow(mask)
                # plt.subplot(133)
                # plt.imshow(seg)
                # plt.savefig('check_offline_aug.png')

            # 创建基础数据字典
            base_data = {"image": img, "mask": mask}
            
            # 为每个seg_pred生成独立增强
            augmented_ori_images = []
            augmented_images = []
            augmented_masks = []
            augmented_preds = []
            augmented_edge_masks = []
            augmented_bdy_masks = []
            
            for i, seg_pred in enumerate(seg_preds):
                # 创建当前seg_pred的数据字典
                current_data = {
                    "image": base_data["image"].copy(),
                    "mask": base_data["mask"].copy(),
                    f"seg_pred_{i}": seg_pred.copy()
                }
                
                # 应用空间变换（包含当前seg_pred）
                spatial_data = self.spatial_aug(**current_data)
                
                current_ori_img = spatial_data["image"]
                # 应用仅图像增强
                image_aug = self.image_only_aug(image=spatial_data["image"])["image"]
                
                # 获取当前seg_pred的增强结果
                current_mask = spatial_data["mask"]
                current_seg_pred = spatial_data[f"seg_pred_{i}"]
                
                image_aug = np.array(image_aug, np.float32) / 255.0
                image_aug = image_aug.transpose(2, 0, 1) 

                # 收集结果          
                tmp_boundary_mask = region2boundary(current_mask)
                tmp_boundary_mask[tmp_boundary_mask > 0] = 1
                edge_map = np.uint8(tmp_boundary_mask)
                edge_map[edge_map > 0] = 1
                boundary_mask = cv2.GaussianBlur(tmp_boundary_mask, ksize=(3, 3), sigmaX=1, sigmaY=1)
                boundary_mask[boundary_mask > 0] = 1
                boundary_mask = np.uint8(boundary_mask)

                augmented_ori_images.append(current_ori_img)
                augmented_images.append(image_aug)
                augmented_masks.append(current_mask)
                augmented_preds.append(current_seg_pred)
                augmented_edge_masks.append(edge_map)
                augmented_bdy_masks.append(boundary_mask)

                # plt.subplot(221)
                # plt.imshow(current_ori_img)
                # plt.subplot(222)
                # plt.imshow(current_mask)
                # plt.subplot(223)
                # plt.imshow(current_seg_pred)
                # plt.subplot(224)
                # plt.imshow(edge_map)
                # plt.savefig('check_offline_aug.png')
                # plt.show()

            ori_img = np.stack(augmented_ori_images)
            img = np.stack(augmented_images, axis=0)
            mask = np.stack(augmented_masks, axis=0)
            seg_preds = np.stack(augmented_preds, axis=0)
            edge_map = np.stack(augmented_edge_masks, axis=0)
            boundary_map = np.stack(augmented_bdy_masks, axis=0)
        else:
            mask_dir = os.path.join(self.uncheck_seg_dir, self.seg_models[0])
            uncheck_mask_path = os.path.join(mask_dir, name+'.tif')
            if self.seg_models[0] == 'raw':
                uncheck_mask_path = check_mask_path.replace('binary_map','label_uncheck')
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
            boundary_map = boundary_mask[None,...]
        
        batch = {
            'ori_images': ori_img,
            'images': img,
            'edge_map': edge_map,
            'boundary_map':boundary_map,
            'masks': mask,
            'unchecked_masks':seg_preds,
            'names': name
        }

        return batch

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    import os

    trainset = Dataset_Assess('/data/SemanticSegmentation/CrowdAI/valid', mode='train')
    # trainset = Dataset_Inria(r'G:\Datasets\BuildingDatasets\AerialImageDataset\cropped300\valid',mode='val',N=256)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2,
        shuffle=False,
        num_workers=0, pin_memory=True)
    for i, batch in enumerate(train_loader):
        print(i)



