# -*- coding: UTF-8 -*-
import os

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
    label_path = os.path.join(filepath, 'binary_map')
    mask_list = os.listdir(label_path)
    # if mode=='train':
    #     total_num = len(image_list)
    #     sample_num = int(0.05*total_num)
    #     image_list = image_list[:sample_num]
    # else:
    #     image_list = image_list

    
    img_lists = []
    lab_lists = []
    # pointmap_lists = []
    for i in range(len(mask_list)):
        label_id = mask_list[i]
        # image_id ='austin1_1_0.tif'
        label = os.path.join(label_path, label_id)
        if not os.path.exists(label):
                    label = os.path.join(label_path, label_id[:-4]+'.tif')

        image_id = label_id
        image = os.path.join(image_path, image_id)
        if not os.path.exists(image):
            image = os.path.join(image_path, image_id[:-4]+'.jpg')
        
        img_lists.append(image)
        lab_lists.append(label)
    return img_lists, lab_lists



class Dataset_base(data.Dataset):

    def __init__(self, ROOT, mode='train',N=256, dilate=5):
        imglists, labellists = read_data(ROOT, mode)
        self.img_list = imglists
        self.mask_list = labellists
        self.mode = mode
        self.dilate_pixels = dilate
        self.N = N

    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        # pointmap_path = self.pointmap_list[index]
        basename = os.path.basename(img_path)

        name = basename.split('.')[0]

        # img = cv.imread(img_path, cv.IMREAD_COLOR)
        img = io.imread(img_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()
        # img = cv.resize(img, (512, 512), interpolation=cv.INTER_NEAREST)
        # mask = cv.resize(mask, (512, 512), interpolation=cv.INTER_NEAREST)

        ori_img = img.copy()

        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()


        img_show = ori_img

        # pointmap, contours, contours_sampled = mask2nodes(mask, self.N, init_stride=20)
        # heatmap = generate_heatmap(pointmap, size=3)

        # img_show = img.copy()
        img = np.array(img, np.float32) / 255.0
        # img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        tmp_boundary_mask = region2boundary(mask)
        tmp_boundary_mask[tmp_boundary_mask > 0] = 1
        edge_map = np.uint8(tmp_boundary_mask)
        edge_map[edge_map > 0] = 1
        boundary_mask = cv2.GaussianBlur(tmp_boundary_mask, ksize=(3, 3), sigmaX=1, sigmaY=1)
        boundary_mask[boundary_mask > 0] = 1
        boundary_mask = np.uint8(boundary_mask)

        # iou_max = 1.0
        # iou_min = 0.8
        # iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
        # currpted_mask = modify_boundary(mask, iou_target=iou_target)
        # currpted_mask[currpted_mask > 0] = 1
        # currpted_mask[currpted_mask != 1] = 0
        # heatmap_ = heatmap * boundary_mask
        # norm_img = np.zeros(heatmap.shape)
        # heatmap_ = heatmap.astype(np.float32)
        # norm_img = cv2.normalize(heatmap_, norm_img, 0, 255, cv2.NORM_MINMAX)
        # norm_img = np.asarray(norm_img, dtype=np.uint8)
        #
        # norm_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        # # norm_img = cv2.cvtColor(norm_img,cv2.COLOR_BGR2RGB)
        # image = np.ascontiguousarray(img_show[:, :, [2, 1, 0]])
        # img_add = cv2.addWeighted(image, 0.7, norm_img, 0.3, 0)
        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(img_show)
        # plt.title('image')
        # plt.axis('off')
        # plt.subplot(222)
        # plt.title('mask')
        # plt.imshow(mask)
        # plt.axis('off')
        # plt.subplot(223)
        # plt.title('boundary_mask')
        # plt.imshow(boundary_mask)
        # plt.axis('off')
        # plt.subplot(224)
        # plt.title('currpted_mask')
        # plt.imshow(currpted_mask)
        # plt.axis('off')
        # plt.savefig('check_data.png')
        # plt.show()

        # batch = {
        #     'ori_img': ori_img,
        #     'img': img,
        #     'heatmap': heatmap,
        #     'edge_map': edge_map,
        #     'boundary_mask': boundary_mask,
        #     'segmap': mask,
        #     'name': name
        # }
        batch = {
            'ori_images': ori_img,
            'images': img,
            'heatmaps': mask[None, :, :],
            'edge_map': edge_map[None, :, :],
            'boundary_map': boundary_mask[None, :, :],
            'masks': mask[None, :, :],
            'currpted_mask':mask[None, :, :],
            'names': name
        }

        return batch

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    import os

    trainset = Dataset_base('/data02/ybn/Datasets/Building/Inria/cropped300/train', mode='train',N=320)
    # trainset = Dataset_Inria(r'G:\Datasets\BuildingDatasets\AerialImageDataset\cropped300\valid',mode='val',N=256)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2,
        shuffle=False,
        num_workers=0, pin_memory=True)
    for i, batch in enumerate(train_loader):
        print(i)
        # pointmap, anglemap, keypoints = get_topoData(mask, img)
        # print(idx, img.shape)

# pre_path1 = '/home/wyj/Downloads/project/road_extraction/submits/our_s_30/'
# pre_path2 = '/home/wyj/Downloads/project/road_extraction/submits/ours_s_30_seg/'
#
# files = os.listdir(pre_path1)
#
# for file in files:
#     pre_img1 = np.array(cv.imread(pre_path1 + file ,cv.IMREAD_GRAYSCALE),np.float32)
#     pre_img2 = np.array(cv.imread(pre_path2 + file, cv.IMREAD_GRAYSCALE),np.float32)
#
#     pre_img1[pre_img1 <=128] = 0
#     pre_img1[pre_img1 >128] = 1
#     pre_img2[pre_img2 <= 128] = 0
#     pre_img2[pre_img2 > 128] = 1
#
#     pre_img = (pre_img1 + pre_img2) / 2
#     pre_img[pre_img>=0.5] = 255
#     pre_img[pre_img<0.5] = 0
#     cv.imwrite('/home/wyj/Downloads/project/road_extraction/submits/ours_s_30_fusion/'+file, np.array(pre_img, np.uint8))
