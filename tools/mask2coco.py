import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import pickle
from tqdm import tqdm
import math
from scipy.spatial import distance
from skimage import measure
import scipy.io as scio
import pycocotools.mask as mask_util

root = "/data02/ybn/Datasets/Multi_Class_Segmentation/LandCover.ai/cropped256/multi_class/valid/"
label_dir = os.path.join(root, "binary_map/")
image_dir = os.path.join(root, "image/")
txt_dir = root
mask = cv.imread('D:\\sample\\cropped640/labels/P0003_0_0.tif')


# mask[mask > 0] = 1


def read_txt(path, split='train'):
    txt_path = os.path.join(path, "{}.txt".format(split))
    with open(txt_path) as f:
        ids = f.readlines()
    return ids


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    polygons = []
    contoursss = []
    # pad mask to close contours of shapes which start and end at an edge
    contours, hierarchy = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        # contour = close_contour(contour)
        # contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        # contour = np.squeeze(contour)
        area = int(cv.contourArea(contour))
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
        contoursss.append(contour)
        # cv.drawContours(mask, contour, -1, (0, 255, 0), 5)
        # cv.imshow('contour', mask)
        # cv.waitKey(0)
    return polygons, contoursss


def generate_anno(label, images_info, annotations, img_name, ins_count, i):
    semantic_mask = label.copy()  # 0/category mask

    # label[label > 0] = 1
    instance_mask = label.copy()

    instance_ids = np.unique(instance_mask)  # 0/1 mask

    tif_name = i  # img_name.split('.')[0]
    imw = instance_mask.shape[1]
    imh = instance_mask.shape[0]

    has_object = False
    for instance_id in instance_ids:
        if instance_id == 0 or instance_id == 255:  # background or edge, pass
            continue

        instance_mask = label.copy()
        instance_mask[instance_mask != instance_id] = 0
        instance_mask[instance_mask == instance_id] = 1

        instance_id = 1

        # extract instance
        temp = np.zeros(instance_mask.shape)

        temp.fill(instance_id)
        tempMask = (instance_mask == temp)
        cat_id = np.max(np.unique(semantic_mask * tempMask))  # semantic category of this instance
        instance = instance_mask * tempMask
        instance_temp = instance.copy()  # findContours will change instance, so copy first
        polys, contoursss = binary_mask_to_polygon(instance)
        if len(polys) == 0:
            continue
        for (c, cs) in zip(polys, contoursss):
            # c=np.float32(c)
            area = int(cv.contourArea(cs))
            x, y, w, h = cv.boundingRect(cs)
            has_object = True
            ins_count += 1
            anno = {'segmentation': c, 'area': area,
                    'image_id': tif_name, 'bbox': [x, y, w, h],
                    'iscrowd': 0, 'category_id': int(cat_id), 'id': ins_count}

            annotations.append(anno)

    if has_object == True:
        info = {'file_name': img_name,
                'height': imh, 'width': imw, 'id': int(tif_name)}  # [:-4].replace('_', ''))}
        images_info.append(info)

    return images_info, annotations, ins_count


def save_annotations(ann, path, split='train'):
    os.system('mkdir -p {}'.format(path))
    instance_path = os.path.join(path, "coco_{}_instance.json".format(split))
    with open(instance_path, 'w') as f:
        json.dump(ann, f)


categories = [
    {'supercategory': 'none', 'id': 1, 'name': 'building'},
    {'supercategory': 'none', 'id': 2, 'name': 'woodland'},
    {'supercategory': 'none', 'id': 3, 'name': 'water'},
]


def convert_labels(ids, split):
    images = []
    annotations = []
    label_save_dir = root
    ins_count = 0
    for i in tqdm(range(len(ids))):
        # inst_path = os.path.join(instance_dir, ids[i][:-1] + '.mat')
        anno_path = os.path.join(label_dir, ids[i])
        anno = cv.imread(anno_path, cv.IMREAD_GRAYSCALE)
        images, annotations, ins_count = generate_anno(anno_path, images, annotations, ids[i], ins_count, i)
    voc_instance = {'images': images, 'annotations': annotations, 'categories': categories}
    save_annotations(voc_instance, label_save_dir, split=split)


def convert_coco():
    a = label_dir
    ids_train_noval = os.listdir(label_dir)  # read_txt(txt_dir, 'train')
    # ids_train = read_txt(txt_dir, 'train')
    # ids_val = read_txt(txt_dir, 'train')
    # ids_val5732 = []
    #
    # for id in ids_train + ids_val:
    #     if id not in ids_train_noval:
    #         ids_val5732.append(id)

    convert_labels(ids_train_noval, 'train')
    # convert_labels(ids_val5732, 'trainval', 'snake')
    # convert_labels(ids_val5732, 'val', 'mask')


if __name__ == '__main__':
    convert_coco()
