import torch
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
import shapely
from skimage import measure
import cv2
import pandas as pd
from pandas import Series


def get_confusion_matrix_with_counter(label, predict, class_num=2):
    confu_list = []
    for i in range(class_num):
        c = Counter(label[np.where(predict == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int64)

def metrics(confu_mat_total):
    '''
    confu_mat_total: 总的混淆矩阵
    keep_background：是否干掉背景
    '''

    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float64) + 1e-9

    col_sum = np.sum(confu_mat, axis=1)
    raw_sum = np.sum(confu_mat, axis=0)

    '''计算各类面积比，以求OA值'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FP = col_sum - TP
    FN = raw_sum - TP

    # 计算并写出precision，recall, IOU
    precision = TP / col_sum
    recall = TP / raw_sum
    f1 = 2 * (precision * recall) / (precision + recall)
    iou = TP / (TP + FP + FN)

    return oa, precision, recall, f1, iou



def performMetrics(pred, true, n_classes=2):
    pred = pred.cpu().numpy()
    pred[pred > 0] = 1
    true = true.cpu().numpy()
    true[true > 0] = 1
    confu_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(pred.shape[0]):
        confu_matrix += get_confusion_matrix_with_counter(true[i], pred[i], class_num=n_classes)
    oa, precision, recall, f1, iou = metrics(confu_matrix)

    stats = {
        'Pixel Accuracy': oa * 100,
        'Precision': np.nanmean(precision) * 100,
        'Recall': np.nanmean(recall) * 100,
        'F1-score': np.nanmean(f1) * 100,
        'IoU': np.nanmean(iou) * 100
    }

    return stats

def performMetrics_multiclass(pred, true, legend_data, n_classes=6):
    pred = pred.detach().cpu().numpy()
    pred[pred > 0] = 1
    true = true.detach().cpu().numpy()
    true[true > 0] = 1
    OA = []
    Precision = []
    Recall = []
    F1 = []
    IoU = []
    stats = {}
    for cls in range(n_classes):
        pred_cls = pred[:, cls, :, :]
        true_cls = true[:, cls, :, :]
        confu_matrix = np.zeros((2, 2), dtype=np.int64)
        for i in range(pred.shape[0]):
            confu_matrix += get_confusion_matrix_with_counter(true_cls[i], pred_cls[i], class_num=2)
        oa, precision, recall, f1, iou = metrics(confu_matrix)
        OA.append(oa*100)
        Precision.append(precision[1] * 100)
        Recall.append(recall[1] * 100)
        F1.append([f1[1] * 100])
        IoU.append(iou[1] * 100)

        stats_cls = {
            'Pixel Accuracy': oa * 100,
            'Precision': precision[1] * 100,
            'Recall': recall[1] * 100,
            'F1-score': f1[1] * 100,
            'IoU': iou[1] * 100
        }
        stats[legend_data[cls][1]] = stats_cls

    stats['Mean'] = {
        'Pixel Accuracy': np.nanmean(OA),
        'Precision': np.nanmean(Precision),
        'Recall': np.nanmean(Recall),
        'F1-score': np.nanmean(F1),
        'IoU': np.nanmean(IoU)
    }
    return stats

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def update_acc(recall, precision, b_recall, b_precision):
    for k, v in recall.items():
        if isinstance(b_recall[k], list):
            continue
        recall[k].append(b_recall[k])
    for k, v in precision.items():
        if isinstance(b_precision[k], list):
            continue
        precision[k].append(b_precision[k])
    return recall, precision


def update_stats(stats, stats_batch, key='Mask'):
    for k, v in stats[key].items():
        stats[key][k].append(stats_batch[k])
    return stats

def update_stats_mulcls(stats, stats_batch, key='Mask'):
    for k, v in stats[key].items():
        for k_cls, v_cls in v.items():
            stats[key][k][k_cls].append(stats_batch[k][k_cls])
    return stats


def initialize_stats(legend, class_num):
    # stats = {'Mask': {},
    #          'Boundary': {}}
    stats = {'Mask': {}}
    dataframe = get_dataframe(legend)
    for ind in range(class_num):
        legend = legend
        cls_name = legend[ind][1]
        stats['Mask'][cls_name] = {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}
        # stats['Boundary'][cls_name] = {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}
    return stats, dataframe


def get_dataframe(legend_data, metrics=None):
    if metrics is None:
        metrics = {'Pixel Accuracy': [0], 'Precision': [0], 'Recall': [0], 'F1-score': [0], 'IoU': [0]}
    row_idx = []
    for d in legend_data:
        row_idx.append(d[1])
    row_idx.append('Mean')
    row_idx = Series(row_idx)
    metrics_table = pd.DataFrame(metrics, index=row_idx)
    # for k, v in metrics:
    #     metrics_table[k] = metrics_table[k].astype('object')
    return metrics_table

def summary_stats(stats):
    for k, v in stats.items():
        print('------', k, '------')
        for key, value in stats[k].items():
            assert isinstance(value, list)
            value = [i for i in value if isinstance(i, float)]

            print(str(key), ':', str(np.nanmean(value)))

def summary_stats_with_dataframe(stats, dataframe, metrics=None):
    if metrics is None:
        metrics = ['Pixel Accuracy', 'Precision', 'Recall', 'F1-score', 'IoU']
    for k, v in stats.items():
        df = dataframe.copy()
        print('-----------------------------', k, '-----------------------------')
        for key, value in v.items():

            for k_cls, v_cls in value.items():
                assert isinstance(v_cls, list)
                v_cls = [i for i in v_cls if isinstance(i, float)]
                df.loc[key, k_cls] = np.nanmean(v_cls)

        for col in metrics:
            df.loc['Mean', col] = df[col].iloc[range(len(v))].mean()
        pd.set_option('expand_frame_repr', False)
        print(df)
