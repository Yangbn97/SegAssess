import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cross_entropy_loss_RCF(prediction, labelf):
    label = labelf.long()
    mask = labelf.float()
    num_positive = torch.sum((label == 1).float()).float()
    num_negative = torch.sum((label == 0).float()).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy_with_logits(
            prediction.float(),labelf.float(), weight=mask, reduction='mean')
    return torch.mean(cost)



class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.params = nn.Parameter(torch.ones(task_num))

    def forward(self, mtLoss):
        loss_sum = 0
        for i, loss in enumerate(mtLoss):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def dice_loss(y_pred, y_true, smooth=1, eps=1e-7):
    """

    @param y_pred: (N, C, H, W)
    @param y_true: (N, C, H, W)
    @param smooth:
    @param eps:
    @return: (N, C)
    """
    numerator = 2 * torch.sum(y_true * y_pred, dim=(-1, -2))
    denominator = torch.sum(y_true, dim=(-1, -2)) + torch.sum(y_pred, dim=(-1, -2))
    return 1 - (numerator + smooth) / (denominator + smooth + eps)

def HDNet_edge_criterion(inputs, target, loss_weight=torch.tensor(1), dice: bool = True):
    bcecriterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    loss = bcecriterion(inputs.squeeze(), target.squeeze().float())
    if dice is True:
        loss += dice_loss_func(torch.sigmoid(inputs.squeeze()), target.squeeze().float())
    return loss

def HDNet_RCF_edge_criterion(inputs, target):
    loss1 = cross_entropy_loss_RCF(inputs, target)
    loss2 = dice_loss_func(F.sigmoid(inputs.squeeze()), target.squeeze().float())
    return loss1 + loss2

def BinSegLoss(pred_seg, gt_seg, bce_coef=0.5, dice_coef=0.5):

    num_positive = torch.sum((gt_seg == 1).float()).float()
    num_negative = torch.sum((gt_seg == 0).float()).float()
    mask = gt_seg.clone()
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

    dice = dice_loss(F.sigmoid(pred_seg), gt_seg)
    mean_dice = torch.mean(dice)
    mean_cross_entropy = F.binary_cross_entropy_with_logits(pred_seg, gt_seg, weight=mask, reduction="mean")

    return bce_coef * mean_cross_entropy + dice_coef * mean_dice


class BinSegLoss_AQSNet(nn.Module):
    def __init__(self, bce_coef=0.5, dice_coef=0.5):

        super(BinSegLoss_AQSNet, self).__init__()
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def forward(self, pred_seg, gt_seg):

        num_positive = torch.sum((gt_seg == 1).float()).float()
        num_negative = torch.sum((gt_seg == 0).float()).float()
        mask = gt_seg.clone()
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

        dice = dice_loss(pred_seg, gt_seg)
        mean_dice = torch.mean(dice)
        mean_cross_entropy = F.binary_cross_entropy(pred_seg, gt_seg, weight=mask, reduction="mean")

        return self.bce_coef * mean_cross_entropy + self.dice_coef * mean_dice


    
if __name__ == "__main__":
    # configs = {'Model': {'class_num': 3, "class_weights": [1, 39, 8]}}
    # PM_pred = [torch.randn(32, 128, 128), torch.randn(32, 128, 128), torch.randn(32, 128, 128)]
    # PM_pred = [log_optimal_transport(s,100) for s in PM_pred]
    # PM_gt = np.random.randint(low=0,high=2,size=(32, 3, 128, 128))
    #
    # loss = geometric_mean_loss(PM_pred, PM_gt,configs)

    loss1 = torch.randn(1)
    loss2 = torch.randn(1)
    # diffusion = create_diffusion(timestep_respacing="")

    model = MultiTaskLossWrapper(task_num=2)
    # summary(model, input_size=(1, 3, 300, 300))
    # flops, params = profile(model, inputs=([loss1, loss2]))
    # macs, params_ = clever_format([flops, params], "%.3f")
    # print('MACs:', macs)
    # print('Paras:', params_)




