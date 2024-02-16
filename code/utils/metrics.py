#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
import torch
from medpy import metric


class DiceScoreStorer(object):
    """
    store dice score of each patch,
    seperate pos and neg patches,
    """

    def __init__(self, sigmoid=False, thresh=0.5, eps=1e-6):
        self.array = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.eps = eps
        self.sigmoid = sigmoid
        self.thresh = thresh

    def __len__(self):
        return len(self.array)

    def update(self, pred_mask, gt_mask):
        N = pred_mask.size(0)
        dice_scores = self._dice_score(pred_mask, gt_mask)
        dice_scores = list(dice_scores.detach().cpu().numpy())
        self.array = self.array + dice_scores
        self.count += N
        self.sum += sum(dice_scores)
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.array[-1]

    def _dice_score(self, preds, gt):
        if self.sigmoid:
            preds = (torch.sigmoid(preds) > self.thresh).type(gt.type())
        else:
            preds = torch.softmax(preds, dim=1)
            preds = preds.max(axis=1)[1].unsqueeze(1)

        preds = preds.view(preds.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        intersect = gt * preds

        return (2.0 * intersect.sum(1).float() + self.eps) / (preds.sum(1).float() + gt.sum(1).float() + self.eps)


class IoUStorer(object):
    """
    store dice score of each patch,
    seperate pos and neg patches,
    """

    def __init__(self, sigmoid=False, thresh=0.5, eps=1e-6):
        self.array = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.sigmoid = sigmoid
        self.thresh = thresh
        self.eps = eps

    def __len__(self):
        return len(self.array)

    def update(self, pred_mask, gt_mask):
        N = pred_mask.size(0)
        iou = self._iou(pred_mask, gt_mask)
        iou = list(iou.detach().cpu().numpy())
        self.array = self.array + iou
        self.count += N
        self.sum += sum(iou)
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.array[-1]

    def _iou(self, preds, gt):
        if self.sigmoid:
            preds = (torch.sigmoid(preds) > self.thresh).type(gt.type())

        else:
            preds = torch.softmax(preds, dim=1)
            preds = preds.max(axis=1)[1].unsqueeze(1)
        preds = preds.view(preds.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        intersect = gt * preds
        union = ((gt + preds) > 0).type(intersect.type())
        return (intersect.sum(1).float() + self.eps) / (union.sum(1).float() + self.eps)

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)