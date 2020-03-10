# Modified from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
# https://github.com/yenchanghsu/out-of-distribution-detection/blob/master/utils/metric.py

import numpy as np
import sklearn.metrics as sk
import time

import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k * 100.0 / batch_size)

        if len(res) == 1:
            return res[0]
        else:
            return res


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
        self.avg = float(self.sum) / self.count


class Timer(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


def tnr_at_tpr95(ind_confidences, ood_confidences):
    # calculate the falsepositive error when tpr is 95%
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 1000000

    print(start)
    print(end)
    print(gap)

    if gap == 0:
        return 1

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1

    if total == 0:
        print('start={start}, end={end}, gap={gap}'.format(start=start, end=end, gap=gap))
        fprBase = 1
    else:
        fprBase = fpr / total

    return 1 - fprBase


def detection(ind_confidences, ood_confidences, n_iter=100000, return_data=False):
    """calculate the minimum detection error"""

    y1 = ood_confidences
    x1 = ind_confidences

    start = np.min([np.min(x1), np.min(y1)])
    end = np.max([np.max(x1), np.max(y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(x1 < delta)) / np.float(len(x1))
        error2 = np.sum(np.sum(y1 > delta)) / np.float(len(y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta


def cal_ood_metrics(measure_in, measure_out):
    """calculate OOD evaluation metrics"""

    # FNR@TPR95
    fnr = tnr_at_tpr95(measure_in, measure_out)
    print("FNR@TPR95 (higher is better): ", fnr)
    # print("TPR95 threshold:", threshold)

    # detection error
    detection_error, best_threshold = detection(measure_in, measure_out)
    print("Detection error (lower is better): ", detection_error)
    print("Detection best threshold:", best_threshold)

    # create_ood_lbl
    in_out_lbl = np.concatenate([np.ones_like(measure_in), np.zeros_like(measure_out)], axis=0)
    # create measure_all
    measure_all = np.concatenate([measure_in, measure_out])

    # AUROC
    auroc = sk.roc_auc_score(in_out_lbl, measure_all)
    print("AUROC (higher is better): ", auroc)

    # aupr in
    aupr_in = sk.average_precision_score(in_out_lbl, measure_all)
    print("AUPR_IN (higher is better): ", aupr_in)

    # aupr out
    aupr_out = sk.average_precision_score((in_out_lbl - 1) * -1, measure_all * -1)
    print("AUPR_OUT (higher is better): ", aupr_out)

    # # Mean of out-dist measure
    # out_mea_mean = np.mean(measure_out)

    return fnr, detection_error, best_threshold, auroc, aupr_in, aupr_out
