# Adapted from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
# Modified by Wenhu Chen

import numpy as np
import sklearn.metrics as sk


def fpr95_approx(ind_confidences, ood_confidences):
    """calculate the false positive error when tpr is 95%"""

    y1 = ood_confidences
    x1 = ind_confidences
    threshold = np.percentile(x1, 5)
    fpr_base = np.sum(np.sum(y1 > threshold)) / np.float(len(y1))

    return fpr_base, threshold


def fpr95(ind_confidences, ood_confidences):
    """calculate the falsepositive error when tpr is 95%"""

    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 100000

    if gap == 0:
        return 1

    total = 0.0
    fpr = 0.0
    threshold = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            threshold += delta
            total += 1

    if total == 0:
        print('start={start}, end={end}, gap={gap}'.format(start=start, end=end, gap=gap))
        fprBase = 1
    else:
        fprBase = fpr / total
        threshold = threshold / total

    return fprBase, threshold


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


def cal_metrics(measure_in, measure_out):
    """calculate OOD evaluation metrics"""

    # FNR@TPR95
    fpr, threshold = fpr95(measure_in, measure_out)
    print("FNR@TPR95 (higher is better): ", 1 - fpr)
    print("TPR95 threshold:", threshold)

    # detection error
    detection_error, best_threshold = detection(measure_in, measure_out)
    print("Detection error (lower is better): ", detection_error)
    print("Detection best threshold:", best_threshold)

    # create_ood_lbl
    all_mea = [np.ones_like(measure_in), np.zeros_like(measure_out)]
    in_out_lbl = np.concatenate(all_mea, axis=0)
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

    # Mean of out-dist measure
    out_mea_mean = np.mean(measure_out)

    return 1 - fpr, threshold, detection_error, best_threshold, auroc, aupr_in, aupr_out, out_mea_mean
