# modified from https://github.com/learningtitans/isbi2017-part1/blob/master/metrics.py


import numpy as np
from sklearn.metrics import jaccard_similarity_score

smooth_default = 1.


def dice_coef(y_true, y_pred, smooth=smooth_default):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    tp = np.sum(y_true_f * y_pred_f)
    tn = np.sum(y_true_f == y_pred_f) - tp

    return (2. * tp + smooth) / (len(y_true_f) - tn + tp + smooth)


def jacc_idx(y_true, y_pred, smooth=smooth_default):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    tp = np.sum(y_true_f * y_pred_f)
    tn = np.sum(y_true_f == y_pred_f) - tp

    return (tp + smooth) / (len(y_true_f) - tn + smooth)


def accuracy(y_true, y_pred, smooth=smooth_default):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f == y_pred_f)

    return (intersection + smooth) / (len(y_true_f) + smooth)


def jacc_loss(y_true, y_pred):
    return -jacc_idx(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_jacc_single(mask_true, mask_pred, smooth=smooth_default):
    bool_true = mask_true.reshape(-1).astype(np.bool)
    bool_pred = mask_pred.reshape(-1).astype(np.bool)
    if bool_true.shape != bool_pred.shape:
        raise ValueError("Masks of different sizes.")

    bool_sum = bool_true.sum() + bool_pred.sum()
    if bool_sum == 0:
        print
        "Empty mask"
        return 0, 0
    intersec = np.logical_and(bool_true, bool_pred).sum()
    dice = 2. * intersec / bool_sum
    jacc = jaccard_similarity_score(bool_true.reshape((1, -1)), bool_pred.reshape((1, -1)), normalize=True, sample_weight=None)
    return dice, jacc


def dice_jacc_mean(mask_true, mask_pred, smooth=smooth_default):
    dice = 0
    jacc = 0
    for i in range(mask_true.shape[0]):
        current_dice, current_jacc = dice_jacc_single(mask_true=mask_true[i], mask_pred=mask_pred[i], smooth=smooth)
        dice = dice + current_dice
        jacc = jacc + current_jacc
    return dice / mask_true.shape[0], jacc / mask_true.shape[0]
