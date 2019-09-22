# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 9/1/19
#
# This file is part of mri project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import sklearn as sk
import numpy as np


def accuracy(predictions, labels):
    correct = (predictions == labels)
    correct = np.array(correct).astype(bool)

    return np.mean(correct)


def multiclass_roc_auc_score(predictions, labels, average="macro"):
    lb = sk.preprocessing.LabelBinarizer()
    lb.fit(predictions)

    predictions = lb.transform(predictions)
    labels = lb.transform(labels)

    return sk.metrics.roc_auc_score(labels, predictions, average=average)


def cal_ind_metrics(predictions, labels):
    """calculate IND evaluation metrics"""

    predictions = np.array(predictions)
    labels = np.array(labels)

    acc = accuracy(predictions, labels)
    try:
        auc = multiclass_roc_auc_score(predictions, labels)
    except:
        auc = 0

    return acc, auc


def cal_ind_metrics_with_print(predictions, labels):
    """calculate IND evaluation metrics"""

    predictions = np.array(predictions)
    labels = np.array(labels)

    acc = accuracy(predictions, labels)
    print("IND Accuracy (higher is better): ", acc)

    auc = multiclass_roc_auc_score(predictions, labels)
    print("IND AUC (higher is better): ", auc)

    return acc, auc
