# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 9/4/19
#
# This file is part of mri project.
#
# This can not be copied and/or distributed
# without the express permission of yilin.shen
# ==================================================

import torch
import torch.nn.functional as F

from metrics.ind import cal_ind_metrics


def ind_eval(args, cnn, data_loader):
    cnn.eval()

    predictions = []
    labels = []

    for input, target in data_loader:

        input = input.cuda()
        target = target.cuda()

        cnn.cuda()
        cnn.zero_grad()

        if 'cosine' in args.model_type:
            pred, _, _ = cnn(input)
        else:
            pred = cnn(input)

        pred = F.softmax(pred, dim=-1)
        pred_value, pred = torch.max(pred.data, 1)

        # append into predictions and labels
        predictions.extend(pred.cpu().numpy())
        labels.extend(target.cpu().numpy())

    test_acc, test_auc = cal_ind_metrics(predictions, labels)

    return {'accuracy(\u2191)': test_acc,
            'auc(\u2191)': test_auc
            }
