# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-18
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.ood_metrics import cal_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_ood(cnn, test_loader, ood_loader, method='odin'):
    """test function for OOD detection"""

    # change model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()

    def get_pred_conf(data_loader):
        all_pred_conf, all_cosine_conf, all_labels = [], [], []

        for data in data_loader:
            if len(data) == 3:
                images, labels, paths = data
            else:
                images, labels = data

            images = Variable(images.to(device), requires_grad=True)
            labels = Variable(labels).to(device)

            cnn.to(device)
            cnn.zero_grad()

            pred = cnn(images)

            if method == 'odin':
                T = 1000
                pred = pred / T

                # input preprocessing
                xent = nn.CrossEntropyLoss()
                loss = xent(pred, labels)
                loss.backward()

                images = images - 0.005 * torch.sign(images.grad)

                pred = cnn(images)
                pred = pred / T
                pred = F.softmax(pred, dim=-1)
                pred_conf, _ = torch.max(pred.data, 1)
                all_pred_conf.extend(pred_conf.cpu().detach().numpy())

                all_labels.extend(labels.cpu().detach().numpy())
            else:
                raise RuntimeError('OOD mode not supported')

        return all_pred_conf, all_labels

    ind_pred_conf, ind_labels = get_pred_conf(test_loader)
    ood_pred_conf, ood_labels = get_pred_conf(ood_loader)

    cal_metrics(ind_pred_conf, ood_pred_conf)
