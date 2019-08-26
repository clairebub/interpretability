# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-15
#
# This file is part of MRI project.
# Cosine network model for OOD detection along with classification
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineNet(nn.Module):
    def __init__(self, cnn, num_classes):
        super(CosineNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = cnn.fc.in_features  # for resnet
        # self.num_features = self.cnn.classifier.in_features # for densenet

        # features after removing last layer
        self.cosine_cnn = nn.Sequential(*list(cnn.children())[:-1])
        # for child in self.cosine_cnn.children():
        #     for param in child.parameters():
        #         param.requires_grad = False

        self.fc = nn.Linear(self.num_features, num_classes, bias=False)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(self.num_features, 1)

    def forward(self, x):

        x = self.cosine_cnn(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1)) # for densenet
        # x = torch.flatten(x, 1)

        # match the size
        x = x.view(x.size(0), -1)

        # temperature scale
        scale = torch.exp(self.bn_scale(self.fc_scale(x)))

        # cosine sim
        x_norm = F.normalize(x)
        w_norm = F.normalize(nn.Parameter(self.fc.weight))
        w_norm_transposed = torch.transpose(w_norm, 0, 1)
        cos_sim = torch.mm(x_norm, w_norm_transposed)

        # scaled cosine sim
        scaled_cosine = cos_sim * scale

        return scaled_cosine, cos_sim
