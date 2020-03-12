# ==================================================
# Copyright (C) 2017-2018
# author: Claire Tang
# email: Claire Tang@gmail.com
# Date: 2019-08-22
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of Claire Tang
# ==================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineResnetEnsemble(nn.Module):
    def __init__(self, resnets, num_classes):
        super(CosineResnetEnsemble, self).__init__()

        self.num_classes = num_classes
        self.num_features = 0

        resnet1, resnet2 = resnets

        self.num_features += resnet1.module.fc.in_features
        self.resnet_en1 = nn.Sequential(*list(resnet1.module.children())[:-3])

        self.num_features += resnet2.module.fc.in_features
        self.resnet_en2 = nn.Sequential(*list(resnet2.module.children())[:-3])

        self.fc = nn.Linear(self.num_features, num_classes, bias=False)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(self.num_features, 1)

    def forward(self, x, *args):

        x1 = self.resnet_en1(x)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.resnet_en2(x)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)

        # temperature scale
        scale = torch.exp(self.bn_scale(self.fc_scale(x)))

        # cosine sim
        x_norm = F.normalize(x)
        w_norm = F.normalize(nn.Parameter(self.fc.weight))
        w_norm_transposed = torch.transpose(w_norm, 0, 1)
        cos_sim = torch.mm(x_norm, w_norm_transposed)

        # scaled cosine sim
        scaled_cosine = cos_sim * scale

        return scaled_cosine, cos_sim, x
