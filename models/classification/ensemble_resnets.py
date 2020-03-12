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


class ResnetEnsemble2(nn.Module):
    def __init__(self, resnets, num_classes):
        super(ResnetEnsemble2, self).__init__()

        resnet1, resnet2 = resnets

        self.num_features = 0

        self.num_features += resnet1.module.fc.in_features
        self.resnet_en1 = nn.Sequential(*list(resnet1.module.children())[:-1])

        self.num_features += resnet2.module.fc.in_features
        self.resnet_en2 = nn.Sequential(*list(resnet2.module.children())[:-1])

        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x, *args):

        x1 = self.resnet_en1(x)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.resnet_en2(x)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        return x


class ResnetEnsemble3(nn.Module):
    def __init__(self, resnets, num_classes):
        super(ResnetEnsemble3, self).__init__()

        resnet1, resnet2, resnet3 = resnets

        self.num_features = 0

        self.num_features += resnet1.module.fc.in_features
        self.resnet_en1 = nn.Sequential(*list(resnet1.module.children())[:-1])

        self.num_features += resnet2.module.fc.in_features
        self.resnet_en2 = nn.Sequential(*list(resnet2.module.children())[:-1])

        self.num_features += resnet3.module.fc.in_features
        self.resnet_en3 = nn.Sequential(*list(resnet3.module.children())[:-1])

        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x, *args):
        x1 = self.resnet_en1(x)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.resnet_en2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.resnet_en3(x)
        x3 = x3.view(x3.size(0), -1)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)

        return x
