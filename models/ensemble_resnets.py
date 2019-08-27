# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-22
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import torch
import torch.nn as nn


class ResnetEnsemble(nn.Module):
    def __init__(self, resnets, num_classes):
        super(ResnetEnsemble, self).__init__()

        self.num_classes = num_classes

        self.num_features = 0

        # resnet1, resnet2, resnet3, resnet4, resnet5 = resnets
        # resnet1, resnet2, resnet3 = resnets
        resnet1, resnet2 = resnets

        self.num_features += resnet1.module.fc.in_features
        self.resnet_en1 = nn.Sequential(*list(resnet1.module.children())[:-1])

        self.num_features += resnet2.module.fc.in_features
        self.resnet_en2 = nn.Sequential(*list(resnet2.module.children())[:-1])

        # self.num_features += resnet3.module.fc.in_features
        # self.resnet_en3 = nn.Sequential(*list(resnet3.module.children())[:-1])
        # for param in self.resnet_en3.parameters():
        #     param.requires_grad = False
        #
        # self.num_features += resnet4.module.fc.in_features
        # self.resnet_en4 = nn.Sequential(*list(resnet4.module.children())[:-1])
        # for param in self.resnet_en4.parameters():
        #     param.requires_grad = False
        #
        # self.num_features += resnet5.module.fc.in_features
        # self.resnet_en5 = nn.Sequential(*list(resnet5.module.children())[:-1])
        # for param in self.resnet_en5.parameters():
        #     param.requires_grad = False

        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x, *args):

        x1 = self.resnet_en1(x)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.resnet_en2(x)
        x2 = x2.view(x2.size(0), -1)

        # x3 = self.resnet_en3(x)
        # x3 = x3.view(x3.size(0), -1)
        #
        # x4 = self.resnet_en4(x)
        # x4 = x4.view(x4.size(0), -1)
        #
        # x5 = self.resnet_en5(x)
        # x5 = x5.view(x5.size(0), -1)

        # x = torch.cat(x, dim=1)
        x = torch.cat((x1, x2), dim=1)
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.classifier(x)

        return x

