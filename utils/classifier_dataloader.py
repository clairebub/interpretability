# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-10
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from PIL import Image


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        # tuple_with_path = original_tuple

        return tuple_with_path


class Denormalize(object):
    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        tmp_df = pd.read_csv(csv_path)

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.X_train[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, transform=None, target_transform=None):
        self.imgs = []
        self.targets = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        # root / <label> / <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label):
                views = []
                for view in os.listdir(root + '/' + label + '/' + item):
                    views.append(root + '/' + label + '/' + item + '/' + view)

                self.imgs.append(views)
                self.targets.append(self.class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        views = []

        orginal_views = self.imgs[index]
        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)

            views.append(im)

        return views, self.targets[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.imgs)


class MultiViewDataSetWithPaths(MultiViewDataSet):
    """Custom dataset that includes image file paths. Extends MultiViewDataSet"""

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(MultiViewDataSetWithPaths, self).__getitem__(index)

        # the image file path
        view_paths = self.imgs[index]

        # # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (view_paths,))

        return tuple_with_path
