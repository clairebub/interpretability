# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-07
#
# This file is part of skin project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import os
import shutil
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']


def five_fold_split_isic2019():
    if not os.path.exists('./data/isic2019/isic2019_training'):
        os.makedirs('./data/isic2019/isic2019_training')

    with open('./data/isic2019/ISIC_2019_Training_GroundTruth.csv', 'r') as gt_in:
        # skip first line
        next(gt_in)

        for line in gt_in:
            tokens = line.rstrip().split(',')

            # parse the file and label
            img_file = tokens[0]
            label_list = [float(x.strip()) for x in tokens[1:]]
            class_label = label_list.index(1.0)

            # write into new folder
            p = random.uniform(0, 1)

            if p >= 0.2:
                if not os.path.exists('./data/isic2019/isic2019_training/%s' % classes[class_label]):
                    os.makedirs('./data/isic2019/isic2019_training/%s' % classes[class_label])

                shutil.move('./data/isic2019/ISIC_2019_Training_Input/%s.jpg' % img_file, './data/isic2019/isic2019_training/%s' % classes[class_label])

            else:
                if not os.path.exists('./data/isic2019/isic2019_testing/%s' % classes[class_label]):
                    os.makedirs('./data/isic2019/isic2019_testing/%s' % classes[class_label])

                shutil.move('./data/isic2019/ISIC_2019_Training_Input/%s.jpg' % img_file, './data/isic2019/isic2019_testing/%s' % classes[class_label])


def load_full_data_isic2019():
    if not os.path.exists('./data/isic2019/isic2019_training_full'):
        os.makedirs('./data/isic2019/isic2019_training_full')

    with open('./data/isic2019/ISIC_2019_Training_GroundTruth.csv', 'r') as gt_in:
        # skip first line
        next(gt_in)

        for line in gt_in:
            tokens = line.rstrip().split(',')

            # parse the file and label
            img_file = tokens[0]
            label_list = [float(x.strip()) for x in tokens[1:]]
            class_label = label_list.index(1.0)

            # write into new folder
            if not os.path.exists('./data/isic2019/isic2019_training_full/%s' % classes[class_label]):
                os.makedirs('./data/isic2019/isic2019_training_full/%s' % classes[class_label])

            shutil.move('./data/isic2019/ISIC_2019_Training_Input/%s.jpg' % img_file, './data/isic2019/isic2019_training_full/%s' % classes[class_label])


def data_visualization():
    # normalization for isic2019
    normalize = transforms.Normalize(mean=[0.6803108, 0.5250009, 0.5146185], std=[0.18023035, 0.18443975, 0.19847354])

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomRotation(180),
                                         transforms.ToTensor(),
                                         normalize])

    isic2019_train_data = datasets.ImageFolder(root='data/isic2019/isic2019_visualization', transform=data_transform)

    def show_dataset(dataset, n=20):
        imgs = torch.stack([dataset[i][0] for _ in range(n)
                           for i in range(len(dataset))])
        grid = make_grid(imgs).numpy()
        plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
        plt.axis('off')

    show_dataset(isic2019_train_data)


def generate_test_gt_csv(data_path):
    classes_order = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

    # create new result file
    with open('%s_GroundTruth.csv' % data_path, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['image'] + classes_order + ['UNK'])

        for class_label in classes:
            for filename in os.listdir('%s/%s' % (data_path, class_label)):
                img_file = filename.replace('.jpg', '')

                class_binary = np.zeros(len(classes_order) + 1)

                class_idx = classes.index(class_label)
                class_binary[class_idx] = 1.0

                filewriter.writerow([img_file] + class_binary.tolist())


def get_data_statistics(data_path):
    """compute image data statistics (mean, std)"""

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor()])

    isic2019_train_data = datasets.ImageFolder(root=data_path, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(
        isic2019_train_data,
        batch_size=4096,
        num_workers=4,
        shuffle=False
    )

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in enumerate(train_loader, 0):
        # shape (batch_size, 3, height, width)
        images, labels = data
        numpy_image = np.asarray([item.numpy() for item in images])

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    print(pop_mean)
    print(pop_std0)

    return pop_mean, pop_std0, pop_std1


get_data_statistics('./data/isic2019/isic2019_training_full')

# generate_test_gt_csv('./data/isic2019/isic2019_testing')
