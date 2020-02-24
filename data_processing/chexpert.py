# ==================================================
# Copyright (C) 2017-2020
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2020-01-20
#
# This file is part of ImageClassification project.
# Base model train/valid/test
#
# This can not be copied and/or distributed
# without the express permission of yilin.shen
# ==================================================


import os
import numpy as np
from shutil import copy

import torch
from torchvision import datasets, transforms

# from utils import classifier_dataloader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']
classes = ['No Finding', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']

def prepare_multiview_data(in_csv, out_dir, class_label, mode):
    with open('%s/%s.csv' % (in_csv, mode), 'r') as csv_file:

        # get column name
        column_name = csv_file.readline().split(',')
        print(column_name)

        # get column index of class label
        class_idx = column_name.index(class_label)

        # parse each sample
        pair_stack = []
        prev_path_list = []
        prev_label = -10
        samples = 0
        for line in csv_file:
            sample = line.split(',')
            # print(sample)

            # parse path
            path = '../data/' + sample[column_name.index('Path')]
            path_list = path.split('/')

            # get view
            view = path_list[-1]

            if (sample[class_idx] == '') or (sample[class_idx] == '-1.0'):
                # reset
                pair_stack = []
                prev_path_list = []
                prev_label = -10
            else:
                label = int(float(sample[class_idx]))

                if 'frontal' in view:
                    # initialize for a potential new instance pair
                    pair_stack = [view]
                    prev_path_list = path_list.copy()
                    prev_label = label
                elif 'lateral' in view:

                    mv_path = '/'.join(path_list[:-1])
                    prev_mv_path = '/'. join(prev_path_list[:-1])

                    # check if it is a pair
                    if ('view2' in view) and pair_stack and (pair_stack.pop() == 'view1_frontal.jpg') and (prev_mv_path == mv_path) and (prev_label == label):

                        samples = samples + 1

                        save_path = '%s/%s_%s/%d/%s_%s/' % (out_dir, class_label.lower().replace(' ', '_'), mode, label, path_list[-3], path_list[-2])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        # print('Found one {} sample {}'.format(label, save_path))
                        if samples % 100 == 0:
                            print('Found {} samples'.format(samples))

                        # copy images of both views
                        copy('/'.join(prev_path_list), save_path)
                        copy('/'.join(path_list), save_path)
                else:
                    raise RuntimeError('invalid view')


def prepare_singleview_data(in_csv, out_dir, class_label, mode):
    with open('%s/%s.csv' % (in_csv, mode), 'r') as csv_file:

        # get column name
        column_name = csv_file.readline().split(',')
        # print(column_name)

        # get column index of class label
        class_idx = column_name.index(class_label)

        samples = 0
        for line in csv_file:
            sample = line.split(',')

            if (sample[class_idx] == '1.0') or (sample[class_idx] == '0.0'):

                samples = samples + 1

                # print('Found one {} sample {}'.format(label, save_path))
                if samples % 1000 == 0:
                    print('Found {} samples'.format(samples))

                label = int(float(sample[class_idx]))

                save_path = '%s/%s_%s/%d' % (out_dir, class_label.lower().replace(' ', '_'), mode, label)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # path = '../data/' + sample[column_name.index('Path')]
                # path_list = path.split('/')
                # path_list[-1] = '%d.jpg' % samples

                try:
                    copy('../data/' + sample[column_name.index('Path')], '%s/%s.jpg' % (save_path, samples))
                except:
                    continue


def get_multiview_data_statistics(dataset, task):
    data_path = '../data/%s/%s_training' % (dataset, task)

    """compute image data statistics (mean, std)"""
    data_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor()])
    train_data = classifier_dataloader.MultiViewDataSet(root=data_path, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=4096,
        num_workers=16,
        shuffle=False
    )

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in enumerate(train_loader, 0):
        # shape (batch_size, 3, height, width)
        images, labels = data
        # concatenate images from all views
        images = torch.cat(images)
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


def get_singleview_data_statistics(dataset, task):
    data_path = '../data/%s/%s_training' % (dataset, task)

    """compute image data statistics (mean, std)"""
    data_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor()])

    train_data = datasets.ImageFolder(root=data_path, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data,
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


# prepare_multiview_data(in_csv='../data/CheXpert-v1.0', out_dir='../data/chexpert_multiview', class_label='Consolidation', mode='validation')
# get_multiview_data_statistics('chexpert_multiview', 'consolidation')

# prepare_singleview_data(in_csv='../data/CheXpert-v1.0', out_dir='../data/chexpert', class_label='Enlarged Cardiomediastinum', mode='training')
# get_singleview_data_statistics('chexpert', 'enlarged_cardiomediastinum')

for c in classes:
    print('processing %s...' % c)
    prepare_singleview_data(in_csv='../data/CheXpert-v1.0', out_dir='../data/chexpert', class_label=c, mode='training')
