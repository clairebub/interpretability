# ==================================================
# Copyright (C) 2017-2018
# author: Claire Tang
# email: Claire Tang@gmail.com
# Date: 2019-08-07
#
# This file is part of skin project.
# 
# This can not be copied and/or distributed 
# without the express permission of Claire Tang
# ==================================================

import os
import shutil
import random
import csv
import cv2
import ntpath
import numpy as np
import argparse

from PIL import Image

import torch
from torchvision import datasets, transforms

classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']


def five_fold_split_isic2019():
    if not os.path.exists('../data/isic2019/skin_training'):
        os.makedirs('../data/isic2019/skin_training')
    if not os.path.exists('../data/isic2019/skin_validation'):
        os.makedirs('../data/isic2019/skin_validation')

    with open('../data/isic2019/ISIC_2019_Training_GroundTruth.csv', 'r') as gt_in:
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
                if not os.path.exists('../data/isic2019/skin_training/%s' % classes[class_label]):
                    os.makedirs('../data/isic2019/skin_training/%s' % classes[class_label])

                shutil.copy('../data/isic2019/ISIC_2019_Training_Input/%s.jpg' % img_file, '../data/isic2019/skin_training/%s' % classes[class_label])

            else:
                if not os.path.exists('../data/isic2019/skin_validation/%s' % classes[class_label]):
                    os.makedirs('../data/isic2019/skin_validation/%s' % classes[class_label])

                shutil.copy('../data/isic2019/ISIC_2019_Training_Input/%s.jpg' % img_file, '../data/isic2019/skin_validation/%s' % classes[class_label])


def generate_balanced_test(dataset, task):
    # metadata_types = ['image', 'age_approx', 'anatom_site_general', 'lesion_id', 'sex']

    org_test = '../data/%s/%s_validation' % (dataset, task)
    balanced_test = '../data/%s/%s_validation_balanced' % (dataset, task)

    counting = [len(files) for r, d, files in os.walk(org_test)]
    min_class = min(counting[1:])

    files = [files for r, d, files in os.walk(org_test)]
    classes = [d for r, d, files in os.walk(org_test)][0]

    print(classes)
    print(counting)

    if not os.path.exists(balanced_test):
        os.makedirs(balanced_test)

    for images, class_name in zip(files[1:], classes):
        print(class_name, images)

        if not os.path.exists('%s/%s' % (balanced_test, class_name)):
            os.makedirs('%s/%s' % (balanced_test, class_name))

        random.shuffle(images)

        for i in range(min_class):
            shutil.copyfile('%s/%s/%s' % (org_test, class_name, images[i]), '%s/%s/%s' % (balanced_test, class_name, images[i]))


def load_full_data_isic2019():
    if not os.path.exists('../data/isic2019/skin_training_full'):
        os.makedirs('../data/isic2019/skin_training_full')

    with open('../data/isic2019/ISIC_2019_Training_GroundTruth.csv', 'r') as gt_in:
        # skip first line
        next(gt_in)

        for line in gt_in:
            tokens = line.rstrip().split(',')

            # parse the file and label
            img_file = tokens[0]
            label_list = [float(x.strip()) for x in tokens[1:]]
            class_label = label_list.index(1.0)

            # write into new folder
            if not os.path.exists('../data/isic2019/skin_training_full/%s' % classes[class_label]):
                os.makedirs('../data/isic2019/skin_training_full/%s' % classes[class_label])

            shutil.copy('../data/isic2019/ISIC_2019_Training_Input/%s.jpg' % img_file, '../data/isic2019/skin_training_full/%s' % classes[class_label])


def generate_test_gt_csv(data_path):
    """for validation"""

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


def get_data_statistics(dataset, task):
    data_path = '../data/%s/%s_training' % (dataset, task)

    """compute image data statistics (mean, std)"""
    data_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor()])

    # isic2019_train_data = datasets.CIFAR10(root='data/ood/', train=True, transform=data_transform, download=True)

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


def five_fold_split_isic2019_meta(metadata_type):
    metadata_types = ['image', 'age_approx', 'anatom_site_general', 'lesion_id', 'sex']
    metadata_type_idx = metadata_types.index(metadata_type)

    if not os.path.exists('../data/isic2019/%s_training' % metadata_type):
        os.makedirs('../data/isic2019/%s_training' % metadata_type)
    if not os.path.exists('../data/isic2019/%s_validation' % metadata_type):
        os.makedirs('../data/isic2019/%s_validation' % metadata_type)

    with open('../data/isic2019/ISIC_2019_Training_Metadata.csv', 'r') as gt_in:
        # skip first line
        next(gt_in)

        for line in gt_in:
            info = line.rstrip().split(',')

            # write into new folder
            if info[metadata_type_idx] != '':
                p = random.uniform(0, 1)

                class_name = info[metadata_type_idx].replace(' ', '_')

                if p >= 0.2:
                    if not os.path.exists('../data/isic2019/%s_training/%s' % (metadata_type, class_name)):
                        os.makedirs('../data/isic2019/%s_training/%s' % (metadata_type, class_name))

                    shutil.copy('../data/isic2019/ISIC_2019_Training_Input/%s.jpg' % info[0], '../data/isic2019/%s_training/%s' % (metadata_type, class_name))

                else:
                    if not os.path.exists('../data/isic2019/%s_validation/%s' % (metadata_type, class_name)):
                        os.makedirs('../data/isic2019/%s_validation/%s' % (metadata_type, class_name))

                    shutil.copy('../data/isic2019/ISIC_2019_Training_Input/%s.jpg' % info[0], '../data/isic2019/%s_validation/%s' % (metadata_type, class_name))


def load_full_data_isic2019_meta(metadata_type):
    metadata_types = ['image', 'age_approx', 'anatom_site_general', 'lesion_id', 'sex']
    metadata_type_idx = metadata_types.index(metadata_type)

    if not os.path.exists('../data/isic2019/%s_training_full' % metadata_type):
        os.makedirs('../data/isic2019/%s_training_full' % metadata_type)

    with open('../data/isic2019/ISIC_2019_Training_Metadata.csv', 'r') as gt_in:
        # skip first line
        next(gt_in)

        for line in gt_in:
            info = line.rstrip().split(',')

            # write into new folder
            if info[metadata_type_idx] != '':
                p = random.uniform(0, 1)

                class_name = info[metadata_type_idx].replace(' ', '_')

                if not os.path.exists('../data/isic2019/%s_training_full/%s' % (metadata_type, class_name)):
                    os.makedirs('../data/isic2019/%s_training_full/%s' % (metadata_type, class_name))

                shutil.copy('../data/isic2019/ISIC_2019_Training_Input/%s.jpg' % info[0], '../data/isic2019/%s_training_full/%s' % (metadata_type, class_name))


def data_cropping():
    test_dir = 'data/isic2019/skin_testing_full/UNKNOWN'

    for image_file in os.listdir(test_dir):
        print('Processing image %s...' % image_file)

        image_name = ntpath.basename(image_file)

        # crop on original image
        image = cv2.imread('%s/%s' % (test_dir, image_file))
        flip_image = cv2.flip(image, 1)

        roi = {'rb': image[0:896, 0:896], 'lb': image[0:896, 128:1024], 'rt': image[128:1024, 0:896], 'lt': image[128:1024, 128:1024], 'c': image[64:960, 64:960],
               'frb': flip_image[0:896, 0:896], 'flb': flip_image[0:896, 128:1024], 'frt': flip_image[128:1024, 0:896], 'flt': flip_image[128:1024, 128:1024], 'fc': flip_image[64:960, 64:960]}

        for crop_type, croppted_image in roi.items():
            # create directory if not existing
            save_dir = test_dir.replace('testing', 'testing_%s' % crop_type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            cv2.imwrite('%s/%s' % (save_dir, image_name), croppted_image)


def find_identical_images(path2019, img_path, seg_path, out_img_path, out_seg_path):

    for subdir2019, dirs2019, files2019 in os.walk(path2019):
        for file2019 in files2019:
            img2019 = os.path.join(subdir2019, file2019)

            for subdir, dirs, files in os.walk(img_path):
                for file in files:
                    img = os.path.join(subdir, file)

                    identical = open(img2019, "rb").read() == open(img, "rb").read()

                    if identical is True:
                        print('same %s and %s' % (img2019, img))

                        # create data directory if not existing
                        data_dir = img2019.replace(path2019, out_img_path)
                        data_dir = data_dir.replace(ntpath.basename(data_dir), '')
                        if not os.path.exists(data_dir):
                            os.makedirs(data_dir)

                        shutil.copyfile(img2019, img2019.replace(path2019, out_img_path))
                        shutil.copyfile('%s%s' % (seg_path, file.replace('.jpg', '_segmentation.png')), '%s%s' % (out_seg_path, file2019.replace('.jpg', '_segmentation.png')))


def resize_img(data_path):
    width, height = 224, 224

    save_path = data_path.replace('segmentation', 'segmentation_resized')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in os.listdir(data_path):

        # read image
        img = Image.open('%s%s' % (data_path, filename))

        # resize img using blinear
        img_resized = img.resize((width, height), Image.BILINEAR)

        # save resized image
        img_resized.save('%s%s' % (save_path, filename))


if __name__ == "__main__":
    task_options = ['skin', 'meta', 'segmentation', 'data_statistics']

    parser = argparse.ArgumentParser(description='data processing')
    parser.add_argument('--task', default='segmentation', choices=task_options)
    args = parser.parse_args()

    if args.task == 'skin':
        if not os.path.exists('../data/isic2019'):
            os.makedirs('../data/isic2019')

        five_fold_split_isic2019()
        generate_test_gt_csv('../data/isic2019/skin_validation')

        generate_balanced_test('isic2019', 'skin')
        generate_test_gt_csv('../data/isic2019/skin_validation_balanced')

        load_full_data_isic2019()

    elif args.task == 'meta':
        five_fold_split_isic2019_meta('age_approx')
        five_fold_split_isic2019_meta('anatom_site_general')
        five_fold_split_isic2019_meta('sex')

        generate_balanced_test('isic2019', 'age_approx')
        generate_balanced_test('isic2019', 'anatom_site_general')
        generate_balanced_test('isic2019', 'sex')

        load_full_data_isic2019_meta('age_approx')
        load_full_data_isic2019_meta('anatom_site_general')
        load_full_data_isic2019_meta('sex')

    elif args.task == 'segmentation':
        # find segmentation ground truth from isic2018
        find_identical_images('../data/isic2019/skin_validation/', '../data/isic2018/ISIC2018_Task1-2_Training_Input/',
                              '../data/isic2018/ISIC2018_Task1_Training_GroundTruth/', '../data/isic2019/skin_segmentation_validation/',
                              '../data/isic2019/skin_segmentation_validation_groundtruth/')

        resize_img('../data/isic2019/skin_segmentation_validation_groundtruth/')

    elif args.task == 'data_statistics':
        if not os.path.exists('../data/isic2019/skin_training_full'):
            raise RuntimeError('Please process data first by running --task=skin')

        get_data_statistics('isic2019', '../data/isic2019/skin_training_full')

    else:
        RuntimeError('Wrong input')
