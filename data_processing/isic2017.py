# ==================================================
# Copyright (C) 2017-2018
# author: Claire Tang
# email: Claire Tang@gmail.com
# Date: 9/8/19
#
# This file is part of mri project.
# 
# This can not be copied and/or distributed 
# without the express permission of Claire Tang
# ==================================================


import os
import shutil

classes = ['melanoma', 'seborrheic_keratosis']


def prepare_part3():
    if not os.path.exists('../data/isic2017/part3_training'):
        os.makedirs('../data/isic2017/part3_training')
    if not os.path.exists('../data/isic2017/part3_testing'):
        os.makedirs('../data/isic2017/part3_testing')

    # with open('../data/isic2017/ISIC-2017_Training_Part3_GroundTruth.csv', 'r') as gt_in:
    #     # skip first line
    #     next(gt_in)
    #
    #     for line in gt_in:
    #         tokens = line.rstrip().split(',')
    #
    #         # parse the file and label
    #         img_file = tokens[0]
    #         label_list = [float(x.strip()) for x in tokens[1:]]
    #
    #         try:
    #             class_label = label_list.index(1.0)
    #
    #             # write into new folder
    #             if not os.path.exists('../data/isic2017/part3_training/%s' % classes[class_label]):
    #                 os.makedirs('../data/isic2017/part3_training/%s' % classes[class_label])
    #
    #             shutil.move('../data/isic2017/ISIC-2017_Training_Data/%s.jpg' % img_file, '../data/isic2017/part3_training/%s' % classes[class_label])
    #         except:
    #             continue

    with open('../data/isic2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv', 'r') as gt_in:
        # skip first line
        next(gt_in)

        for line in gt_in:
            tokens = line.rstrip().split(',')

            # parse the file and label
            img_file = tokens[0]
            label_list = [float(x.strip()) for x in tokens[1:]]

            try:
                class_label = label_list.index(1.0)

                # write into new folder
                if not os.path.exists('../data/isic2017/part3_testing/%s' % classes[class_label]):
                    os.makedirs('../data/isic2017/part3_testing/%s' % classes[class_label])

                shutil.move('../data/isic2017/ISIC-2017_Test_v2_Data/%s.jpg' % img_file, '../data/isic2017/part3_testing/%s' % classes[class_label])
            except:
                continue


# prepare_part3()
