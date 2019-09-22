# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 9/9/19
#
# This file is part of mri project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import os
import numpy as np
import sklearn
import pandas as pd

from PIL import Image

from metrics import segmentation


def binary_img_to_array(img_name):
    im = Image.open(img_name)
    pix = im.load()
    img_size = im.size

    binary_matrix = np.zeros(img_size)

    for x in range(img_size[0]):
        for y in range(img_size[1]):

            if isinstance(pix[x, y], tuple):
                value = pix[x, y][0]
            else:
                value = pix[x, y]

            if value == 0:
                binary_matrix[x, y] = 0
            else:
                binary_matrix[x, y] = 1

    binary_matrix = binary_matrix.flatten()

    return binary_matrix


def segmentation_eval(gt_path, result_path):

    gt_matrices = []
    result_metrices = []

    for filename in os.listdir(result_path):
        if filename.endswith('_segmentation.png'):
            gt_img_name = '%s/%s' % (gt_path, filename)
            result_img_name = '%s/%s' % (result_path, filename)

            gt_matrices.append(binary_img_to_array(gt_img_name))
            result_metrices.append(binary_img_to_array(result_img_name))

    jacc_idx = sklearn.metrics.jaccard_similarity_score(np.array(gt_matrices), np.array(result_metrices))
    dice_coef = segmentation.dice_coef(np.array(gt_matrices), np.array(result_metrices), smooth=0)
    accuracy = segmentation.accuracy(np.array(gt_matrices), np.array(result_metrices), smooth=0)

    segmentation_performance = {'jacc_idx(\u2191)': jacc_idx,
                                'dice_coef(\u2191)': dice_coef,
                                'accuracy(\u2191)': accuracy,
                                }

    # print out results
    df = pd.DataFrame(segmentation_performance.items(), columns=["metric", "result"])
    print(df)

    return segmentation_performance


def segmentation_eval_each(gt_path, result_path):

    with open('%s.txt' % result_path, 'w') as out_file:
        for filename in os.listdir(result_path):
            if filename.endswith('_segmentation.png'):
                gt_img_name = '%s/%s' % (gt_path, filename)
                result_img_name = '%s/%s' % (result_path, filename)

                gt_matrices = [binary_img_to_array(gt_img_name)]
                result_metrices = [binary_img_to_array(result_img_name)]

                jacc_idx = sklearn.metrics.jaccard_similarity_score(np.array(gt_matrices), np.array(result_metrices))
                dice_coef = segmentation.dice_coef(np.array(gt_matrices), np.array(result_metrices), smooth=0)
                accuracy = segmentation.accuracy(np.array(gt_matrices), np.array(result_metrices), smooth=0)

                out_file.write('{}: {:.5f} {:.5f} {:.5f}\n'.format(filename, jacc_idx, dice_coef, accuracy))

