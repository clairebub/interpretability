# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-18
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import os
import numpy as np
import csv
import datetime
import pandas as pd
import ntpath
import pickle

import torch

from torchvision.utils import save_image

from evaluation import ind_eval
from evaluation import ood_detection

from utils.utils_dataloader import Denormalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

denorm = Denormalize(mean=[0.6796547, 0.5259538, 0.51874095], std=[0.18123391, 0.18504128, 0.19822954])


def save_images_batch(_images, _paths, meta_task):

    for image, path in zip(_images, _paths):
        new_file = path.replace(meta_task, meta_task + '_enhanced')

        # create dir if not existing
        if not os.path.exists(os.path.dirname(new_file)):
            os.makedirs(os.path.dirname(new_file))

        save_image(image, new_file)


def test_ind(args, cnn, test_loader):
    """IND test function"""

    # # create file and input first row
    # current_time = datetime.datetime.now()
    # result_csv = '%s.csv' % current_time.strftime("%Y_%m_%d_%H_%M")
    #
    # if args.generate_result:
    #     # create new result file
    #     with open('results/%s' % result_csv, 'w') as csvfile:
    #         filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #         filewriter.writerow(['image'] + classes + ['UNK'])

    # change model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()

    # test IND performance
    print("IND performance:")
    ind_performance = ind_eval.ind_eval(args, cnn, test_loader)
    df = pd.DataFrame(ind_performance.items(), columns=["metric", "result"])
    print(df)


def test_ood(args, cnn, train_loader, test_loader, ood_loader, classes, ood_method='ODIN'):
    """OOD test function"""

    # change model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()

    # test OOD performance
    print("OOD performance:")
    ood_method = ood_detection.__dict__[ood_method](args, cnn, train_loader, test_loader, len(classes))
    ood_performance = ood_method.ood_eval(ood_loader)
    df = pd.DataFrame(ood_performance.items(), columns=["metric", "result"])
    print(df)
