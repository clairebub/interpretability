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
import ntpath
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image

from utils.ood_metrics import cal_ood_metrics
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


def test_with_ood(cnn, task, test_loader, ood_loader, classes, ood_method='odin', generate_result=False, validation=True):
    """test function for IND and base OOD (OOD without base model retraining)"""

    # create file and input first row
    current_time = datetime.datetime.now()
    result_csv = '%s.csv' % current_time.strftime("%Y_%m_%d_%H_%M")

    if generate_result:
        # create new result file
        with open('results/%s' % result_csv, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['image'] + classes + ['UNK'])

    # change model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()

    def get_pred_conf(data_loader):
        all_pred_conf, all_pred, all_labels, correct, all_full_prob, all_paths, all_cosine_conf = [], [], [], [], [], [], []

        for data in data_loader:
            if len(data) == 3:
                images, labels, paths = data
            else:
                images, labels = data

            images = Variable(images.to(device), requires_grad=True)
            labels = Variable(labels).to(device)

            cnn.to(device)
            cnn.zero_grad()

            if ood_method == 'base':
                # model inference
                pred = cnn(images)
                pred = F.softmax(pred, dim=-1)

                # get full probability distribution
                full_prob_batch = pred.cpu().detach().numpy()
                all_full_prob.extend(full_prob_batch)

                # get conf score
                pred_conf, pred = torch.max(pred.data, 1)
                all_pred_conf.extend(pred_conf.cpu().detach().numpy())

            elif ood_method == 'perturbation':
                # model inference
                pred = cnn(images)

                # T = 1000
                # pred = pred / T

                # input preprocessing
                xent = nn.CrossEntropyLoss()
                loss = xent(pred, labels)
                loss.backward()

                images = images - 0.005 * torch.sign(images.grad)

                # save the enhanced image
                if task != 'skin' and len(data) == 3:
                    save_images_batch(denorm(images).to("cpu").clone().detach(), paths, task)

                pred = cnn(images)
                pred = F.softmax(pred, dim=-1)

                # get full probability distribution
                full_prob_batch = pred.cpu().detach().numpy()
                all_full_prob.extend(full_prob_batch)

                # get conf score
                pred_conf, pred = torch.max(pred.data, 1)
                all_pred_conf.extend(pred_conf.cpu().detach().numpy())

            elif ood_method == 'odin':
                # model inference
                pred = cnn(images)

                # input preprocessing
                T = 1000
                pred = pred / T
                pred_labels = torch.argmax(pred, 1)

                xent = nn.CrossEntropyLoss()
                loss = xent(pred, pred_labels)
                loss.backward()

                images = images - 0.005 * torch.sign(images.grad)

                # model inference on perturbed input
                pred_T = cnn(images)
                pred_T = F.softmax(pred_T, dim=-1)
                full_prob_batch = pred_T.cpu().detach().numpy()
                all_full_prob.extend(full_prob_batch)

                # get conf score for OOD
                pred_conf, pred = torch.max(pred_T.data, 1)
                all_pred_conf.extend(pred_conf.cpu().detach().numpy())

            elif ood_method == 'cosine':
                # # model inference
                # pred, cos_sim = cnn(images)
                # pred_labels = torch.argmax(pred, 1)
                #
                # # input preprocessing
                # xent = nn.CrossEntropyLoss()
                # loss = xent(pred, pred_labels)
                # loss.backward()
                #
                # images = images - 0.005 * torch.sign(images.grad)

                # re-calculate pred on processed input
                pred, cos_sim = cnn(images)
                pred = F.softmax(pred, dim=-1)

                full_prob_batch = pred.cpu().detach().numpy()
                all_full_prob.extend(full_prob_batch)

                pred_conf, pred = torch.max(pred, 1)
                cosine_conf, _ = torch.max(cos_sim, 1)

                all_pred_conf.extend(pred_conf.cpu().detach().numpy())
                all_cosine_conf.extend(cosine_conf.cpu().detach().numpy())

            else:
                raise RuntimeError('OOD mode not supported')

            all_labels.extend(labels.cpu().detach().numpy())
            if len(data) == 3:
                all_paths.extend(paths)

            correct_batch = (pred == labels).cpu().numpy()
            correct.extend(correct_batch)

        return all_pred_conf, all_labels, correct, all_full_prob, all_paths, all_cosine_conf

    """IND testing"""
    ind_pred_conf, ind_labels, correct, full_prob, paths, _ = get_pred_conf(test_loader)

    # # save conf into pickle for real test data
    # with open('results/%s%s.p' % (model, '' if validation else '_full'), 'wb') as f:
    #     pickle.dump([ind_pred_conf, ind_labels, correct, full_prob, paths], f)

    # get IND accuracy
    correct = np.array(correct).astype(bool)
    test_acc = np.mean(correct)
    print("test_acc: %.4f" % test_acc)

    # get IND multiclass auc score
    # ToDo

    """OOD testing"""
    if ood_method == 'cosine':
        _, ood_labels, _, _, _, ood_pred_conf = get_pred_conf(ood_loader)
    else:
        ood_pred_conf, ood_labels, _, _, _, _ = get_pred_conf(ood_loader)

    fnr, threshold, detection_error, best_threshold, auroc, aupr_in, aupr_out, out_mea_mean = cal_ood_metrics(ind_pred_conf, ood_pred_conf)

    # generate challenge results
    if generate_result:
        print("Generating result csv file %s" % result_csv)

        with open('results/%s' % result_csv, 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for _ind_pred_conf, _full_prob, _paths in zip(ind_pred_conf, full_prob, paths):
                image_name = ntpath.basename(_paths).replace('.jpg', '')

                # compute entropy
                _full_prob = _full_prob[:len(classes)]

                # write into result file

                # filewriter.writerow([image_name] + list(_full_prob) + [0.0])
                # prediction = np.zeros(len(_full_prob))
                # prediction[_ind_pred] = 1.0

                if _ind_pred_conf > threshold:
                    filewriter.writerow([image_name] + list(_full_prob) + [0.0])
                else:
                    filewriter.writerow([image_name] + list(_full_prob) + [0.4])

    # return test_acc, fnr, detection_error, auroc, aupr_in, aupr_out, out_mea_mean

    # ToDo
    # # write wrong prediction into csv file for error analysis
    # if args.error_analysis:
    #     error_indices = np.where(np.array(correct_batch) == 0)[0]
    #
    #     labels_list = labels.tolist()
    #     pred_list = pred.tolist()
    #     with open('logs/%s.error' % filename, 'a') as error_out:
    #         for error_idx in error_indices:
    #             error_out.write('%s,%d,%d\n' % (paths[error_idx], labels_list[error_idx], pred_list[error_idx]))
