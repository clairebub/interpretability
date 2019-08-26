# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-10
#
# This file is part of MRI project.
# Cosine network model training/testing
#
# This can not be copied and/or distributed
# without the express permission of yilin.shen
# ==================================================

import os
import argparse
import numpy as np
import logging
import ntpath
import datetime
import csv
from sklearn import metrics

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import torchvision.models as ml

from utils import utils_dataloader
from utils import resource_allocation
from utils.ood_metrics import fpr95, cal_metrics
from models import cosine_net

"""input arguments"""
model_options = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg13', 'vgg16']
optim_options = ['SGD', 'Adam', 'RMSprop']
ood_options = ['tinyImageNet_resize', 'LSUN_resize', 'iSUN', 'cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='train_cosine_model')
parser.add_argument('--gpu_no', type=int, default=4)
parser.add_argument('--dataset', default='isic2019')
parser.add_argument('--ood_dataset', default='LSUN_resize', choices=ood_options)
parser.add_argument('--model', default='resnet34', choices=model_options)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--optim', default='SGD', choices=optim_options)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_augmentation', action='store_true', default=False, help='augment data by color')
parser.add_argument('--test', type=bool, default=True)
parser.add_argument('--test_ood', type=bool, default=True)
parser.add_argument('--error_analysis', type=bool, default=False)
parser.add_argument('--generate_result', type=bool, default=False)
parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--pretrained', type=bool, default=True)

# get and show input arguments
args = parser.parse_args()
print(args)

"""restrict GPU option"""
if args.gpu_no > 0:
    # find most open GPU
    gpu_ids = ','.join(map(str, resource_allocation.get_default_gpus(args.gpu_no)))

    # allocate GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    print("Allocated GPU %s" % gpu_ids)

    # support multiple GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# should make training should go faster for large models
cudnn.benchmark = True

np.random.seed(0)
torch.cuda.manual_seed(args.seed)

"""define a universal filename"""
filename = args.dataset + '_' \
           + ('aug' if args.data_augmentation else 'noaug') + '_' \
           + 'cosine_' + args.model + '_' \
           + str(args.batch_size) + '_' \
           + ('ft' if args.pretrained else '') \
           + ('' if args.validation else '_full')

# initialize log file
if not os.path.exists('logs/cosine_model'):
    os.makedirs('logs/cosine_model')
logging.basicConfig(filename='logs/cosine_model/%s.log' % filename, level=logging.DEBUG)

# initialize tensorboard writer
tensorboard_path = 'logs/cosine_model/%s' % filename
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
tb_writer = SummaryWriter(tensorboard_path)

"""data loading"""

# # normalization for imagenet
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# normalization for isic2019
if args.validation:
    normalize = transforms.Normalize(mean=[0.6803108, 0.5250009, 0.5146185], std=[0.18023035, 0.18443975, 0.19847354])
else:
    normalize = transforms.Normalize(mean=[0.6805612, 0.5264354, 0.5190888], std=[0.18034580, 0.18460624, 0.19756213])

if args.data_augmentation:
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor(),
                                          normalize])
else:
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

test_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(),
                                     normalize])

# isic2019_train_data = datasets.ImageFolder(root='data/isic2019/isic2019_training', transform=data_transform)
# isic2019_test_data = datasets.ImageFolder(root='data/isic2019/isic2019_testing', transform=data_transform)

isic2019_train_data = utils_dataloader.ImageFolderWithPaths(root='data/isic2019/isic2019_training%s' % ('' if args.validation else '_full'), transform=train_transform)
isic2019_test_data = utils_dataloader.ImageFolderWithPaths(root='data/isic2019/isic2019_testing%s' % ('' if args.validation else '_full'), transform=test_transform)

# ood datasets
if args.ood_dataset == 'tinyImageNet_resize':
    ood_dataset = datasets.ImageFolder(root='data/ood/TinyImagenet_resize', transform=test_transform)
elif args.ood_dataset == 'LSUN_resize':
    ood_dataset = datasets.ImageFolder(root='data/ood/LSUN_resize', transform=test_transform)
elif args.ood_dataset == 'iSUN':
    ood_dataset = datasets.ImageFolder(root='data/ood/iSUN', transform=test_transform)
elif args.ood_dataset == 'cifar10':
    ood_dataset = datasets.CIFAR10(root='data/ood/', train=False, transform=test_transform, download=False)
elif args.ood_dataset == 'cifar100':
    ood_dataset = datasets.CIFAR100(root='data/ood/', train=False, transform=test_transform, download=False)
elif args.ood_dataset == 'svhn':
    ood_dataset = datasets.SVHN(root='data/ood/svhn', split='train', transform=test_transform, download=False)
else:
    raise RuntimeError('OOD dataset not available')

# define data loaders
train_loader = torch.utils.data.DataLoader(dataset=isic2019_train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=16)

classes = isic2019_train_data.classes
num_classes = len(isic2019_train_data.classes)

test_loader = torch.utils.data.DataLoader(dataset=isic2019_test_data,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=16)

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=16)

"""model loading"""

# model architecture selection
if args.model == 'densenet121':
    cnn = cosine_net.CosineNet(ml.densenet121(pretrained=args.pretrained), num_classes)
elif args.model == 'densenet161':
    cnn = cosine_net.CosineNet(ml.densenet161(pretrained=args.pretrained), num_classes)
elif args.model == 'densenet169':
    cnn = cosine_net.CosineNet(ml.densenet169(pretrained=args.pretrained), num_classes)
elif args.model == 'densenet201':
    cnn = cosine_net.CosineNet(ml.densenet201(pretrained=args.pretrained), num_classes)
elif args.model == 'resnet18':
    cnn = cosine_net.CosineNet(ml.resnet18(pretrained=args.pretrained), num_classes)
elif args.model == 'resnet34':
    cnn = cosine_net.CosineNet(ml.resnet34(pretrained=args.pretrained), num_classes)
elif args.model == 'resnet50':
    cnn = cosine_net.CosineNet(ml.resnet50(pretrained=args.pretrained), num_classes)
elif args.model == 'resnet101':
    cnn = cosine_net.CosineNet(ml.resnet101(pretrained=args.pretrained), num_classes)
elif args.model == 'vgg13':
    cnn = cosine_net.CosineNet(ml.vgg13(pretrained=args.pretrained), num_classes)
elif args.model == 'vgg16':
    cnn = cosine_net.CosineNet(ml.vgg16(pretrained=args.pretrained), num_classes)
else:
    raise RuntimeError('model not supported')

cnn = nn.DataParallel(cnn)

# load model if trained before
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

model_file = 'checkpoints/cosine_model/%s.pt' % filename
if os.path.isfile(model_file):
    pretrained_dict = torch.load(model_file)
    cnn.load_state_dict(pretrained_dict)
    print("Reloading model from {}".format(model_file))
else:
    if not os.path.exists('checkpoints/cosine_model'):
        os.makedirs('checkpoints/cosine_model')


def test():
    """test function"""

    # create file and input first row
    current_time = datetime.datetime.now()
    if args.generate_result:
        # create new result file
        with open('results/%s.csv' % current_time.strftime("%Y_%m_%d_%H_%M"), 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['image'] + classes + ['UNK'])

    # change model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()

    correct = []
    probability = []
    errors = []

    for images, labels, paths in test_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        cnn.to(device)
        cnn.zero_grad()

        pred, _ = cnn(images)
        pred = F.softmax(pred, dim=-1)
        full_prob_batch = pred.cpu().detach().numpy()

        pred_value, pred = torch.max(pred.data, 1)

        # append into result list
        correct_batch = (pred == labels).cpu().numpy()
        correct.extend(correct_batch)
        prob_batch = pred_value.cpu().numpy()
        probability.extend(prob_batch)

        # write wrong prediction into csv file for error analysis
        if args.error_analysis:
            error_indices = np.where(np.array(correct_batch) == 0)[0]

            labels_list = labels.tolist()
            pred_list = pred.tolist()
            with open('logs/%s.error' % filename, 'a') as error_out:
                for error_idx in error_indices:
                    error_out.write('%s,%d,%d\n' % (paths[error_idx], labels_list[error_idx], pred_list[error_idx]))

        # generate challenge results
        if args.generate_result:
            with open('results/%s.csv' % current_time.strftime("%Y_%m_%d_%H_%M"), 'a') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                for full_prob, path in zip(full_prob_batch, paths):
                    image_name = ntpath.basename(path)

                    # compute entropy
                    full_prob = full_prob[:len(classes)]
                    entropy = -np.sum(full_prob * np.log2(full_prob + 1e-12))

                    # write into result file
                    if entropy > 2.5:
                        filewriter.writerow([image_name] + list(full_prob) + [entropy / 3])
                    else:
                        filewriter.writerow([image_name] + list(full_prob) + [0.05])

    correct = np.array(correct).astype(bool)
    test_acc = np.mean(correct)

    # output accuracy results
    tqdm.write('test_acc: %.4f' % test_acc)

    return test_acc


def test_ood():
    """test function for OOD detection"""

    # change model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()

    def get_cosine_conf(data_loader):
        all_pred_conf, all_cosine_conf, all_labels = [], [], []

        for data in data_loader:
            if len(data) == 3:
                images, labels, paths = data
            else:
                images, labels = data

            images = Variable(images).to(device)
            # labels = Variable(labels).to(device)

            cnn.to(device)
            cnn.zero_grad()

            scaled_cosine, cos_sim = cnn(images)

            pred = F.softmax(scaled_cosine, dim=-1)
            pred_conf, pred = torch.max(pred, 1)

            cosine_conf, _ = torch.max(cos_sim, 1)

            all_pred_conf.extend(pred_conf.cpu().detach().numpy())
            all_cosine_conf.extend(cosine_conf.cpu().detach().numpy())
            all_labels.extend(labels)

        return all_pred_conf, all_cosine_conf, all_labels

    ind_pred_conf, ind_cosine_conf, ind_labels = get_cosine_conf(test_loader)
    ood_pred_conf, ood_cosine_conf, ood_labels = get_cosine_conf(ood_loader)

    fpr = fpr95(ind_cosine_conf, ood_cosine_conf)
    print("Cosine FNR@TPR95 (lower is better): ", 1 - fpr)

    cal_metrics(ind_pred_conf, ood_pred_conf)


def train():
    """train function"""

    # define loss function
    prediction_criterion = nn.NLLLoss().to(device)

    # define optimizer

    if args.optim == 'SGD':
        if args.model.startswith('densenet'):
            cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
            # scheduler = MultiStepLR(cnn_optimizer, milestones=[150, 225], gamma=0.1)
            scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 180], gamma=0.1)
        else:
            cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
            scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
    # elif args.optim == 'Adam':
    #     cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    else:
        raise RuntimeError('optimizer not supported')

    # start training
    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct_count = 0.
        total = 0.
        train_acc = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels, paths) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            cnn.to(device)
            cnn.zero_grad()

            pred_original, _ = cnn(images)
            pred_original = torch.softmax(pred_original, dim=-1)

            # make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            pred_new = torch.log(pred_original)

            xentropy_loss = prediction_criterion(pred_new, labels)

            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            pred_idx = torch.argmax(pred_original.data, 1)
            total += labels.size(0)
            correct_count += (pred_idx == labels.data).sum()
            train_acc = correct_count.item() / total

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % train_acc)

        scheduler.step(epoch)

        # write log for tensorboard
        tb_writer.add_scalar('Training_Loss', xentropy_loss_avg, epoch)
        tb_writer.add_scalar('Training_Accuracy', train_acc, epoch)

        if args.validation:
            # test after each epoch
            test_acc = test()

            # write log for tensorboard
            tb_writer.add_scalar('Testing_Accuracy', test_acc, epoch)

            # add into log file
            row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc), 'test_acc': str(test_acc)}

        else:
            # skip test if not validation

            # add into log file
            row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc)}

        logging.info(row)

        # save model
        torch.save(cnn.state_dict(), 'checkpoints/cosine_model/%s.pt' % filename)


if args.test:
    # test()

    if args.test_ood:
        test_ood()
else:
    train()
