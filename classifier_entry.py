# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-10
#
# This file is part of MRI project.
# Base model train/valid/test
#
# This can not be copied and/or distributed
# without the express permission of yilin.shen
# ==================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
import logging
import ntpath

from utils import resource_allocation

"""restrict GPU option"""
# find most open GPU (default use 4 gpus)
gpu_list = resource_allocation.get_default_gpus(4)
gpu_ids = ','.join(map(str, gpu_list))

# allocate GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

print("Allocated GPU %s" % gpu_ids)

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as ml
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from utils import optimizer, dataloader

from models import ensemble_resnets, ensemble_cosine_resnets, resnet, cosine_resnet

from evaluation.ind_classification import ind_eval
from evaluation.eval_classification import ind_eval_io, ood_eval_io
from evaluation.eval_segmentation import segmentation_eval, segmentation_eval_each

from gradcam.main import base_cam, ensemble_cam

"""input arguments"""
dataset_options = ['isic2019', 'cifar10', 'cifar100', 'fashioniq2019']
task_options = ['skin', 'age_approx', 'anatom_site_general', 'sex', 'general']
model_options = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext101_32x8d', 'vgg13', 'vgg16']
model_type_options = ['base', 'cosine', 'ensemble', 'ensemble_cosine']
# optim_options = ['SGD', 'Adam', 'RMSprop']
optim_options = ['SGD']
ood_dataset = ['tinyImageNet_resize', 'LSUN_resize', 'iSUN', 'cifar10', 'cifar100', 'svhn']
ood_options = ['Baseline', 'ODIN', 'Mahalanobis', 'Mahalanobis_IPP', 'DeepMahalanobis', 'DeepMahalanobis_IPP']

parser = argparse.ArgumentParser(description='train_model')

parser.add_argument('--gpu_no', type=int, default=4)
parser.add_argument('--task', default='skin', choices=task_options)
parser.add_argument('--dataset', default='isic2019', choices=dataset_options)
parser.add_argument('--model', default='resnet34', choices=model_options)
parser.add_argument('--model_type', default='ensemble', choices=model_type_options)
parser.add_argument('--ensemble_models', nargs="+", type=float, default=['resnet18', 'resnet34'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--valid_steps', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--optim', default='SGD', choices=optim_options)
parser.add_argument('--learning_rate', type=float, default=0.001)
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

# default: no doing any of these
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')

parser.add_argument('--gradcam', action='store_true')
parser.add_argument('--gradcam_conf', type=float, default=0.95)
parser.add_argument('--gradcam_threshold', type=float, default=0.6)

parser.add_argument('--test_segmentation', action='store_true')

parser.add_argument('--train_augmentation', action='store_true', help='augment train data by color')
parser.add_argument('--test_augmentation', action='store_true', help='augment test data by five crop')
parser.add_argument('--error_analysis', action='store_true')
parser.add_argument('--generate_result', action='store_true')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--ood_dataset', default='LSUN_resize', choices=ood_dataset)
parser.add_argument('--ood_method', default='all', choices=ood_options)
parser.add_argument('--data_perturb_magnitude', nargs="+", type=float, default=[0.01])

# get and show input arguments
args = parser.parse_args()

args_dict = pd.DataFrame(vars(args).items(), columns=["argument", "value"])
print(args_dict)

# support multiple GPU training
if args.gpu_no == 0:
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# should make training should go faster for large models
cudnn.benchmark = True

np.random.seed(0)
torch.cuda.manual_seed(args.seed)

"""define a universal filename"""
folder_name = args.model_type + '_' + args.task

filename = args.dataset + '_' \
           + args.task + '_' \
           + ('aug' if args.train_augmentation else 'noaug') + '_' \
           + args.model_type + '_' \
           + (''.join(args.ensemble_models) if args.model_type == 'ensemble' else args.model) + '_' \
           + str(args.batch_size) + '_' \
           + ('ft' if args.pretrained else '') \
           + ('' if args.validation else '_full')

# initialize tensorboard writer
tensorboard_path = 'logs/%s/%s/%s' % (args.dataset, folder_name, filename)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
tb_writer = SummaryWriter(tensorboard_path)

# initialize log file
logging.basicConfig(filename='logs/%s/%s/%s.log' % (args.dataset, folder_name, filename), level=logging.DEBUG)

"""data loading"""

# read statistics
with open('data_processing/data_statistics.json') as json_file:
    data = json.load(json_file)

    mean = data[args.dataset]['mean']
    std = data[args.dataset]['std']

    print("%s: mean=[%s] std=[%s]" % (args.dataset, ', '.join(map(str, mean)), ', '.join(map(str, std))))

# data normalization
normalize = transforms.Normalize(mean=mean, std=std)

# load training data

if 'isic' in args.dataset:
    if args.train_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    train_data = datasets.ImageFolder(root='data/%s/%s_training%s' % (args.dataset, args.task, '' if args.validation else '_full'), transform=train_transform)
elif args.dataset == 'cifar10':
    if args.train_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_data = datasets.CIFAR10(root='data/standard/', train=True, transform=train_transform, download=False)
elif args.dataset == 'cifar100':
    if args.train_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_data = datasets.CIFAR100(root='data/standard/', train=True, transform=train_transform, download=False)
else:
    raise RuntimeError('Training dataset not available')

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=16)

classes = train_data.classes
num_classes = len(train_data.classes)

# load valid data
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

if 'isic' in args.dataset:
    valid_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    valid_data = datasets.ImageFolder(root='data/%s/%s_validation%s' % (args.dataset, args.task, '' if args.validation else '_full'), transform=valid_transform)
elif args.dataset == 'cifar10':
    valid_data = datasets.CIFAR10(root='data/standard/', train=False, transform=train_transform, download=True)
elif args.dataset == 'cifar100':
    valid_data = datasets.CIFAR100(root='data/standard/', train=False, transform=train_transform, download=True)
else:
    raise RuntimeError('Valid dataset not available')

valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=16)

valid_data_balanced = None
if args.validation and os.path.exists('data/%s/%s_validation_balanced' % (args.dataset, args.task)):
    valid_data_balanced = datasets.ImageFolder(root='data/%s/%s_validation_balanced' % (args.dataset, args.task), transform=valid_transform)
    valid_balanced_loader = torch.utils.data.DataLoader(dataset=valid_data_balanced,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=16)

# load ood data
if args.ood_dataset == 'isic':
    ood_dataset = datasets.ImageFolder(root='data/standard/isic', transform=valid_transform)
elif args.ood_dataset == 'tinyImageNet_resize':
    ood_dataset = datasets.ImageFolder(root='data/standard/TinyImagenet_resize', transform=valid_transform)
elif args.ood_dataset == 'LSUN_resize':
    ood_dataset = datasets.ImageFolder(root='data/standard/LSUN_resize', transform=valid_transform)
elif args.ood_dataset == 'iSUN':
    ood_dataset = datasets.ImageFolder(root='data/standard/iSUN', transform=valid_transform)
elif args.ood_dataset == 'cifar10':
    ood_dataset = datasets.CIFAR10(root='data/standard/', train=False, transform=valid_transform, download=True)
elif args.ood_dataset == 'cifar100':
    ood_dataset = datasets.CIFAR100(root='data/standard/', train=False, transform=valid_transform, download=True)
elif args.ood_dataset == 'svhn':
    ood_dataset = datasets.SVHN(root='data/standard/svhn', split='valid', transform=valid_transform, download=True)
else:
    raise RuntimeError('OOD dataset not available')

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=16)

"""model loading"""

# model architecture selection
if args.model_type == 'base':
    if 'resnet' in args.model:
        cnn = resnet.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
        # cnn = ml.__dict__[args.model](pretrained=args.pretrained)
    else:
        cnn = ml.__dict__[args.model](pretrained=args.pretrained)
elif args.model_type == 'cosine':
    cnn = cosine_resnet.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
    # if 'resnet' in args.model:
    #     cnn = cosine_resnet.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
    # else:
    #     cnn = cosine_net.CosineNet(ml.__dict__[args.model](pretrained=args.pretrained), num_classes)
elif 'ensemble' in args.model_type:

    if 'cosine' in args.model_type:
        single_nets = []

        for ensemble_model in args.ensemble_models:
            ensemble_model_checkpoint = args.dataset + '_' \
                                        + args.task + '_' \
                                        + ('aug' if args.train_augmentation else 'noaug') + '_' \
                                        + 'cosine_' \
                                        + ensemble_model + '_' \
                                        + str(args.batch_size) + '_' \
                                        + ('ft' if args.pretrained else '') \
                                        + ('' if args.validation else '_full')

            print('Loading pretrained %s model...' % ensemble_model)

            cnn_en = nn.DataParallel(cosine_resnet.__dict__[ensemble_model](pretrained=args.pretrained, num_classes=num_classes))
            cnn_en.load_state_dict(torch.load('checkpoints/%s/cosine_%s/%s.pt' % (args.dataset, args.task, ensemble_model_checkpoint)))

            single_nets.append(cnn_en)

        cnn = ensemble_cosine_resnets.CosineResnetEnsemble(single_nets, num_classes)
    
    else:
        ensemble_model_lookup = {2: ensemble_resnets.ResnetEnsemble2,
                                 3: ensemble_resnets.ResnetEnsemble3}

        single_nets = []

        for ensemble_model in args.ensemble_models:
            ensemble_model_checkpoint = args.dataset + '_' \
                                        + args.task + '_' \
                                        + ('aug' if args.train_augmentation else 'noaug') + '_' \
                                        + 'base_' \
                                        + ensemble_model + '_' \
                                        + str(args.batch_size) + '_' \
                                        + ('ft' if args.pretrained else '') \
                                        + ('' if args.validation else '_full')

            print('Loading pretrained %s model...' % ensemble_model)

            cnn_en = nn.DataParallel(resnet.__dict__[ensemble_model](pretrained=args.pretrained, num_classes=num_classes))
            # cnn_en = nn.DataParallel(ml.__dict__[ensemble_model](pretrained=args.pretrained))
            cnn_en.load_state_dict(torch.load('checkpoints/%s/base_%s/%s.pt' % (args.dataset, args.task, ensemble_model_checkpoint)))

            single_nets.append(cnn_en)

        cnn = ensemble_model_lookup[len(args.ensemble_models)](single_nets, num_classes)

else:
    raise RuntimeError('customized model not supported')

# cnn = nn.DataParallel(cnn, device_ids=gpu_list)
cnn = nn.DataParallel(cnn)

# load model if trained before
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

model_file = 'checkpoints/%s/%s/%s.pt' % (args.dataset, folder_name, filename)
if os.path.isfile(model_file):
    pretrained_dict = torch.load(model_file)
    cnn.load_state_dict(pretrained_dict)
    print("Reloading model from {}".format(model_file))
else:
    if args.test:
        raise RuntimeError('model not trained')
    else:
        if not os.path.exists('checkpoints/%s/%s' % (args.dataset, folder_name)):
            os.makedirs('checkpoints/%s/%s' % (args.dataset, folder_name))


def train():
    """train function"""

    # define loss function
    prediction_criterion = nn.NLLLoss().to(device)

    # define optimizer

    if 'cosine' in args.model_type:
        optim = 'SGDNoWeightDecayLast'
    else:
        optim = args.optim

    if optim == 'SGD':
        if args.model.startswith('densenet'):
            cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
            # scheduler = MultiStepLR(cnn_optimizer, milestones=[150, 225], gamma=0.1)
            scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 180], gamma=0.1)
        else:
            cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
            scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
            # scheduler = MultiStepLR(cnn_optimizer, milestones=[100, 150, 200], gamma=0.1)
    # elif args.optim == 'Adam':
    #     cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    elif optim == 'SGDNoWeightDecayLast':
        cnn_optimizer = optimizer.SGDNoWeightDecayLast(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        raise RuntimeError('optimizer not supported')

    if 'ensemble' in args.model_type:
        scheduler = MultiStepLR(cnn_optimizer, milestones=[20, 40, 60, 80, 100], gamma=0.2)

    # start training
    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct_count = 0.
        total = 0.
        train_acc = 0.

        progress_bar = tqdm(train_loader)
        # for i, (images, labels, paths) in enumerate(progress_bar):
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            cnn.to(device)
            cnn.zero_grad()

            if 'cosine' in args.model_type:
                pred_original, _, _ = cnn(images)
            else:
                pred_original = cnn(images)

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

        # valid every 10 epochs
        if epoch % args.valid_steps == 0:
            # write log for tensorboard
            tb_writer.add_scalar('train_loss', xentropy_loss_avg, epoch)
            tb_writer.add_scalar('train_accuracy', train_acc, epoch)

            if args.validation:

                valid_acc, valid_balanced_acc = 0.0, 0.0

                def valid(data_loader, data_desc='valid'):
                    # valid after each epoch
                    valid_performance = ind_eval(args, cnn, data_loader)

                    valid_acc = valid_performance['accuracy(\u2191)']
                    valid_auc = valid_performance['auc(\u2191)']

                    # output accuracy results
                    tqdm.write('%s_acc: %.4f,   %s_auc: %.4f' % (data_desc, valid_acc, data_desc, valid_auc))

                    # write log for tensorboard
                    tb_writer.add_scalar('%s_accuracy' % data_desc, valid_acc, epoch)
                    tb_writer.add_scalar('%s_auc' % data_desc, valid_auc, epoch)

                    return valid_acc

                valid_acc = valid(valid_loader, data_desc='valid')

                # only isic2019 dataset has balanced valid dataset
                if valid_balanced_loader is not None:
                    valid_balanced_acc = valid(valid_balanced_loader, data_desc='valid_balanced')

                # add into log file
                row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc), 'valid_acc': str(valid_acc), 'valid_balanced_acc': str(valid_balanced_acc)}

            else:
                # skip valid if not validation (full training)
                row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc)}

            logging.info(row)

        # save model
        torch.save(cnn.state_dict(), 'checkpoints/%s/%s/%s.pt' % (args.dataset, folder_name, filename))


if args.train:
    train()
else:
    # run test
    if args.test:

        def run_test(test_data_loader):

            print('\n%s\n' % filename)

            """test ind performance"""
            ind_eval_io(args, cnn, test_data_loader)

            """test ood performance"""
            if args.ood_method != 'all':
                ood_methods = [args.ood_method]
            else:
                ood_methods = ood_options

            # test ood
            for ood_method in ood_methods:
                print('\n[%s] ' % ood_method, end='')
                ood_eval_io(args, cnn, train_loader, test_data_loader, ood_loader, classes, ood_method=ood_method)

        # original test/val data
        run_test(valid_loader)

        # # balanced test/val data
        # if valid_balanced_loader is not None:
        #     run_test(valid_balanced_loader)

    # run grad-cam
    gradcam_result_path = 'results/grad_cam/{}/{}/{}/{}_{}'.format(args.dataset, folder_name, filename, args.gradcam_conf, args.gradcam_threshold)
    if args.gradcam:

        # load new validation data
        valid_data = dataloader.ImageFolderWithPaths(root='data/%s/%s_segmentation_validation%s' % (args.dataset, args.task, '' if args.validation else '_full'), transform=valid_transform)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=16)

        if not os.path.exists(gradcam_result_path):
            os.makedirs(gradcam_result_path)

        for image_batch, label_batch, path_batch in valid_loader:

            # skip if already processed
            if not os.path.exists('%s/%s' % (gradcam_result_path, ntpath.basename(path_batch[0]))):
                label_batch = torch.unsqueeze(label_batch, dim=1)

                if args.model_type == 'ensemble':
                    ensemble_cam(list(path_batch), mean, std, label_batch, cnn, gradcam_result_path, args.gradcam_conf, args.gradcam_threshold)
                else:
                    base_cam(list(path_batch), mean, std, label_batch, cnn, gradcam_result_path, args.gradcam_conf, args.gradcam_threshold)

    # evaluate segmentation results
    if args.test_segmentation:
        gt_path = 'data/isic2019/skin_segmentation_resized_validation_groundtruth'

        segmentation_eval(gt_path, gradcam_result_path)
