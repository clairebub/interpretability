#
# Author: Claire Tang (claire.smurfs@gmail.com)
# General entry for interpret2
#
import os
import json
import argparse
import numpy as np
import pandas as pd
import logging
import ntpath

from utils import resource_allocation

"""restrict GPU option"""
# find most open GPU (default use 8 gpus)
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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_processing import mrnet

from utils import optimizer, classifier_dataloader

from models.classification import ensemble_resnets, resnet, densenet, cosine_resnet
from models.multiview_classification import mvresnet, mvcnn

from evaluation.ind_classification import ind_eval
from evaluation.eval_classification import ind_eval_io, ood_eval_io
from evaluation.eval_segmentation import segmentation_eval, segmentation_eval_each
from evaluation import ood_detection

from gradcam.entry import base_cam, ensemble_cam, multiview_cam

"""input arguments"""
dataset_options = ['isic2019', 'cifar10', 'cifar100', 'fashioniq2019', 'mrnet']
# subdataset_options = ['skin', 'age_approx', 'anatom_site_general', 'sex', 'general']
model_options = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext101_32x8d', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg13', 'vgg16']
model_type_options = ['base', 'cosine', 'ensemble', 'ensemble_cosine', 'multiview_pooling', 'multiview_cat']
optim_options = ['SGD', 'Adam', 'RMSprop']
# optim_options = ['SGD']
ood_dataset = ['isic', 'isic_ind', 'tinyImageNet_resize', 'LSUN_resize', 'iSUN', 'cifar10', 'cifar100', 'svhn']
# ood_options = ['Baseline', 'InputPreProcess', 'ODIN', 'Mahalanobis', 'Mahalanobis_IPP', 'DeepMahalanobis', 'DeepMahalanobis_IPP']
ood_options = ['Baseline', 'InputPreProcess', 'ODIN', 'Mahalanobis_IPP']
# ood_options = ['Baseline']

parser = argparse.ArgumentParser(description='train_model')

parser.add_argument('--gpu_no', type=int, default=8)

# parser.add_argument('--dataset', default='isic2019', choices=dataset_options)
# parser.add_argument('--sub_dataset', default='skin')
parser.add_argument('--dataset', default='chexpert', choices=dataset_options)
parser.add_argument('--sub_dataset', default='edema_frontal')

parser.add_argument('--model', default='resnet50', choices=model_options)
parser.add_argument('--model_type', default='base', choices=model_type_options)
parser.add_argument('--ensemble_models', nargs="+", type=float, default=['resnet101', 'resnet152'])

# parser.add_argument('--model', default='mvresnet34', choices=model_options)
# parser.add_argument('--model_type', default='multiview_cat', choices=model_type_options)
# # parser.add_argument('--multiview_model_type', default='pooling', choices=model_type_options)

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--valid_steps', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--optim', default='SGD', choices=optim_options)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')

# default: no doing any of these
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--save_all_models', action='store_true')

parser.add_argument('--gradcam', action='store_true')
parser.add_argument('--ood_gradcam', action='store_true')
parser.add_argument('--test_gradcam', action='store_true')
parser.add_argument('--gradcam_dataset', default='')
parser.add_argument('--gradcam_conf', type=float, default=0.95)
parser.add_argument('--gradcam_threshold', type=float, default=0.6)

parser.add_argument('--test_segmentation', action='store_true')

parser.add_argument('--train_augmentation', action='store_true', help='augment train data by color')
parser.add_argument('--test_augmentation', action='store_true', help='augment test data by five crop')
parser.add_argument('--error_analysis', action='store_true')
parser.add_argument('--generate_result', action='store_true')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--test_ood', action='store_true')
parser.add_argument('--ood_dataset', default='ood', choices=ood_dataset)
parser.add_argument('--ood_method', default='all', choices=ood_options)
# parser.add_argument('--data_perturb_magnitude', nargs="+", type=float, default=[0.0012])
parser.add_argument('--data_perturb_magnitude', nargs="+", type=float, default=[0.01])

parser.add_argument('--best_model', type=int, default=-1)

# get and show input arguments
args = parser.parse_args()

args_dict = pd.DataFrame(vars(args).items(), columns=["argument", "value"])
print(args_dict)

# support multiple GPU training
if args.gpu_no == 0:
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# should make training faster for large models
cudnn.benchmark = True

np.random.seed(0)
torch.cuda.manual_seed(args.seed)

"""define a universal filename"""
folder_name = args.model_type + '_' + args.sub_dataset

filename = args.dataset + '_' \
           + args.sub_dataset + '_' \
           + ('aug' if args.train_augmentation else 'noaug') + '_' \
           + args.model_type + '_' \
           + (''.join(args.ensemble_models) if args.model_type == 'ensemble' else args.model) + '_' \
           + str(args.batch_size) \
           + ('_ft' if args.pretrained else '') \
           + ('' if args.validation else '_full')

# initialize tensorboard writer
tensorboard_path = 'logs/classification/%s/%s/%s' % (args.dataset, folder_name, filename)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
tb_writer = SummaryWriter(tensorboard_path)

# initialize log file
logging.basicConfig(filename='logs/classification/%s/%s/%s.log' % (args.dataset, folder_name, filename), level=logging.DEBUG)

"""data loading"""

# read statistics
with open('data_processing/data_statistics.json') as json_file:
    data = json.load(json_file)

    entry = '%s_%s' % (args.dataset, args.sub_dataset)
    mean = data[entry]['mean']
    std = data[entry]['std']

    print("%s: mean=[%s] std=[%s]" % (args.dataset, ', '.join(map(str, mean)), ', '.join(map(str, std))))

# data normalization
normalize = transforms.Normalize(mean=mean, std=std)

# load train data
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

if 'multiview' in args.model_type:

    train_data = classifier_dataloader.MultiViewDataSet(root='data/%s/%s_training%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full'), transform=train_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16)

    # load valid data
    valid_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    valid_data = classifier_dataloader.MultiViewDataSet(root='data/%s/%s_validation%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full'), transform=valid_transform)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=16)
    valid_balanced_loader = None

else:

    # load training data

    if args.dataset == 'mrnet':
        train_data, valid_data = mrnet.load_data()
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
    elif args.dataset == 'chexpert':
        train_transform = transforms.Compose([
            transforms.Resize(size=(320, 320)),
            transforms.ToTensor(),
            normalize
        ])

        train_data = datasets.ImageFolder(root='data/%s/%s_training%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full'), transform=train_transform)
    else:
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

        train_data = datasets.ImageFolder(root='data/%s/%s_training%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full'), transform=train_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16)

    # load valid data
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'mrnet':
        pass
    elif args.dataset == 'cifar10':
        valid_data = datasets.CIFAR10(root='data/standard/', train=False, transform=train_transform, download=True)
    elif args.dataset == 'cifar100':
        valid_data = datasets.CIFAR100(root='data/standard/', train=False, transform=train_transform, download=True)
    elif args.dataset == 'chexpert':
        valid_transform = transforms.Compose([
            transforms.Resize(size=(320, 320)),
            transforms.ToTensor(),
            normalize
        ])

        valid_data = datasets.ImageFolder(root='data/%s/%s_validation%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full'), transform=valid_transform)
    else:
        valid_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        valid_data = datasets.ImageFolder(root='data/%s/%s_validation%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full'), transform=valid_transform)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=16)

    valid_data_balanced, valid_balanced_loader = None, None
    if args.validation and os.path.exists('data/%s/%s_validation_balanced' % (args.dataset, args.sub_dataset)):
        valid_data_balanced = datasets.ImageFolder(root='data/%s/%s_validation_balanced' % (args.dataset, args.sub_dataset), transform=valid_transform)
        valid_balanced_loader = torch.utils.data.DataLoader(dataset=valid_data_balanced,
                                                            batch_size=args.batch_size,
                                                            shuffle=False,
                                                            num_workers=16)

    # load ood data

    # run for each ood method
    if args.ood_method != 'all':
        ood_methods = [args.ood_method]
    else:
        ood_methods = ood_options

    if args.test_ood:
        if args.ood_dataset == 'tinyImageNet_resize':
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
            ood_dataset = datasets.ImageFolder(root='data/%s/%s_ood' % (args.dataset, args.sub_dataset), transform=valid_transform)

        ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=16)

classes = train_data.classes
num_classes = len(train_data.classes)

"""model loading"""

# model architecture selection
if args.model_type == 'base':
    if 'resnet' in args.model:
        cnn = resnet.__dict__[args.model](pretrained=args.pretrained, num_classes=1000)
        # cnn = resnet.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
    elif 'densenet' in args.model:
        cnn = densenet.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
    else:
        cnn = ml.__dict__[args.model](pretrained=args.pretrained)
elif args.model_type == 'cosine':
    cnn = cosine_resnet.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
elif 'ensemble' in args.model_type:

    ensemble_model_lookup = {2: ensemble_resnets.ResnetEnsemble2,
                             3: ensemble_resnets.ResnetEnsemble3}

    single_nets = []

    for ensemble_model in args.ensemble_models:
        ensemble_model_checkpoint = args.dataset + '_' \
                                    + args.sub_dataset + '_' \
                                    + ('aug' if args.train_augmentation else 'noaug') + '_' \
                                    + 'base_' \
                                    + ensemble_model + '_' \
                                    + str(args.batch_size) + '_' \
                                    + ('ft' if args.pretrained else '') \
                                    + ('' if args.validation else '_full')

        print('Loading pretrained %s model...' % ensemble_model)

        cnn_en = nn.DataParallel(resnet.__dict__[ensemble_model](pretrained=args.pretrained, num_classes=1000))
        # cnn_en = nn.DataParallel(resnet.__dict__[ensemble_model](pretrained=args.pretrained, num_classes=num_classes))
        cnn_en.load_state_dict(torch.load('checkpoints/classification/%s/base_%s/%s.pt' % (args.dataset, args.sub_dataset, ensemble_model_checkpoint)))

        single_nets.append(cnn_en)

    cnn = ensemble_model_lookup[len(args.ensemble_models)](single_nets, num_classes)

elif 'multiview' in args.model_type:
    multiview_model_type = args.model_type.split('_')[-1]

    cnn = mvresnet.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes, multiview_model_type=multiview_model_type)
else:
    raise RuntimeError('customized model not supported')

# cnn = nn.DataParallel(cnn, device_ids=gpu_list)
cnn = nn.DataParallel(cnn)

# ### test test
# modules = []
# for name, module in cnn.named_modules():
#     modules.append(name)

# load model if trained before
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if args.best_model == -1:
    model_file = 'checkpoints/classification/%s/%s/%s.pt' % (args.dataset, folder_name, filename)
else:
    model_file = 'checkpoints/classification/%s/%s/%s__%d.pt' % (args.dataset, folder_name, filename, args.best_model)

print(model_file)

if os.path.isfile(model_file):
    print("Reloading model from {}".format(model_file))

    pretrained_dict = torch.load(model_file)
    cnn.load_state_dict(pretrained_dict)
else:
    if args.test:
        raise RuntimeError('model not trained')
    else:
        if not os.path.exists('checkpoints/classification/%s/%s' % (args.dataset, folder_name)):
            os.makedirs('checkpoints/classification/%s/%s' % (args.dataset, folder_name))


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
        # cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
        # scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

        if args.model.startswith('densenet'):
            cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
            # scheduler = MultiStepLR(cnn_optimizer, milestones=[150, 225], gamma=0.1)
            scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 180], gamma=0.1)
        else:
            cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
            scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
            # scheduler = MultiStepLR(cnn_optimizer, milestones=[100, 150, 200], gamma=0.1)
    elif args.optim == 'Adam':
        cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
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

            if 'multiview' in args.model_type:
                images = torch.stack(images, dim=1)

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
            pred = torch.log(pred_original)

            xentropy_loss = prediction_criterion(pred, labels)

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

                    return valid_acc, valid_auc

                valid_acc, valid_auc = valid(valid_loader, data_desc='valid')

                # only isic2019 dataset has balanced valid dataset
                if valid_balanced_loader is not None:
                    valid_balanced_acc, valid_balanced_auc = valid(valid_balanced_loader, data_desc='valid_balanced')

                # add into log file
                row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc),
                       'valid_acc': str(valid_acc), 'valid_auc': str(valid_auc), 'valid_balanced_acc': str(valid_balanced_acc), 'valid_balanced_auc': str(valid_balanced_auc)}

            else:
                # skip valid if not validation (full training)
                row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc)}

            logging.info(row)

        # save model
        if args.save_all_models:
            torch.save(cnn.state_dict(), 'checkpoints/classification/%s/%s/%s_%d.pt' % (args.dataset, folder_name, filename, epoch))

        torch.save(cnn.state_dict(), 'checkpoints/classification/%s/%s/%s.pt' % (args.dataset, folder_name, filename))


def test():

    print('\n%s\n' % filename)

    # """test ind performance"""
    ind_eval_io(args, cnn, valid_loader)
    # if valid_balanced_loader is not None:
    #     ind_eval_io(args, cnn, valid_balanced_loader)

    """test ood performance"""
    if args.test_ood:
        if args.ood_method != 'all':
            ood_methods = [args.ood_method]
        else:
            ood_methods = ood_options

        # test ood
        for ood_method in ood_methods:
            print('\n[%s] ' % ood_method, end='')
            ood_eval_io(args, cnn, train_loader, valid_loader, ood_loader, classes, ood_method=ood_method)


if args.train:
    train()
else:
    # run test
    if args.test:
        test()

    # run grad-cam
    segmentation_valid_path = 'data/%s/%s_segmentation_validation%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full')

    if os.path.exists(segmentation_valid_path):
        # prepare for segmentation test

        if args.best_model == -1:
            gradcam_result_path = 'results/grad_cam/{}/{}/{}/{}_{}'.format(args.dataset, folder_name, filename, args.gradcam_conf, args.gradcam_threshold)
        else:
            gradcam_result_path = 'results/grad_cam/{}/{}/{}_{}/{}_{}'.format(args.dataset, folder_name, filename, args.best_model, args.gradcam_conf, args.gradcam_threshold)
    else:
        # no need segmentation test

        if args.best_model == -1:
            gradcam_result_path = 'results/grad_cam/{}/{}/{}'.format(args.dataset, folder_name, filename)
        else:
            gradcam_result_path = 'results/grad_cam/{}/{}/{}_{}'.format(args.dataset, folder_name, filename, args.best_model)

        segmentation_valid_path = 'data/%s/%s_validation%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full')

    if args.gradcam:

        if not os.path.exists(gradcam_result_path):
            os.makedirs(gradcam_result_path)

        # load new validation data
        if 'multiview' in args.model_type:
            gradcam_dataset = classifier_dataloader.MultiViewDataSetWithPaths(root=segmentation_valid_path, transform=valid_transform)
        else:
            gradcam_dataset = classifier_dataloader.ImageFolderWithPaths(root=segmentation_valid_path, transform=valid_transform)
            # gradcam_dataset = classifier_dataloader.ImageFolderWithPaths(root='data/%s/%s_ood' % (args.dataset, args.sub_dataset), transform=valid_transform)

        gradcam_loader = torch.utils.data.DataLoader(dataset=gradcam_dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=16)

        for image_batch, label_batch, path_batch in gradcam_loader:

            if 'multiview' in args.model_type:
                path_token = path_batch[0][0].split('/')

                # skip if already processed the first view (others by default)
                if not os.path.exists('%s/%s' % (gradcam_result_path, '/'.join(path_token[-3:]))):
                    multiview_cam(image_batch, label_batch, path_batch, cnn, gradcam_result_path, args.gradcam_conf, args.gradcam_threshold)
            else:
                # skip if already processed
                # if not os.path.exists('%s/%s' % (gradcam_result_path, ntpath.basename(path_batch[0]).replace('jpg', 'png'))):
                if not os.path.exists('%s/%s' % (gradcam_result_path, ntpath.basename(path_batch[0]))):
                    label_batch = torch.unsqueeze(label_batch, dim=1)

                    if args.model_type == 'base':
                        base_cam(list(path_batch), mean, std, label_batch, cnn, gradcam_result_path, args.gradcam_conf, args.gradcam_threshold)
                    elif args.model_type == 'ensemble':
                        # print('Generating cam for model 1 ...')
                        # base_cam(list(path_batch), mean, std, label_batch, cnn, gradcam_result_path, args.gradcam_conf, args.gradcam_threshold, target_layer="module.resnet_en1.7")
                        print('Generating cam for model 2 ...')
                        base_cam(list(path_batch), mean, std, label_batch, cnn, gradcam_result_path, args.gradcam_conf, args.gradcam_threshold, target_layer="module.resnet_en2.7")

                        # print('Generating cam for ensemble model ...')
                        # ensemble_cam(list(path_batch), mean, std, label_batch, cnn, gradcam_result_path, args.gradcam_conf, args.gradcam_threshold)
                    else:
                        raise RuntimeError('customized model not supported')

    # run ood_gradcam
    if args.ood_gradcam:

        for ood_method in ood_methods:
            print('[%s]' % ood_method)

            gradcam_ood_result_path = 'results/grad_cam/{}/{}/{}/{}/{}/{}_{}'.format(args.dataset, folder_name, args.ood_dataset, ood_method, filename, args.gradcam_conf, args.gradcam_threshold)
            if not os.path.exists(gradcam_ood_result_path):
                os.makedirs(gradcam_ood_result_path)

            ood_method = ood_detection.__dict__[ood_method](args, cnn, train_loader, valid_loader, len(classes))
            ood_cnn, perturb_magnitude = ood_method.prepare_ood()

            if args.ood_dataset == 'ood':
                # ood_gradcam_dataset = datasets.ImageFolder(root='data/isic2019/%s_ood' % args.sub_dataset, transform=valid_transform)
                # ood_gradcam_dataset_with_path = classifier_dataloader.ImageFolderWithPaths(root='data/isic2019/%s_ood' % args.sub_dataset, transform=valid_transform)
                ood_gradcam_dataset = classifier_dataloader.ImageFolderWithPaths(root='data/%s/%s_ood' % (args.dataset, args.sub_dataset), transform=valid_transform)
            elif args.ood_dataset == 'ind':
                ood_gradcam_dataset = classifier_dataloader.ImageFolderWithPaths(root='data/%s/%s_segmentation_validation%s' % (args.dataset, args.sub_dataset, '' if args.validation else '_full'), transform=valid_transform)
            elif args.ood_dataset == 'tinyImageNet_resize':
                ood_gradcam_dataset = classifier_dataloader.ImageFolderWithPaths(root='data/standard/TinyImagenet_resize', transform=valid_transform)
            elif args.ood_dataset == 'LSUN_resize':
                ood_gradcam_dataset = classifier_dataloader.ImageFolderWithPaths(root='data/standard/LSUN_resize', transform=valid_transform)
            elif args.ood_dataset == 'iSUN':
                ood_gradcam_dataset = classifier_dataloader.ImageFolderWithPaths(root='data/standard/iSUN', transform=valid_transform)
            else:
                raise RuntimeError('OOD dataset not available')

            ood_gradcam_loader = torch.utils.data.DataLoader(dataset=ood_gradcam_dataset,
                                                             batch_size=1,
                                                             shuffle=False,
                                                             num_workers=16)

            for image_batch, label_batch, path_batch in ood_gradcam_loader:
                # if not os.path.exists('%s/%s' % (gradcam_ood_result_path, ntpath.basename(path_batch[0]).replace('jpg', 'png'))):
                if not os.path.exists('%s/%s' % (gradcam_ood_result_path, ntpath.basename(path_batch[0]))):
                    # ood_method.get_scores(ood_gradcam_dataset)
                    # ood_method.get_scores(ood_gradcam_dataset_with_path)
                    base_cam(list(path_batch), mean, std, label_batch, ood_cnn, gradcam_ood_result_path, args.gradcam_conf, args.gradcam_threshold, perturb_magnitude=perturb_magnitude)

    # evaluate gradcam via accuracy
    if args.test_gradcam:

        if 'multiview' in args.model_type:

            valid_gradcam_data = classifier_dataloader.MultiViewDataSet(root=gradcam_result_path, transform=valid_transform)

            valid_gradcam_loader = torch.utils.data.DataLoader(dataset=valid_gradcam_data,
                                                               batch_size=args.batch_size,
                                                               shuffle=False,
                                                               num_workers=16)
        else:
            raise RuntimeError('model not supported')

        test(valid_gradcam_loader)

    # evaluate segmentation results
    if args.test_segmentation:
        gt_path = 'data/isic2019/skin_segmentation_validation_groundtruth_resized'

        # for ood method generated segmentation
        for ood_method in ood_methods:
            gradcam_ood_result_path = 'results/grad_cam/{}/{}/{}/{}/{}/{}_{}'.format(args.dataset, folder_name, args.ood_dataset, ood_method, filename, args.gradcam_conf, args.gradcam_threshold)
            segmentation_eval(gt_path, gradcam_ood_result_path)

        # # for ind method (gradcam) generated segmentation
        # segmentation_eval(gt_path, gradcam_result_path)
