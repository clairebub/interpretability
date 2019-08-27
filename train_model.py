# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-10
#
# This file is part of MRI project.
# Base model training/testing
#
# This can not be copied and/or distributed
# without the express permission of yilin.shen
# ==================================================

import os
import argparse
import numpy as np
import logging

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as ml
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models, datasets, transforms

from utils import utils_dataloader
from utils import resource_allocation
from models import cosine_net, ensemble_resnets
from test_ood_base_model import test_with_ood

"""input arguments"""
task_options = ['skin', 'age_approx', 'anatom_site_general', 'sex']
model_options = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext101_32x8d', 'vgg13', 'vgg16']
model_customize_options = ['base', 'cosine', 'ensemble']
optim_options = ['SGD', 'Adam', 'RMSprop']
ood_dataset = ['tinyImageNet_resize', 'LSUN_resize', 'iSUN', 'cifar10', 'cifar100', 'svhn']
ood_options = ['base', 'odin', 'cosine']

parser = argparse.ArgumentParser(description='train_model')
parser.add_argument('--gpu_no', type=int, default=4)
parser.add_argument('--task', default='skin', choices=task_options)
parser.add_argument('--dataset', default='isic2019')
parser.add_argument('--ood_dataset', default='cifar10', choices=ood_dataset)
parser.add_argument('--model', default='resnet34', choices=model_options)
parser.add_argument('--model_customize', default='base', choices=model_customize_options)
parser.add_argument('--ood_method', default='odin', choices=ood_options)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--optim', default='SGD', choices=optim_options)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--train_augmentation', type=bool, default=False, help='augment train data by color')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--test_augmentation', type=bool, default=False, help='augment test data by five crop')
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
    gpu_list = resource_allocation.get_default_gpus(args.gpu_no)
    gpu_ids = ','.join(map(str, gpu_list))

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
folder_name = args.model_customize + '_' + args.task

filename = args.dataset + '_' \
           + args.task + '_' \
           + ('aug' if args.train_augmentation else 'noaug') + '_' \
           + args.model_customize + '_' \
           + args.model + '_' \
           + str(args.batch_size) + '_' \
           + ('ft' if args.pretrained else '') \
           + ('' if args.validation else '_full')

# initialize log file
if not os.path.exists('logs/' + folder_name):
    os.makedirs('logs/' + folder_name)

logging.basicConfig(filename='logs/%s/%s.log' % (folder_name, filename), level=logging.DEBUG)

# initialize tensorboard writer
tensorboard_path = 'logs/%s/%s' % (folder_name, filename)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
tb_writer = SummaryWriter(tensorboard_path)


"""data loading"""

# # normalization for imagenet
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# # normalization for isic2019 test data
# normalize = transforms.Normalize(mean=[0.6805612, 0.5264354, 0.5190888], std=[0.20700188, 0.19518976, 0.20581853])

# normalization for isic2019
normalize = transforms.Normalize(mean=[0.6796547, 0.5259538, 0.51874095], std=[0.18123391, 0.18504128, 0.19822954])

# load training data
if args.train_augmentation:
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor(),
                                          normalize])
else:
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

# isic2019_train_data = datasets.ImageFolder(root='data/isic2019/isic2019_training', transform=data_transform)
# isic2019_test_data = datasets.ImageFolder(root='data/isic2019/isic2019_testing', transform=data_transform)

isic2019_train_data = utils_dataloader.ImageFolderWithPaths(root='data/isic2019/isic2019_%s_training%s' % (args.task, '' if args.validation else '_full'), transform=train_transform)

train_loader = torch.utils.data.DataLoader(dataset=isic2019_train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=16)

classes = isic2019_train_data.classes
num_classes = len(isic2019_train_data.classes)

# load testing data
test_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(),
                                     normalize])

isic2019_test_data = utils_dataloader.ImageFolderWithPaths(root='data/isic2019/isic2019_%s_testing%s' % (args.task, '' if args.validation else '_full'), transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=isic2019_test_data,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=16)

if args.validation:
    isic2019_test_data_balanced = utils_dataloader.ImageFolderWithPaths(root='data/isic2019/isic2019_%s_testing_balanced' % args.task, transform=test_transform)
    test_balanced_loader = torch.utils.data.DataLoader(dataset=isic2019_test_data_balanced,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=16)

# load ood data
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

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=16)

"""model loading"""

# model architecture selection
if args.model_customize == 'base':
    cnn = models.__dict__[args.model](pretrained=args.pretrained)
elif args.model_customize == 'cosine':
    cnn = cosine_net.CosineNet(models.__dict__[args.model](pretrained=args.pretrained), num_classes)
elif 'ensemble' in args.model_customize:
    # add the models to be ensembled

    # print('Loading pretrained resnet50 model...')
    # cnn_resnet50 = nn.DataParallel(ml.resnet50(pretrained=args.pretrained))
    # cnn_resnet50.load_state_dict(torch.load('checkpoints/base_skin/isic2019_skin_noaug_base_resnet50_256_ft_full.pt'))

    print('Loading pretrained resnet101 model...')
    cnn_resnet101 = nn.DataParallel(ml.resnet101(pretrained=args.pretrained))
    cnn_resnet101.load_state_dict(torch.load('checkpoints/base_skin/isic2019_skin_noaug_base_resnet101_256_ft_full.pt'))

    print('Loading pretrained resnet152 model...')
    cnn_resnet152 = nn.DataParallel(ml.resnet152(pretrained=args.pretrained))
    cnn_resnet152.load_state_dict(torch.load('checkpoints/base_skin/isic2019_skin_noaug_base_resnet152_256_ft_full.pt'))

    if 'meta' not in args.model_customize:
        cnn = ensemble_resnets.ResnetEnsemble([cnn_resnet101, cnn_resnet152], num_classes)
    else:
        # load meta data pretrained models
        print('Loading pretrained age_approx resnet152 model...')
        cnn_age_approx = nn.DataParallel(ml.resnet152(pretrained=args.pretrained))
        cnn_age_approx.load_state_dict(torch.load('checkpoints/base_age_approx/isic2019_age_approx_noaug_base_resnet152_256_ft_full.pt'))

        print('Loading pretrained anatom_site_general resnet152 model...')
        cnn_anatom_site_general = nn.DataParallel(ml.resnet152(pretrained=args.pretrained))
        cnn_anatom_site_general.load_state_dict(torch.load('checkpoints/base_anatom_site_general/isic2019_anatom_site_general_noaug_base_resnet152_256_ft_full.pt'))

        print('Loading pretrained sex resnet152 model...')
        cnn_sex = nn.DataParallel(ml.resnet152(pretrained=args.pretrained))
        cnn_sex.load_state_dict(torch.load('checkpoints/base_sex/isic2019_sex_noaug_base_resnet152_256_ft_full.pt'))

        cnn = ensemble_resnets.ResnetEnsemble([cnn_resnet101, cnn_resnet152, cnn_age_approx, cnn_anatom_site_general, cnn_sex], num_classes)
else:
    raise RuntimeError('customized model not supported')

# cnn = nn.DataParallel(cnn, device_ids=gpu_list)
cnn = nn.DataParallel(cnn)

# load model if trained before
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

model_file = 'checkpoints/%s/%s.pt' % (folder_name, filename)
if os.path.isfile(model_file):
    pretrained_dict = torch.load(model_file)
    cnn.load_state_dict(pretrained_dict)
    print("Reloading model from {}".format(model_file))
else:
    if not os.path.exists('checkpoints/' + folder_name):
        os.makedirs('checkpoints/' + folder_name)

    if args.test:
        raise RuntimeError('model not trained')


def test(data_loader):
    """test function"""

    # change model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()

    correct = []
    probability = []
    errors = []

    for images, labels, paths in data_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        cnn.to(device)
        cnn.zero_grad()

        pred = cnn(images)
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

    correct = np.array(correct).astype(bool)
    test_acc = np.mean(correct)

    return test_acc


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

        # write log for tensorboard
        tb_writer.add_scalar('Train_Loss', xentropy_loss_avg, epoch)
        tb_writer.add_scalar('Train_Accuracy', train_acc, epoch)

        if args.validation:
            # test after each epoch
            test_acc = test(test_loader)
            test_balanced_acc = test(test_balanced_loader)

            # output accuracy results
            tqdm.write('test_acc: %.4f' % test_acc)
            tqdm.write('test_balanced_acc: %.4f' % test_balanced_acc)

            # write log for tensorboard
            tb_writer.add_scalar('Test_Accuracy', test_acc, epoch)
            tb_writer.add_scalar('Test_Balanced_Accuracy', test_balanced_acc, epoch)

            # add into log file
            row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc), 'test_acc': str(test_acc), 'test_balanced_acc': str(test_balanced_acc)}

        else:
            # skip test if not validation

            # add into log file
            row = {'epoch': str(epoch), 'loss': str(xentropy_loss_avg), 'train_acc': str(train_acc)}

        logging.info(row)

        # save model
        torch.save(cnn.state_dict(), 'checkpoints/%s/%s.pt' % (folder_name, filename))


if args.test:
    if args.model_customize == 'cosine':
        ood_method = 'cosine'
    else:
        ood_method = args.ood_method

    print("Testing on original test data...")
    test_with_ood(cnn, filename, test_loader, ood_loader, classes, ood_method=ood_method, generate_result=args.generate_result, validation=args.validation)

    if args.validation:
        print("\nTesting on balanced test data...")
        test_with_ood(cnn, filename, test_balanced_loader, ood_loader, classes, ood_method=ood_method, generate_result=args.generate_result, validation=args.validation)

else:
    train()
