# Modified based on http://kazuto1011.github.io

from __future__ import print_function

import shutil
import ntpath
import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
from torchvision import transforms

import sys
sys.path.append("..")


from gradcam.grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    return device


def preprocess(image_path, mean, std):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)

    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    # gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_segmentation(filename, gcam, threshold):
    # gcam = gcam.cpu().numpy()

    # prune with threshold
    gcam_binary = np.copy(gcam)
    gcam_binary[gcam <= threshold] = 1.0
    gcam_binary[gcam > threshold] = 0.0

    cmap = cm.binary(gcam_binary)[..., :3] * 255.0
    gcam = cmap.astype(np.float)

    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    # maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


def base_cam(image_inputs, mean, std, labels, cnn, output_dir, conf, threshold, topk=1, cuda=True):
    """Generate Grad-CAM with original models"""

    device = get_device(cuda)

    cnn.to(device)
    cnn.eval()

    # # Check available layer names
    # print("Layers:")
    # for m in model.named_modules():
    #     print("\t", m[0])

    # Here we choose the last convolution layer
    # target_layers = ["module.relu", "module.layer1", "module.layer2", "module.layer3", "module.layer4"]
    target_layers = ["module.layer4"]

    # Images
    image_names = []
    image_paths = []
    images = []
    raw_images = []

    for i, image_path in enumerate(image_inputs):
        image_paths.append(image_path)

        image_names.append(ntpath.basename(image_path).replace('.jpg', ''))

        image, raw_image = preprocess(image_path, mean, std)
        images.append(image)
        raw_images.append(raw_image)

    images = torch.stack(images).to(device)
    # labels = labels.to(device)

    gcam = GradCAM(model=cnn)
    probs, ids = gcam.forward(images)

    # Grad-CAM
    for target_layer in target_layers:

        for i in range(topk):

            # use predicted class labels
            gcam.backward(ids=ids[:, [i]])

            # # use ground truth class labels
            # gcam.backward(ids=labels)

            regions = gcam.generate(target_layer=target_layer)

            for j in range(len(images)):

                # copy original image
                shutil.copy(image_paths[j], output_dir)

                print("\t#{}: {} ({:.5f})".format(image_names[j], ids[j, i], probs[j, i]))

                save_gradcam(
                    filename=osp.join(
                        output_dir,
                        "{}-gradcam-{}-{}-{}.png".format(
                            image_names[j], target_layer.replace('module.', ''), labels[j, i], ids[j, i]
                        ),
                    ),
                    gcam=regions[j, 0].cpu().numpy(),
                    raw_image=raw_images[j]
                )

                # ToDo: threshold to be learned
                if probs[j, i] > conf:
                    save_segmentation(
                        filename=osp.join(
                            output_dir,
                            "{}_segmentation.png".format(image_names[j]),
                        ),
                        gcam=regions[j, 0].cpu().numpy(),
                        threshold=threshold
                    )


def ensemble_cam(image_inputs, mean, std, labels, cnn, output_dir, conf, threshold, gradcam_alg='grad_pooling', topk=1, cuda=True):
    """
    Generate Grad-CAM with original models
    """

    device = get_device(cuda)

    cnn.to(device)
    cnn.eval()

    # # Check available layer names
    # print("Layers:")
    # for m in model.named_modules():
    #     print("\t", m[0])

    # Here we choose the last convolution layer
    target_layers = ["module.resnet_en1.7", "module.resnet_en2.7"]

    # Images
    image_names = []
    image_paths = []
    images = []
    raw_images = []

    for i, image_path in enumerate(image_inputs):
        image_paths.append(image_path)

        image_names.append(ntpath.basename(image_path).replace('.jpg', ''))

        image, raw_image = preprocess(image_path, mean, std)
        images.append(image)
        raw_images.append(raw_image)

    images = torch.stack(images).to(device)
    # labels = labels.to(device)

    gcam = GradCAM(model=cnn)
    probs, ids = gcam.forward(images)

    gcam.backward(ids=ids[:, [0]])
    regions = gcam.generate_ensemble(target_layers=target_layers)

    # regions_layers = []
    # for target_layer in target_layers:
    #     # print("Generating Grad-CAM @{}".format(target_layer))
    #
    #     # Grad-CAM
    #
    #     # # use predicted class labels
    #     gcam.backward(ids=ids[:, [0]])
    #
    #     # use ground truth class labels
    #     # gcam.backward(ids=labels)
    #
    #     regions = gcam.generate(target_layer=target_layer)
    #     regions_layers.append(regions)
    #
    # regions_layers = torch.stack(regions_layers, dim=0)
    # # regions = regions_layers[1]
    # # print(regions_layers.shape)

    for j in range(len(images)):
        # avg_regions = regions
        # avg_regions = regions_layers.mean(dim=0)
        # print(avg_regions.shape)

        # copy original image
        shutil.copy(image_paths[j], output_dir)

        print("\t#{}: {} ({:.5f})".format(image_names[j], ids[j, 0], probs[j, 0]))

        probs = probs.cpu().detach().numpy()
        ids = ids.cpu().detach().numpy()
        regions = regions.cpu().numpy()

        save_gradcam(
            filename=osp.join(
                output_dir,
                "{}-gradcam-{}-{}-{}.png".format(
                    image_names[j], target_layers[0].replace('module.', ''), labels[j, 0], ids[j, 0]
                ),
            ),
            gcam=regions[j, 0],
            raw_image=raw_images[j]
        )

        # ToDo: threshold to be learned
        if probs[j, 0] > conf:
            save_segmentation(
                filename=osp.join(
                    output_dir,
                    "{}_segmentation.png".format(image_names[j]),
                ),
                gcam=regions[j, 0],
                threshold=threshold
            )
