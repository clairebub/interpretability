#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class _BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model, perturb_magnitude=0.0):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

        self.perturb_magnitude = perturb_magnitude

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """
        Simple classification
        """
        self.model.eval()

        if self.perturb_magnitude == 0.0:
            self.model.zero_grad()
            self.logits = self.model(image)
            self.probs = F.softmax(self.logits, dim=1)

            return image, self.probs.sort(dim=1, descending=True)
        else:
            image = image.requires_grad_(True)

            org_logits = self.model(image)
            probs = F.softmax(org_logits, dim=1)
            score, _ = probs.max(dim=1)

            loss = -score.mean()
            loss.backward()

            gradient = torch.ge(image.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            perturbed_image = torch.add(image.detach(), -self.perturb_magnitude, gradient)

            self.model.zero_grad()
            self.logits = self.model(perturbed_image)
            self.probs = F.softmax(self.logits, dim=1)

            return perturbed_image, self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)
        # (self.logits * one_hot).sum().backward(retain_graph=True)

    # def generate(self):
    #     raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, perturb_magnitude=0.0, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        self.perturb_magnitude = perturb_magnitude

        # self.grad_list = []
        # def print_grad(grad):
        #     self.grad_list.append(grad)

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save feature maps
                self.fmap_pool.setdefault(key, []).append(output.detach())
                # self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the feature maps
                self.grad_pool.setdefault(key, []).append(grad_out[0].detach())
                # self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # if any candidates are not specified, the hook is registered to all the layers
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

                # module.register_backward_hook(print_grad)

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image, multiview=False):
        if multiview is False:
            self.image_shape = image.shape[2:]
        else:
            self.image_shape = image.shape[3:]

        return super(GradCAM, self).forward(image)

    def generate_base(self, target_layer):
        # fmaps = self._find(self.fmap_pool, target_layer)[0]
        # grads = self._find(self.grad_pool, target_layer)[0]

        fmaps = torch.cat(self._find(self.fmap_pool, target_layer), dim=0)
        grads = torch.cat(self._find(self.grad_pool, target_layer), dim=0)
        weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps.cuda(), weights.cuda()).sum(dim=1, keepdim=True)
        # gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

    def generate_ensemble(self, target_layers):
        # ToDo: To be generalized
        # ToDo: Current code support two target_layers with same size

        fmaps, grads, weights = [], [], []

        for target_layer in target_layers:
            fmaps.append(self._find(self.fmap_pool, target_layer)[0])
            grads = self._find(self.grad_pool, target_layer)[0]
            weights.append(self._compute_grad_weights(grads))

        # get the max elements in weights
        max_weights = torch.max(weights[0], weights[1])

        weights0_idx = (max_weights == weights[0])
        weights0_idx = weights0_idx.float()
        weights[0] = weights[0] * weights0_idx

        weights1_idx = (max_weights == weights[1])
        weights1_idx = weights1_idx.float()
        weights[1] = weights[1] * weights1_idx

        gcam = torch.mul(fmaps[0].cuda(), weights[0].cuda()).sum(dim=1, keepdim=True) + torch.mul(fmaps[1].cuda(), weights[1].cuda()).sum(dim=1, keepdim=True)

        # gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

    def generate_multiview(self, target_layer):
        fmaps_views = self._find(self.fmap_pool, target_layer)
        grads_views = self._find(self.grad_pool, target_layer)

        gcam_views = []
        for fmaps, grads in zip(fmaps_views, grads_views):
            weights = self._compute_grad_weights(grads)

            gcam = torch.mul(fmaps.cuda(), weights.cuda()).sum(dim=1, keepdim=True)
            # gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
            gcam = F.relu(gcam)

            gcam = F.interpolate(
                gcam, self.image_shape, mode="bilinear", align_corners=False
            )

            B, C, H, W = gcam.shape
            gcam = gcam.view(B, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam /= gcam.max(dim=1, keepdim=True)[0]
            gcam = gcam.view(B, C, H, W)

            gcam_views.append(gcam)

        return gcam_views


def occlusion_sensitivity(
        model, images, ids, mean=None, patch=35, stride=1, n_batches=128
):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17
    
    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """

    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i: i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h: grid_h + patch_H, grid_w: grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps
