#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of the paper:
# "A Simple Framework for Contrastive Learning of Visual Representations", Chen et al. (2020)
# Paper: https://arxiv.org/abs/2002.05709
# Code (adapted from):
# https://github.com/pietz/simclr
# https://github.com/google-research/simclr

import math
import time
import collections

import kornia.augmentation
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

from models import VanillaVAE
from utils import AverageMeter
from torch.autograd import Variable
from torchvision.models import *


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).cuda()  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

class Model(torch.nn.Module):
    def __init__(self, feature_extractor):
        super(Model, self).__init__()

        self.net = nn.Sequential(collections.OrderedDict([
            ("feature_extractor", feature_extractor)
        ]))

        self.head = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(feature_extractor.feature_size, 256)),
            ("bn1", nn.BatchNorm1d(256)),
            ("relu", nn.LeakyReLU()),
            ("linear2", nn.Linear(256, 64)),
        ]))

        self.optimizer = Adam([{"params": self.net.parameters(), "lr": 0.001},
                               {"params": self.head.parameters(), "lr": 0.001}])

    def return_loss_fn(self, x, t=0.5, eps=1e-8, it=0, d_1=None, d_2=None):
        x = F.normalize(x, dim=1)
        if d_1 is None and d_2 is None:
            s_x = x[2::3]
            d_1 = torch.sum(x[::3] * s_x, dim=1)
            d_2 = torch.sum(x[1::3] * s_x, dim=1)
        pos_sample = torch.cat([x[::3][d_1 >= d_2], x[1::3][d_1 < d_2]], dim=0)
        neg_sample = torch.cat([x[::3][d_1 < d_2], x[1::3][d_1 >= d_2]], dim=0)
        origin_img = torch.cat([x[2::3][d_1 >= d_2], x[2::3][d_1 < d_2]], dim=0)

        # origin_img = nn.functional.normalize(origin_img)
        # x = nn.functional.normalize(x)
        # y = nn.functional.normalize(y)

        # y 是远的
        l_pos = torch.einsum('nc,nc->n', [origin_img, pos_sample]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,kc->nk', [origin_img, pos_sample])
        l_neg_2 = torch.einsum('nc,kc->nk', [origin_img, neg_sample])
        l_neg_3 = torch.einsum('nc,kc->nk', [origin_img, origin_img])
        d = origin_img.shape[0]
        diag_mask = torch.eye(d, device=origin_img.device, dtype=torch.bool)

        l_neg_1 = l_neg_1[~diag_mask].view(d, -1)
        l_neg_3 = l_neg_3[~diag_mask].view(d, -1)

        l_neg = torch.cat([l_neg_1, l_neg_2, l_neg_3], dim=1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= t
        # labels = 1  # undefine
        labels = torch.zeros(logits.size(0), logits.size(1)).cuda()
        labels[:, 0] = 1.0

        loss = -torch.sum(labels.detach() * F.log_softmax(logits, 1), 1).mean()
        return loss


    def show(self, train_x_adv, train_x):

        mean = [0.491, 0.482, 0.447]
        std = [0.247, 0.243, 0.262]
        adv = kornia.augmentation.Denormalize(mean=mean, std=std)(train_x_adv).permute(
            [0, 2, 3, 1]).clamp(0, 1)
        source = kornia.augmentation.Denormalize(mean=mean, std=std)(train_x).permute(
            [0, 2, 3, 1]).clamp(0, 1)
        import matplotlib.pyplot as plt
        plt.imshow(adv[0].detach().cpu().numpy())
        plt.show()
        plt.imshow(source[0].detach().cpu().numpy())
        plt.show()
        plt.imshow((source[0] - adv[0]).detach().cpu().numpy())
        plt.show()
        plt.imshow(adv[1].detach().cpu().numpy())
        plt.show()
        plt.imshow(source[1].detach().cpu().numpy())
        plt.show()
        plt.imshow(adv[2].detach().cpu().numpy())
        plt.show()
        plt.imshow(source[2].detach().cpu().numpy())
        plt.show()

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.net.train()
        self.head.train()
        loss_meter = AverageMeter()
        statistics_dict = {}
        for i, (data_s, data_augmented, target) in enumerate(train_loader):
            data_augmented.append(data_s)
            data = torch.stack(data_augmented, dim=1)
            d = data.size()
            train_x = data.view(d[0] * 3, d[2], d[3], d[4]).cuda()
            # train_x = torch.cat([data_s.cuda(),train_x], dim=0)
            # train_x.requires_grad = True
            self.optimizer.zero_grad()
            features = self.net(train_x.detach())
            embeddings = self.head(features)
            loss_std = self.return_loss_fn(embeddings)

            # features = self.net(train_x_adv.detach())
            # embeddings = self.head(features)
            # loss_adv = self.return_loss_fn(embeddings, it=epoch, d_1=d_11, d_2=d_22)
            # loss_adv = self.CL_loss_std(embeddings)

            loss = loss_std

            tot_pairs = int(features.shape[0] * features.shape[0])
            loss_meter.update(loss.item(), features.shape[0])
            loss.backward()

            self.optimizer.step()
            if (i == 0):
                statistics_dict["batch_size"] = data.shape[0]
                statistics_dict["tot_pairs"] = tot_pairs

        elapsed_time = time.time() - start_time
        print("Epoch [" + str(epoch) + "]"
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + str(loss_meter.avg)
              + "; batch-size: " + str(statistics_dict["batch_size"])
              + "; tot-pairs: " + str(statistics_dict["tot_pairs"]))

        return loss_meter.avg, -loss_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        feature_extractor_state_dict = self.net.feature_extractor.state_dict()
        head_state_dict = self.head.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": feature_extractor_state_dict,
                    "head": head_state_dict,
                    "optimizer": optimizer_state_dict},
                   file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.net.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.head.load_state_dict(checkpoint["head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
