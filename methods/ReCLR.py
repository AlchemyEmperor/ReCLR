#!/usr/bin/env python

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
        # The complete code of ReCLR will be published after the manuscript is accepted. 

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
        # The complete code of ReCLR will be published after the manuscript is accepted. 
        return loss

    def CL_loss_std(self, z1, z2, t=0.5):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,kc->nk', [z1, z2])
        d = z1.shape[0]
        diag_mask = torch.eye(d, device=z1.device, dtype=torch.bool)
        l_neg_1 = l_neg_1[~diag_mask].view(d, -1)
        l_neg = l_neg_1

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

    def get_attack_X(self, train_x=None, ep=None):
        # The complete code of ReCLR will be published after the manuscript is accepted. 
        return train_x.detach(), d_1, d_2

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.net.train()
        self.head.train()
        loss_meter = AverageMeter()
        statistics_dict = {}
        # The complete code of ReCLR will be published after the manuscript is accepted. 


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
