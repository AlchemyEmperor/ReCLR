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

        # checkpoint_path = "./checkpoint/VAE/BCISC1K/3_2/VAE_BCISC1K_conv4_seed_1024_epoch_1000.tar"
        # checkpoint = torch.load(checkpoint_path)
        # self.vae = VanillaVAE(in_channels=3, latent_dim=128)
        # self.vae.load_state_dict(checkpoint["backbone"])
        # for name, value in self.vae.named_parameters():
        #     value.requires_grad = False


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

        with torch.no_grad():
            logits_pd = l_neg
            pseudo_labels = F.softmax(logits_pd, 1)
            log_pseudo_labels = F.log_softmax(logits_pd, 1)
            entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
            c = 1 - entropy / np.log(pseudo_labels.shape[1])
            pseudo_labels = 5 * c * pseudo_labels  # num of neighbors * uncertainty * pseudo_labels
            pseudo_labels = torch.minimum(pseudo_labels,
                                          torch.tensor(1).to(pseudo_labels.device))  # upper thresholded by 1
            labels[:, 1:] = pseudo_labels
            # label normalization
            labels = labels / labels.sum(dim=1, keepdim=True)

        loss = -torch.sum(labels.detach() * F.log_softmax(logits, 1), 1).mean()
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
        with torch.no_grad():
            x = self.head(self.net(train_x))
            x = F.normalize(x, dim=1)
        #
            s_x = x[2::3]
            d_1 = torch.sum(x[::3] * s_x, dim=1)
            d_2 = torch.sum(x[1::3] * s_x, dim=1)
        #     pos_sample_adv = torch.cat([x[::3][d_1 >= d_2], x[1::3][d_1 < d_2]], dim=0)
            neg_sample = torch.cat([x[::3][d_1 < d_2], x[1::3][d_1 >= d_2]], dim=0)
            origin_img = torch.cat([x[2::3][d_1 >= d_2], x[2::3][d_1 < d_2]], dim=0)

        # attack
        t = 5
        # train_x_adv = train_x.clone()
        train_x_pos = torch.cat([train_x[::3][d_1 >= d_2], train_x[1::3][d_1 < d_2]], dim=0)
        train_x_neg = torch.cat([train_x[1::3][d_1 >= d_2], train_x[::3][d_1 < d_2]], dim=0)
        train_x_anchor = torch.cat([train_x[2::3][d_1 >= d_2], train_x[2::3][d_1 < d_2]], dim=0)
        train_x_adv = train_x_pos.clone()
        beta1 = 0.9
        beta2 = 0.999
        eps = 4 / 255.
        eps_iter = 2 / 255.
        momentum = 0
        vol = 0
        prev_loss = 1.
        for i in range(t):
            train_x_adv = Variable(train_x_adv.detach(), requires_grad=True)
            train_x_adv.retain_grad()
            x = self.head(self.net(train_x_adv))
            x = F.normalize(x, dim=1)

            pos_sample_adv = x
            with torch.no_grad():
                # neg_sample = torch.cat([x[::3][d_1 < d_2], x[1::3][d_1 >= d_2]], dim=0)
                # origin_img = torch.cat([x[2::3][d_1 >= d_2], x[2::3][d_1 < d_2]], dim=0)
                # l_pos = torch.einsum('nc,nc->n', [origin_img, pos_sample]).unsqueeze(-1)
                # l_neg_1 = torch.einsum('nc,kc->nk', [origin_img, pos_sample_adv])
                # l_neg_2 = torch.einsum('nc,kc->nk', [origin_img, neg_sample])
                l_neg_3 = torch.einsum('nc,kc->nk', [origin_img, origin_img])
                d = origin_img.shape[0]
                diag_mask = torch.eye(d, device=origin_img.device, dtype=torch.bool)
                # l_neg_1 = l_neg_1[~diag_mask].view(d, -1)
                l_neg_3 = l_neg_3[~diag_mask].view(d, -1)
                # l_neg = torch.cat([l_neg_1, l_neg_2, l_neg_3], dim=1)
                G_neg_min, G_neg_min_index = torch.max(l_neg_3, dim=1, keepdim=True)

            l_pos = torch.einsum('nc,nc->n', [origin_img, pos_sample_adv]).unsqueeze(-1)
            # G_neg_min = torch.min(l_neg, dim=1, keepdim=True)[0]
            # G_neg_max = torch.max(l_neg, dim=1, keepdim=True)[0]
            # alpha = ep / 200
            # G_neg = (1 - alpha) * G_neg_min + alpha * G_neg_max
            # G_neg = alpha * G_neg_min + (1 - alpha) * G_neg_max
            self.optimizer.zero_grad()

            gap = torch.abs(l_pos - G_neg_min.detach())
            index = (gap > 0.5) * (G_neg_min > 0.9)
            if torch.sum(index) == 0 or torch.mean(l_pos) < 0.8:
                break
            else:print('ooo')
            loss_adv = torch.sum(gap * index) / torch.sum(index)

            # loss_adv = torch.mean(l_pos)
            # loss_adv = torch.mean(1-F.cosine_similarity(pos_sample_adv, neg_t.detach()))
            # loss_adv = F.mse_loss(origin_img, pos_sample_adv)
            loss_adv.backward() # retain_graph=True
            # print(loss_adv.item())
            with torch.no_grad():
                # grad_p = torch.sign(train_x_adv.grad)
                # # grad_p = train_x_adv.grad
                # train_x_temp = train_x_adv - 0.001 * grad_p
                # train_x_res = (train_x_temp - train_x_adv)  # .clamp(-(1e-8), 1e-8)
                # train_x_adv += train_x_res

                grad_p = train_x_adv.grad
                momentum = beta1 * momentum + (1 - beta1) * grad_p
                vol = beta2 * vol + (1 - beta2) * grad_p ** 2
                m_ = momentum / (1 - beta1 ** t)
                v_ = vol / (1 - beta2 ** t)
                grad = m_ / (v_ ** 0.5 + 1e-8)
                signed_grad = torch.sign(grad)
                train_x_adv_temp = train_x_adv - eps_iter * signed_grad
                eta = (train_x_adv_temp - train_x_pos).clamp(-eps, eps)
                train_x_adv = (train_x_pos + eta * index.unsqueeze(1).unsqueeze(1)).clamp(-2, 2)
                if torch.abs(loss_adv - prev_loss) / prev_loss < 1e-3:
                    break
                prev_loss = loss_adv


        # show
        # de = kornia.augmentation.Denormalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        # adv_show = de(train_x_adv).permute([0, 2, 3, 1])[0].detach().cpu().numpy()
        # pos_show = de(train_x_pos).permute([0, 2, 3, 1])[0].detach().cpu().numpy()
        # ori_show = de(train_x_anchor).permute([0, 2, 3, 1])[0].detach().cpu().numpy()
        # neg_show = de(train_x_neg).permute([0, 2, 3, 1])[0].detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(adv_show[:,:,::-1])
        # plt.show()
        # plt.imshow(pos_show[:, :, ::-1])
        # plt.show()
        # plt.imshow(ori_show[:, :, ::-1])
        # plt.show()
        # plt.imshow(neg_show[:, :, ::-1])
        # plt.show()

        train_x[::3][d_1 >= d_2] = train_x_adv[:torch.sum(d_1 >= d_2)]
        train_x[1::3][d_1 < d_2] = train_x_adv[torch.sum(d_1 >= d_2):]
        return train_x.detach(), d_1, d_2

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
            train_x_adv, d_11, d_22 = self.get_attack_X(train_x=train_x.clone(), ep=epoch)
            self.optimizer.zero_grad()
            features = self.net(train_x_adv.detach())
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

