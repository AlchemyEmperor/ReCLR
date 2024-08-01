#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of a standard neural network (no self-supervised components).
# This is used as baseline (upper bound) and during linear-evaluation and fine-tuning.

import math
import time

from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tqdm
import numpy as np
from utils import AverageMeter


class StandardModel(torch.nn.Module):
    def __init__(self, feature_extractor, num_classes, tot_epochs=200):
        super(StandardModel, self).__init__()
        self.num_classes = num_classes
        self.tot_epochs = tot_epochs
        self.feature_extractor = feature_extractor
        feature_size = feature_extractor.feature_size
        self.classifier = nn.Linear(feature_size, num_classes)
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = SGD([{"params": self.feature_extractor.parameters(), "lr": 0.1, "momentum": 0.9},
                              {"params": self.classifier.parameters(), "lr": 0.1, "momentum": 0.9}])
        self.optimizer_lineval = Adam([{"params": self.classifier.parameters(), "lr": 0.1}])
        self.optimizer_finetune = Adam(
            [{"params": self.feature_extractor.parameters(), "lr": 0.001, "weight_decay": 1e-5},
             {"params": self.classifier.parameters(), "lr": 0.1, "weight_decay": 1e-5}])

    def forward(self, x, detach=False):
        if (detach):
            f = self.feature_extractor(x).detach()
        else:
            f = self.feature_extractor(x)
        out = self.classifier(f)
        return out, f

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.feature_extractor.train()
        self.classifier.train()
        if (epoch == int(self.tot_epochs * 0.5) or epoch == int(self.tot_epochs * 0.75)):
            for i_g, g in enumerate(self.optimizer.param_groups):
                g["lr"] *= 0.1  # divide by 10
                print("Group[" + str(i_g) + "] learning rate: " + str(g["lr"]))
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for i, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available(): data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output, _ = self.forward(data)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = (100.0 * correct / float(len(target)))
            accuracy_meter.update(accuracy.item(), len(target))
        elapsed_time = time.time() - start_time
        print("Epoch [" + str(epoch) + "]"
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + str(loss_meter.avg)
              + "; acc: " + str(accuracy_meter.avg) + "%")
        return loss_meter.avg, accuracy_meter.avg

    def linear_evaluation(self, epoch, train_loader):
        self.feature_extractor.eval()
        self.classifier.train()
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for data, target in minibatch_iter:
            if torch.cuda.is_available(): data, target = data.cuda(), target.cuda()
            self.optimizer_lineval.zero_grad()
            output, _ = self.forward(data, detach=True)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer_lineval.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = (100.0 * correct / float(len(target)))
            accuracy_meter.update(accuracy.item(), len(target))
            minibatch_iter.set_postfix({"loss": loss_meter.avg, "acc": accuracy_meter.avg})
        return loss_meter.avg, accuracy_meter.avg

    def finetune(self, epoch, train_loader):
        self.feature_extractor.train()
        self.classifier.train()

        if (epoch == int(self.tot_epochs * 0.5) or epoch == int(self.tot_epochs * 0.75)):
            for i_g, g in enumerate(self.optimizer_finetune.param_groups):
                g["lr"] *= 0.1  # divide by 10
                print("Group[" + str(i_g) + "] learning rate: " + str(g["lr"]))

        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for data, target in minibatch_iter:
            if torch.cuda.is_available(): data, target = data.cuda(), target.cuda()
            self.optimizer_finetune.zero_grad()
            output, _ = self.forward(data)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer_finetune.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = (100.0 * correct / float(len(target)))
            accuracy_meter.update(accuracy.item(), len(target))
            minibatch_iter.set_postfix({"loss": loss_meter.avg, "acc": accuracy_meter.avg})
        return loss_meter.avg, accuracy_meter.avg

    # def test(self, test_loader):
    #     self.feature_extractor.eval()
    #     self.classifier.eval()
    #     loss_meter = AverageMeter()
    #     accuracy_meter = AverageMeter()
    #
    #     accuracy_cl_list = [AverageMeter() for i in range(8)]
    #
    #     correct_counts = 0
    #     total_counts = 0
    #
    #     with torch.no_grad():
    #         from sklearn.manifold import TSNE
    #         # from openTSNE import TSNE
    #         # from umap import UMAP
    #         # import umap
    #         import matplotlib.pyplot as plt
    #         f_list = []
    #         y = []
    #         for data, target in test_loader:
    #             if torch.cuda.is_available(): data, target = data.cuda(), target.cuda()
    #             output, f = self.forward(data)
    #             f = F.normalize(f, dim=1)
    #             loss = self.ce(output, target)
    #             loss_meter.update(loss.item(), len(target))
    #             pred = output.argmax(-1)
    #             correct = pred.eq(target.view_as(pred)).cpu().sum()
    #             accuracy = (100.0 * correct / float(len(target)))
    #             accuracy_meter.update(accuracy.item(), len(target))
    #
    #             for i in range(8):
    #                 num = torch.sum(target == i)
    #                 if num == 0: continue
    #                 correct = pred[target == i].eq(i).cpu().sum()
    #                 accuracy = (100.0 * correct / float(num))
    #                 accuracy_cl_list[i].update(accuracy.item(), num)
    #
    #             f_list.append(f.detach().cpu().numpy())
    #             y.append(target.detach().cpu().numpy())
    #         # f_list = np.concatenate(f_list)
    #         # y = np.concatenate(y)
    #         # # X_embedded = TSNE(n_components=2, n_jobs=32, verbose=True, initialization='spectral').fit(f_list)
    #         # X_embedded = TSNE(n_components=2, verbose=True).fit_transform(f_list)
    #         # # X_embedded = umap.UMAP(verbose=True, random_state=0,init="random").fit_transform(f_list)
    #         # name = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
    #         # for i in range(8):
    #         #     index = y == i
    #         #     plt.scatter(X_embedded[index, 0], X_embedded[index, 1], color=plt.cm.Set1(i), linewidths=0.1,
    #         #                 marker='o', edgecolors='k', label=name[i])
    #         # # plt.legend(fontsize=12, loc='upper left')
    #         # plt.xticks([])
    #         # plt.yticks([])
    #         # plt.axis('off')
    #         # plt.savefig('demo.svg', bbox_inches='tight', pad_inches=0.0, dpi=300)
    #         # plt.show()
    #         # print()
    #     for i in range(8):
    #         print(accuracy_cl_list[i].avg)
    #     return loss_meter.avg, accuracy_meter.avg

    def test(self, test_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        accuracy_cl_list = [AverageMeter() for i in range(8)]

        correct_counts = 0
        total_counts = 0

        with torch.no_grad():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output, f = self.forward(data)
                f = F.normalize(f, dim=1)
                loss = self.ce(output, target)
                loss_meter.update(loss.item(), len(target))
                pred = output.argmax(-1)
                correct = pred.eq(target.view_as(pred)).cpu().sum()
                accuracy = (100.0 * correct / float(len(target)))
                accuracy_meter.update(accuracy.item(), len(target))

                for i in range(8):
                    num = torch.sum(target == i)
                    if num == 0:
                        continue
                    correct = pred[target == i].eq(i).cpu().sum()
                    accuracy = (100.0 * correct / float(num))
                    accuracy_cl_list[i].update(accuracy.item(), num)

            for i in range(8):
                print("Accuracy for class {}: {:.2f}%".format(i, accuracy_cl_list[i].avg))

        return loss_meter.avg, accuracy_meter.avg

    def return_embeddings(self, data_loader, portion=0.5):
        self.feature_extractor.eval()
        embeddings_list = []
        target_list = []
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if torch.cuda.is_available(): data, target = data.cuda(), target.cuda()
                features = self.feature_extractor(data)
                embeddings_list.append(features)
                target_list.append(target)
                if (i >= int(len(data_loader) * portion)): break
        return torch.cat(embeddings_list, dim=0).cpu().detach().numpy(), torch.cat(target_list,
                                                                                   dim=0).cpu().detach().numpy()

    def save(self, file_path="./checkpoint.dat"):
        state_dict = self.classifier.state_dict()
        feature_extractor_state_dict = self.feature_extractor.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        optimizer_lineval_state_dict = self.optimizer_lineval.state_dict()
        optimizer_finetune_state_dict = self.optimizer_finetune.state_dict()
        torch.save({"classifier": state_dict,
                    "backbone": feature_extractor_state_dict,
                    "optimizer": optimizer_state_dict,
                    "optimizer_lineval": optimizer_lineval_state_dict,
                    "optimizer_finetune": optimizer_finetune_state_dict},
                   file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimizer_lineval.load_state_dict(checkpoint["optimizer_lineval"])
        self.optimizer_finetune.load_state_dict(checkpoint["optimizer_finetune"])
