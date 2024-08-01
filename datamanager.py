#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning

# Data manager that returns transformations and samplers for each method/dataset.
# If a new method is included it should be added to the "DataManager" class.
# The dataset classes with prefix "Multi" are overriding the original dataset class
# to allow multi-sampling of more images in parallel (required by our method).

import os
import os.path as osp

import sys
import random
from sklearn.model_selection import train_test_split
import pickle
from torch.utils import data

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
from transforms import *
from scipy import misc


class MultiSTL10(dset.STL10):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pic = Image.fromarray(np.transpose(img, (1, 2, 0)))

        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
                img_transformed = self.transform(pic.copy())
                img_list.append(img_transformed)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class MultiCIFAR10(dset.CIFAR10):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        pic = Image.fromarray(img)

        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
                img_transformed = self.transform(pic.copy())
                img_list.append(img_transformed)
        else:
            img_list = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        img_s = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        ])(pic)

        return img_s, img_list, target


class MultiCIFAR100(dset.CIFAR100):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        pic = Image.fromarray(img)

        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
                img_transformed = self.transform(pic.copy())
                img_list.append(img_transformed)
        else:
            img_list = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class SuperCIFAR100(dset.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train,
                         transform, target_transform,
                         download)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, "rb")
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding="latin1")
                self.train_data.append(entry["data"])
                self.train_labels += entry["coarse_labels"]
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, "rb")
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding="latin1")
            self.test_data = entry["data"]
            self.test_labels = entry["coarse_labels"]
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def train_val_test_split(root, list_path, seed):
    img_ids = [i_id.strip() for i_id in open(list_path)]
    x_all = []
    y_all = []
    for item in img_ids:
        img_path = item.split(' ')[0]
        label = int(item.split(' ')[1])
        img_file = osp.join(root, "%s" % img_path)
        x_all.append(img_file)
        y_all.append(label)

    # calculate the number of each class
    label_idxs = np.unique(y_all)
    class_stat_all = {}
    for idx in label_idxs:
        class_stat_all[idx] = len(np.where(y_all == idx)[0])

    # print("[Stat] All class: {}".format(class_stat_all))

    x_train_, x_test, y_train_, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=seed)  # 0.6
    x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, test_size=0.2, random_state=seed)

    label_idxs = np.unique(y_train)
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(y_train == idx)[0])
    # print("[Stat] Train class: {}".format(class_stat))
    # print("[Stat] Train class: mean={}, std={}".format(np.mean(list(class_stat.values())),
    #                                                    np.std(list(class_stat.values()))))

    label_idxs = np.unique(y_val)
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(y_val == idx)[0])
    # print("[Stat] Test class: {}".format(class_stat))
    # print("[Stat] Val class: mean={}, std={}".format(np.mean(list(class_stat.values())),
    #                                                  np.std(list(class_stat.values()))))

    label_idxs = np.unique(y_test)
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(y_test == idx)[0])
    # print("[Stat] Test class: {}".format(class_stat))
    # print("[Stat] Test class: mean={}, std={}".format(np.mean(list(class_stat.values())),
    #                                                   np.std(list(class_stat.values()))))

    # print(x_train[10:12], x_val[10:12], x_test[10:12])
    return x_train, y_train, x_val, y_val, x_test, y_test


class LeukocyteDataSet(data.Dataset):

    def __init__(self, repeat_augmentations, img_ids, y,
                 transform=None, target_transform=None):

        if repeat_augmentations:
            self.repeat_augmentations = repeat_augmentations
        else:
            self.repeat_augmentations = 1
        self.transform = transform
        self.target_transform = target_transform
        self.files = []

        for img_file, label in zip(img_ids, y):
            self.files.append({
                "img": img_file,
                "label": int(label),
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # img, target = self.data[index], self.targets[index]
        datafiles = self.files[index]
        img = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        target = datafiles["label"]

        pic = Image.fromarray(img)
        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
                img_transformed = self.transform(pic.copy())
                img_list.append(img_transformed)
        else:
            img_list = None

        if self.target_transform is not None:
            target = self.target_transform(target)
        img_s = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])(pic)
        return img_s, img_list, target


class LeukocyteDataSet_Single(data.Dataset):

    def __init__(self, img_ids, y,
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.files = []
        for img_file, label in zip(img_ids, y):
            # if label != 7:
            #     continue
            self.files.append({
                "img": img_file,
                "label": int(label),
            })
        print()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # img, target = self.data[index], self.targets[index]
        datafiles = self.files[index]
        img = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        target = datafiles["label"]
        # print(datafiles["img"], img.shape)

        pic = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(pic.copy())

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class LeukocyteDataSet_Trans(data.Dataset):

    def __init__(self, img_ids, y,
                 transforms=None,
                 totensor_transform=None,
                 rnd_crop=None
                 ):

        self.transforms = transforms
        self.totensor_transform = totensor_transform
        self.rnd_crop = rnd_crop
        self.files = []
        for img_file, label in zip(img_ids, y):
            self.files.append({
                "img": img_file,
                "label": int(label),
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # img, target = self.data[index], self.targets[index]
        datafiles = self.files[index]
        img = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        target = datafiles["label"]
        # print(datafiles["img"], img.shape)

        img = Image.fromarray(img)
        img_list0 = list()
        label_list = list()
        target_list = list()

        img_name = datafiles["img"].split(os.sep)[-1].split('.jpg')[0]

        # if not os.path.exists('images/augments'):
        #     os.makedirs('images/augments')

        for label, transform in enumerate(self.transforms):
            img_1 = transform(img.copy())
            # print("img_1.__class__", img_1.__class__)
            # img_t = self.rnd_crop(Image.fromarray(img_1).copy())
            # print("img_t.__class__", img_t.__class__)
            img_transformed = img_1
            img_list0.append(self.totensor_transform(img_transformed.copy()))
            label_list.append(label)
            target_list.append(target)
            # cv2.imwrite('images/augments/{}_{}.jpg'.format(img_name, label),
            #             np.array(img_transformed))
            # print('images/augments/{}_{}.jpg'.format(img_name, label))

        return img_list0, label_list, target_list


class LeukocyteDataSet_Trans_1(data.Dataset):

    def __init__(self, img_ids, y,
                 transforms=None,
                 totensor_transform=None,
                 rnd_crop=None
                 ):

        self.transforms = transforms
        self.totensor_transform = totensor_transform
        self.rnd_crop = rnd_crop
        self.files = []
        self.repeat_augmentations = 2
        for img_file, label in zip(img_ids, y):
            self.files.append({
                "img": img_file,
                "label": int(label),
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # img, target = self.data[index], self.targets[index]
        datafiles = self.files[index]
        img = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        target = datafiles["label"]
        # print(datafiles["img"], img.shape)

        img = Image.fromarray(img)
        orgin_img = list()
        img_list = list()
        target_list = list()

        img_name = datafiles["img"].split(os.sep)[-1].split('.jpg')[0]

        # if not os.path.exists('images/augments'):
        #     os.makedirs('images/augments')
        if isinstance(self.transforms.transforms, list):
            img_transformed_0 = self.transforms.transforms[0](img.copy())
            img_list.append(img_transformed_0)
            # for _ in range(self.repeat_augmentations):
            #     img_transformed = self.transforms.transform.transforms[1](pic.copy())
            #     img_list.append(img_transformed)
            for _ in range(self.repeat_augmentations):
                img_transformed = self.transforms.transforms[1](img.copy())
                img_list.append(img_transformed)
            # img_transformed_1 = self.transforms.transform.transforms[1](pic.copy())
            # img_list.append(img_transformed_1)
            # img_transformed_2 = self.transforms.transform.transforms[2](pic.copy())
            # img_list.append(img_transformed_2)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return orgin_img, img_list, target_list


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


class LeukocyteDataSet_jigsaw(data.Dataset):

    def __init__(self, data_path, txt_list, classes=1000):

        self.data_path = data_path
        self.names, _ = self.__dataset_info(txt_list)
        self.N = len(self.names)
        self.permutations = self.__retrive_permutations(classes)

        self.__image_transformer = transforms.Compose([
            transforms.Resize(224, Image.BILINEAR),
            transforms.CenterCrop(224)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]

        img = Image.open(framename).convert('RGB')
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrive_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


class TinyImageFolder(dset.ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)


class MultiTinyImageFolder(dset.ImageFolder):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations

    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        pic = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(pic, np.uint8, copy=False))

        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
                img_transformed = self.transform(pic.copy())
                img_list.append(img_transformed)
        else:
            img_list = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class SlimImageFolder(dset.ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)


class MultiSlimImageFolder(dset.ImageFolder):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations

    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        pic = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(pic, np.uint8, copy=False))

        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
                img_transformed = self.transform(pic.copy())
                img_list.append(img_transformed)
        else:
            img_list = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class DataManager():
    def __init__(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _check(self, dataset):
        datasets_list = ["cifar10", "stl10", "cifar100",
                         "supercifar100", "tiny", "slim"]
        if (dataset not in datasets_list):
            raise Exception("[ERROR] The dataset " + str(dataset) + " is not supported!")
        if (dataset == "slim"):
            if (os.path.isdir("./data/SlimageNet64/train") == False):
                raise Exception(
                    "[ERROR] The train data of SlimageNet64 has not been found in ./data/SlimageNet64/train \n"
                    + "1) Download the dataset from: https://zenodo.org/record/3672132 \n"
                    + "2) Uncompress the dataset in ./data/SlimageNet64  \n"
                    + "3) Place training images in /train and test images in /test")
        elif (dataset == "tiny"):
            if (os.path.isdir("./data/tiny-imagenet-200/train") == False):
                raise Exception(
                    "[ERROR] The train data of TinyImagenet has not been found in ./data/tiny-imagenet-200/train \n"
                    + "1) Download the dataset \n"
                    + "2) Uncompress the dataset in ./data/tiny-imagenet-200  \n"
                    + "3) Place training images in /train and test images in /test")

    def get_num_classes(self, dataset):
        # self._check(dataset)
        if (dataset == "cifar10"):
            return 10
        elif (dataset == "stl10"):
            return 10
        elif (dataset == "supercifar100"):
            return 20
        elif (dataset == "cifar100"):
            return 100
        elif (dataset == "tiny"):
            return 200
        elif (dataset == "slim"):
            return 1000
        elif (dataset == "BCISC1K"):
            return 6
        elif (dataset == "PBC"):
            return 8
        elif (dataset == "Raabin-WBC"):
            return 5

    def get_train_transforms(self, method, dataset):
        """Returns the training torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
          dataset: The name of the dataset.
        Returns:
          train_transform: An object of type torchvision.transforms.
        """
        # self._check(dataset)
        if (dataset == "cifar10"):
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            side = 32;
            padding = 4;
            cutout = 0.25
        elif (dataset == "stl10"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            side = 96;
            padding = 12;
            cutout = 0.111
        elif (dataset == "cifar100" or dataset == "supercifar100"):
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            side = 32;
            padding = 4;
            cutout = 0.0625
        elif (dataset == "tiny"):
            # Image-Net --> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            side = 64;
            padding = 8
        elif (dataset == "slim"):
            # Image-Net --> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            side = 64;
            padding = 8
        elif (dataset in ['BCISC1K', 'PBC','Raabin-WBC']):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            side = 224;
            padding = 0

        if (method == "relationnet" or method == "simclr" or method == 'HeCLR' or method == 'ReCLR' or method == 'DirectCLR' or method == 'VicReg'):
            color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.08, 1.0),
                                                           ratio=(0.75, 1.3333333333333333), interpolation=2)
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            # rnd_rot = transforms.RandomRotation(10., resample=2),
            train_transform = transforms.Compose([rnd_resizedcrop,
                                                  rnd_hflip,
                                                  rnd_color_jitter,
                                                  rnd_gray,
                                                  transforms.ToTensor(), normalize])
        elif (method == "mixup"):
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif method == 'VAE':
            train_transform = transforms.Compose([transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])])

        elif (method == 'transformation' or method == 'retransformation'):
            rnd_crop = transforms.RandomCrop(224)
            raw = Raw()
            color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            gray = transforms.Grayscale(num_output_channels=3)
            blend_alpha = BlendAlpha()
            channel_shuffle = ChannelShuffle()
            gamma_contrast = GammaContrast()
            elastic_trans = ElasticTransformation()
            average_pool = AveragePooling()
            super_pixels = Superpixels()
            AdditiveGaussianNoises = AdditiveGaussianNoise()
            BilateralBlurs = BilateralBlur()
            AverageBlurs = AverageBlur()
            GuassianBlurs = GuassianBlur()
            MedianBlurs = MedianBlur()
            Embosss = Emboss()
            CoarseDropouts = CoarseDropout()
            FilterEdgeEnhances = FilterEdgeEnhance()
            Affines = Affine()
            augments = list()

            transA = transforms.Compose(
                [raw, transforms.RandomCrop(224)]
            )
            transB = transforms.Compose(
                [color_jitter, transforms.RandomCrop(224)]
            )
            transC = transforms.Compose(
                [gray, transforms.RandomCrop(224)]
            )
            transD = transforms.Compose(
                [blend_alpha, transforms.RandomCrop(224)]
            )
            transE = transforms.Compose(
                [gamma_contrast, transforms.RandomCrop(224)]
            )
            transF = transforms.Compose(
                [elastic_trans, transforms.RandomCrop(224)]
            )
            transG = transforms.Compose(
                [average_pool, transforms.RandomCrop(224)]
            )
            transH = transforms.Compose(
                [super_pixels, transforms.RandomCrop(224)]
            )
            transI = transforms.Compose(
                [BilateralBlurs, transforms.RandomCrop(224)]
            )
            transJ = transforms.Compose(
                [Embosss, transforms.RandomCrop(224)]
            )
            transK = transforms.Compose(
                [AverageBlurs, transforms.RandomCrop(224)]
            )
            transL = transforms.Compose(
                [GuassianBlurs, transforms.RandomCrop(224)]
            )
            transM = transforms.Compose(
                [MedianBlurs, transforms.RandomCrop(224)]
            )
            transN = transforms.Compose(
                [CoarseDropouts, transforms.RandomCrop(224)]
            )
            transO = transforms.Compose(
                [Affine, transforms.RandomCrop(224)]
            )  # 有问题
            transP = transforms.Compose(
                [FilterEdgeEnhances, transforms.RandomCrop(224)]
            )
            # augments = [transA, transB, transC, transD, transE, transF,transH,transI,transK,transM,transI,transP]
            augments = [transB, transC]
            #                         transH, transI, transM, transK, transN, transP]

            return augments, transforms.ToTensor(), rnd_crop


        elif (method == "deepinfomax"):
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        elif (method == 'constvs'):
            weak = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.08, 1.0),
                                                           ratio=(0.75, 1.3333333333333333), interpolation=2)
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            # rnd_rot = transforms.RandomRotation(10., resample=2),
            strong_1 = transforms.Compose([rnd_resizedcrop, rnd_hflip,
                                           rnd_color_jitter, rnd_gray, transforms.ToTensor(), normalize])
            strong_2 = transforms.Compose([rnd_resizedcrop, rnd_hflip,
                                           rnd_color_jitter, rnd_gray, transforms.ToTensor(), normalize])
            train_transform = transforms.Compose([weak, strong_1])

        elif (method == "standard" or method == "rotationnet" or method == "deepcluster"):
            train_transform = transforms.Compose([  # transforms.RandomCrop(side, padding=padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize])

        elif (method == "jigsaw"):
            train_transform = transforms.Compose([  # transforms.RandomCrop(side, padding=padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize])

        elif (method == "finetune"):
            rnd_affine = transforms.RandomApply([transforms.RandomAffine(18, scale=(0.9, 1.1),
                                                                         translate=(0.1, 0.1), shear=10,
                                                                         resample=Image.BILINEAR, fillcolor=0)], p=0.5)
            train_transform = transforms.Compose([  # transforms.RandomCrop(side, padding=padding),
                transforms.RandomHorizontalFlip(),
                rnd_affine,
                transforms.ToTensor(), normalize,
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))]) #pytorch default
                transforms.RandomErasing(p=0.5, scale=(cutout, cutout), ratio=(1.0, 1.0))])
        elif (method == "lineval"):
            train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            raise Exception("[ERROR] The method " + str(method) + " is not supported!")

        return train_transform

    def get_train_loader(self, dataset, data_type, data_size,
                         train_transform,
                         repeat_augmentations, num_workers=8, drop_last=False,
                         augments=None,
                         totensor_transform=None,
                         rnd_crop=None
                         ):
        """Returns the training loader for each dataset/method.
           If a new method or dataset is added, this method should by modified
           accordingly.
        Args:
          method: The name of the method.
          dataset: The name of the dataset.
          data_type: The type of data multi (multiple images in parallel),
            single (one image at the time), unsupervised (used in STL10 to load
            the unlabeled data split).
          data_size: the mini-batch size.
          train_transform: the transformations used by the sampler, they
            should be returned by the method get_train_transforms().
          repeat_augmentations: repeat the augmentations on the same image
            for the specified number of times (needed by RelationNet and SimCLR).
          num_workers: the total number of parallel workers for the samples.
          drop_last: it drops the last sample if the mini-batch cannot be
             aggregated, necessary for methods like DeepInfomax.
        Returns:
          train_loader: The loader that can be used a training time.
          train_set: The train set (used in DeepCluster)
        """
        # self._check(dataset)
        from torch.utils.data.dataset import Subset
        if (data_type == "multi"):
            # Used for: Relational reasoning, SimCLR
            if (dataset == "cifar10"):
                train_set = MultiCIFAR10(repeat_augmentations, root="data", train=True, transform=train_transform,
                                         download=True)
            elif (dataset == "stl10"):
                train_set = MultiSTL10(repeat_augmentations, root="data", split="unlabeled", transform=train_transform,
                                       download=True)
            elif (dataset == "cifar100"):
                train_set = MultiCIFAR100(repeat_augmentations, root="data", train=True, transform=train_transform,
                                          download=True)
            elif (dataset == "tiny"):
                train_set = MultiTinyImageFolder(repeat_augmentations, root="./data/tiny-imagenet-200/train",
                                                 transform=train_transform)
            elif (dataset == "slim"):
                train_set = MultiSlimImageFolder(repeat_augmentations, root="./data/SlimageNet64/train",
                                                 transform=train_transform)
            elif (dataset in ['BCISC1K', 'PBC', 'Raabin-WBC']):
                x_train, y_train, x_val, y_val, x_test, y_test = \
                    train_val_test_split(root=".\data\wangcs\Data\image",  # E:\selfSL_Whiteblood\leukocyte\graduate\data\wangcs\Data\image; /data/yao/wjw/WBC/data/wangcs/Data/image
                                         list_path='./data/{}.txt'.format(dataset), seed=self.seed)
                train_set = LeukocyteDataSet(repeat_augmentations,
                                             img_ids=x_train, y=y_train,
                                             transform=train_transform)

        elif (data_type == "single"):
            # Used for: deepinfomax, rotationnet, standard, lineval, finetune, deepcluster
            if (dataset == "cifar10"):
                train_set = dset.CIFAR10("data", train=True, transform=train_transform, download=True)
            elif (dataset == "stl10"):
                train_set = dset.STL10(root="data", split="train", transform=train_transform, download=True)
            elif (dataset == "supercifar100"):
                train_set = SuperCIFAR100("data", train=True, transform=train_transform, download=True)
            elif (dataset == "cifar100"):
                train_set = dset.CIFAR100("data", train=True, transform=train_transform, download=True)
            elif (dataset == "tiny"):
                train_set = TinyImageFolder(root="./data/tiny-imagenet-200/train", transform=train_transform)
            elif (dataset == "slim"):
                train_set = SlimImageFolder(root="./data/SlimageNet64/train", transform=train_transform)
            elif (dataset in ['BCISC1K', 'PBC', 'Raabin-WBC']):
                x_train, y_train, x_val, y_val, x_test, y_test = \
                    train_val_test_split(root="./data/wangcs/Data/image",
                                         list_path='./data/{}.txt'.format(dataset), seed=self.seed)
                train_set = LeukocyteDataSet_Single(img_ids=x_train, y=y_train,
                                                    transform=train_transform)
        elif (data_type == "trans"):
            # Used for: deepinfomax, rotationnet, standard, lineval, finetune, deepcluster
            if (dataset in ['BCISC1K', 'PBC', 'Raabin-WBC']):
                x_train, y_train, x_val, y_val, x_test, y_test = \
                    train_val_test_split(root=".././data/wangcs/Data/image",
                                         list_path='.././data/{}.txt'.format(dataset), seed=self.seed)
                train_set = LeukocyteDataSet_Trans(img_ids=x_train, y=y_train,
                                                   transforms=augments,
                                                   totensor_transform=totensor_transform,
                                                   rnd_crop=rnd_crop)
        elif (data_type == "const"):
            # Used for: deepinfomax, rotationnet, standard, lineval, finetune, deepcluster
            if (dataset in ['BCISC1K', 'PBC', 'Raabin-WBC']):
                x_train, y_train, x_val, y_val, x_test, y_test = \
                    train_val_test_split(root=".././data/wangcs/Data/image",
                                         list_path='.././data/{}.txt'.format(dataset), seed=self.seed)
                train_set = LeukocyteDataSet_Trans_1(img_ids=x_train, y=y_train,
                                                     transforms=train_transform,
                                                     totensor_transform=totensor_transform,
                                                     rnd_crop=rnd_crop)

        elif (data_type == "unsupervised"):
            if (dataset == "stl10"):
                train_set = dset.STL10(root="data", split="unlabeled", transform=train_transform, download=True)
        else:
            raise Exception("[ERROR] The type " + str(data_type) + " is not supported!")

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=data_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        return train_loader, train_set

    def get_val_loader(self, dataset, data_size, num_workers=8):
        # self._check(dataset)
        if (dataset in ['BCISC1K', 'PBC', 'Raabin-WBC']):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            x_train, y_train, x_val, y_val, x_test, y_test = \
                train_val_test_split(root="./data/wangcs/Data/image",
                                     list_path='./data/{}.txt'.format(dataset), seed=self.seed)
            val_set = LeukocyteDataSet_Single(img_ids=x_val, y=y_val,
                                              transform=test_transform)

        val_loader = torch.utils.data.DataLoader(val_set, batch_size=data_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=True)
        return val_loader

    def get_test_loader(self, dataset, data_size, num_workers=8):
        # self._check(dataset)
        if (dataset == "cifar10"):
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.CIFAR10("data", train=False, transform=test_transform, download=True)
        elif (dataset == "stl10"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.STL10("data", split="test", transform=test_transform, download=True)
        elif (dataset == "supercifar100"):
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = SuperCIFAR100("data", train=False, transform=test_transform, download=True)
        elif (dataset == "cifar100"):
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.CIFAR100("data", train=False, transform=test_transform, download=True)
        elif (dataset == "tiny"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = TinyImageFolder(root=".././data/tiny-imagenet-200/val", transform=test_transform)
        elif (dataset == "slim"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = SlimImageFolder(root=".././data/SlimageNet64/test", transform=test_transform)
        elif (dataset in ['BCISC1K', 'PBC', 'Raabin-WBC']):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            x_train, y_train, x_val, y_val, x_test, y_test = \
                train_val_test_split(root="./data/wangcs/Data/image",
                                     list_path='./data/{}.txt'.format(dataset), seed=self.seed)
            test_set = LeukocyteDataSet_Single(img_ids=x_test, y=y_test,
                                               transform=test_transform)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=data_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=True)
        return test_loader
