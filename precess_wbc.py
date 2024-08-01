# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 22:47
# @Author  : Haoyi Fan
# @Email   : isfanhy@gmail.com
# @File    : 02_dataset_preprocessing.py
# @Software: PyCharm

# import the necessary packages
import glob
import os
import cv2
import numpy as np


# DT='LISC'
DT='BCISC1K'
# DT='PBC'
# ROOT='/data/wangcs/Data/image/WBC/{}/images'.format(DT)
ROOT='/data/wangcs/Data/image/WBC/{}'.format(DT)
SAVE_PATH='./data'

DICT_TYPE={
    'baso': 0,
    # 'Baso':0,
    'eosi':1,
    'lymp': 2,
    'mono': 3,
    'neut': 4,
    'multi': 5,
    # 'erythroblast': 5,
    # 'ig': 6,
    # 'platelet': 7,
}
DICT_TYPE1={
    'basophil': 0,
    'eosinophil':1,
    'lymphocyte': 2,
    'monocyte': 3,
    'neutrophil': 4,
    'erythroblast': 5,
    'ig': 6,
    'platelet': 7,
}
dict_path = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],

}

if __name__ == '__main__':
    images_path = ROOT

    images_list = glob.glob(images_path + '/*/*.jpg')

    for m, img_path in enumerate(images_list):
        img_name = img_path.split(os.sep)[-1].split('_')[0]
        # img_name = img_path.split(os.sep)[-2]

        img_p = img_path.split('image/')[-1]

        if img_name in DICT_TYPE.keys():
            dict_path[DICT_TYPE[img_name]].append(img_p)


    fw_train=open('{}/{}.txt'.format(SAVE_PATH, DT), 'w')

    for label in dict_path.keys():
        for i, path in enumerate(dict_path[label]):
            fw_train.write("{} {}\n".format(path, label))

            print(path, label)
    fw_train.flush()






















