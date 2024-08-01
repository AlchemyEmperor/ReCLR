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


if __name__ == '__main__':
    images_path = './data/wangcs/Data/image/WBC/Raabin-WBC'
    images_list = glob.glob(images_path + '/*/*.jpg')
    # result_path = os.path.sep.join(['.', 'PBC'])
    result_path=images_path

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for m, img_path in enumerate(images_list):
        img_name = img_path.split(os.sep)[-1].split('.jpg')[0]
        img_dir = os.sep.join([result_path, img_path.split(os.sep)[-2]])

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        rs_name = img_name+'.jpg'

        print('[INFO] path: {} ### {}'.format(img_path, os.sep.join([img_dir,
                                 rs_name])))

        img = cv2.imread(img_path)
        (img_h, img_w) = img.shape[:2]

        img_rs = cv2.resize(img, dsize=(256, 256))
        cv2.imwrite(os.sep.join([img_dir,
                                 rs_name]), img_rs)




