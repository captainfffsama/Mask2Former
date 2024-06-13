# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2024-06-12 16:55:18
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-06-12 16:55:18
@FilePath: /Mask2Former/own_tools/mask2json.py
@Description:
'''
import os
import json
import glob
import base64
import random
import cv2 as cv
import numpy as np
import labelme2coco as l2c


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(f'-- new folder "{path}" --')
    else:
        print(f'-- the folder "{path}" is already here --')


def find_hulls(img):

    # 检测二进制图像的轮廓
    contours, hierarchy = cv.findContours(img[:, :, 0], cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)

    # 获取所有轮廓的凸包
    hulls_list = []

    for i in range(len(contours)):
        l0 = []
        cnt = contours[i]
        hull = cv.convexHull(cnt)
        for j in range(len(hull)):
            l1 = [float(hull[j][0][0]), float(hull[j][0][1])]
            l0.append(l1)
        hulls_list.append(l0)

    return hulls_list


def mask2json(img, img_path, points):
    shapes = []
    # for i in range(len(points)):
    for point in points:
        if len(point) > 2:  # 防止有的没有形成闭合的轮廓
            shape = {
                'label': '1',
                'points': point,
                'group_id': None,
                'shape_type': "polygon",
                'flags': {}
            }
            shapes.append(shape)

    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]

    data = {
        'version':
        '4.6.0',
        'flags': {},
        'shapes':
        shapes,
        'imagePath':
        '.' + img_path,
        'imageData':
        str(base64.b64encode(open(img_path, "rb").read())).split("b'")[1],
        'imageHeight':
        img_height,
        'imageWidth':
        img_width
    }

    return data


def main():
    img_folder_path = '图片所在目录'
    mask_folder_path = '图片掩码所在目录'
    json_folder_path = 'JOSN文件将要保存的目录'
    mkdir(json_folder_path)
    mkdir('coco/annotations/')
    mkdir('coco/train2017/')
    mkdir('coco/val2017/')

    img_names = os.listdir(img_folder_path)
    for img_name in img_names:

        # Get path
        name = img_name.split('.jpg')[0]
        print(img_name)
        mask_name = name + '.png'
        img_path = img_folder_path + img_name
        mask_path = mask_folder_path + mask_name
        print(f'图片路径：{img_path}')

        # Read img
        img = cv.imread(img_path)
        mask = cv.imread(mask_path)

        # Processing
        hulls = find_hulls(mask)
        data = mask2json(img, img_path, hulls)
        with open(json_folder_path + name + '.json', "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            print(f'已写入 {name}.json')

    # 读取目录
    labelme_json = glob.glob(
        r'json文件所在目录/*.json')
    json_size = len(labelme_json)
    print(f'数据集数量：{json_size}')

    # 划分训练集和验证集
    random.seed(10)
    train_dataset = random.sample(labelme_json, int(float(json_size * 0.6)))
    labelme_json_remain = list(set(labelme_json) - set(train_dataset))  # 取补集
    val_dataset = random.sample(labelme_json_remain,
                                int(float(json_size * 0.2)))
    test_dataset = list(set(labelme_json_remain) - set(val_dataset))
    print(f'训练集数量：{len(train_dataset)}')
    print(f'验证集数量：{len(val_dataset)}')
    print(f'测试集数量：{len(test_dataset)}')

    # 格式转换
    l2c.labelme2coco(train_dataset, './coco/annotations/instances_train2017.json')
    l2c.labelme2coco(val_dataset, './coco/annotations/instances_val2017.json')
    l2c.labelme2coco(test_dataset, './coco/annotations/instances_test2017.json')


if __name__ == "__main__":
    main()