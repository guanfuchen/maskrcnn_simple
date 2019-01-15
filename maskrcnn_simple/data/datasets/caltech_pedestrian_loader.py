# -*- coding: utf-8 -*-
# 数据集转换可参考[caltech-pedestrian-dataset-converter](https://github.com/mitmul/caltech-pedestrian-dataset-converter)

import random
import time

from PIL import Image
import torch
import os

from maskrcnn_simple.config.paths_catalog import DatasetCatalog

from maskrcnn_simple.structures.bounding_box import BoxList
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class CaltechPedestrianLoader(data.Dataset):

    CLASSES = (
        "__background__ ",  # 背景为第一类，总共2类
        "pedestrian",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        """
        :param root:
        :param split:
        :param img_size:
        :param transform:
        :param boxcoder:
        """
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self.files = []

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.files)

    def get_img_info(self, index):
        pass

    def map_class_id_to_class_name(self, class_id):
        return CaltechPedestrianLoader.CLASSES[class_id]


if __name__ == '__main__':
    # 数据集加载
    is_train = True
    dataset_name = 'caltech_pedestrian_train'
    data = DatasetCatalog.get(dataset_name)
    args = data["args"]
    args["use_difficult"] = not is_train
    args["transforms"] = None
    dataset = CaltechPedestrianLoader(**args)  # 使用**可以将dict变化为参数列表
    # for iteration, (images, targets, _) in enumerate(dataset):
    #     images_np = np.array(images)
    #     # print('images_np.shape:', images_np.shape)
    #     bboxes = targets.bbox
    #     for bbox in bboxes:
    #         # print('bbox:', bbox)
    #         cv2.rectangle(images_np, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])), color=(255, 0, 0))
    #     # print('images:', images)
    #     # print('targets:', targets)
    #     plt.imshow(images_np)
    #     if iteration == 0:
    #         break
    # plt.show()
