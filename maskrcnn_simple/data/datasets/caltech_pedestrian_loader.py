# -*- coding: utf-8 -*-
# 数据集转换可参考[caltech-pedestrian-dataset-converter](https://github.com/mitmul/caltech-pedestrian-dataset-converter)
import glob
import json
import random
import time

from PIL import Image
import torch
import os

from scipy.io import loadmat

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

        # print('data_dir:', self.root)
        # print(os.path.join(self.root, 'convert/annotations.json'))
        self.annotations_all = json.load(open(os.path.join(self.root, 'convert/annotations.json')))
        if split=='train':
            self.files = glob.glob(os.path.join(self.root, 'convert/images/set0[0-5]_*.png'))
        elif split=='test':
            self.files = glob.glob(os.path.join(self.root, 'convert/images/set0[6-10]*.png'))
        # self.files.sort()

        # print('annotations_all:', self.annotations_all)
        # print('seqs_all top_n:', self.seqs_all[:5])

    def __getitem__(self, index):
        img_path = self.files[index]
        img_name = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
        # print('img_name:', img_name)
        set_name, video_name, frame_num = img_name.split('_')
        # print('set_name:', set_name)
        # print('video_name:', video_name)
        # print('frame_num:', frame_num)
        img = Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)

        annotations = self.annotations_all[set_name][video_name]['frames'][frame_num]
        # print('annotations:', annotations)

        boxes = []
        labels = []
        difficult_boxes = []
        for datum in annotations:
            # print(datum)
            x, y, w, h = [v for v in datum['pos']]
            boxes.append((x, y, x+w, y+h))
            labels.append(1)
            difficult_boxes.append(0)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        width, height = img.size
        im_info = {'height': height, 'width': width}

        # boxes = self.boxes_all[index]
        # labels = self.labels_all[index]
        # difficult_boxes = self.difficult_boxes_all[index]
        #
        anno = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels),
            'difficult': torch.tensor(difficult_boxes),
            'im_info': im_info,
        }

        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])  # 增加labels
        target.add_field("difficult", anno["difficult"])  # 增加是否difficult

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)  # transform img and target, like ReSize

        return img, target, index

    def __len__(self):
        return len(self.files)

    def get_img_info(self, index):
        im_info = {'height': 480, 'width': 640}
        return im_info

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
    for iteration, (images, targets, _) in enumerate(dataset):
        images_np = np.array(images)
        # print('images_np.shape:', images_np.shape)
        bboxes = targets.bbox
        for bbox in bboxes:
            # print('bbox:', bbox)
            cv2.rectangle(images_np, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])), color=(255, 0, 0))
        # print('images:', images)
        # print('targets:', targets)
        plt.imshow(images_np)
        if iteration == 0:
            break
    plt.show()
