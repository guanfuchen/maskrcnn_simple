# -*- coding: utf-8 -*-
import random

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


class WiderFaceLoader(data.Dataset):

    CLASSES = (
        "__background__ ",  # 背景为第一类，总共2类
        "face",
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

        self.boxes_all = []
        self.difficult_boxes_all = []
        self.labels_all = []
        self.files = []
        # self.targets = []
        self.bbox_fname = None

        cls = WiderFaceLoader.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))  # class to indices, __background__:0, face:1

        if self.image_set == 'train':
            self.bbox_fname = os.path.join(self.root, 'wider_face_split/wider_face_train_bbx_gt.txt')
        elif self.image_set == 'test':
            self.bbox_fname = os.path.join(self.root, 'wider_face_split/wider_face_test_filelist.txt')

        # print('bbox_fname:', self.bbox_fname)
        with open(self.bbox_fname) as bbox_fp:
            bbox_lines = bbox_fp.readlines()

        # print(bbox_lines)
        bbox_lines_num = len(bbox_lines)
        for bbox_lines_id in range(bbox_lines_num):
            boxes = []
            labels = []
            difficult_boxes = []
            bbox_line = bbox_lines[bbox_lines_id].strip()

            if 'jpg' not in bbox_line:
                continue

            image_name = bbox_line
            if self.image_set == 'train':
                image_name = os.path.join(self.root, 'WIDER_train/images', image_name)
            elif self.image_set == 'test':
                image_name = os.path.join(self.root, 'WIDER_test/images', image_name)

            if not os.path.isfile(image_name):
                # 不存在文件
                continue
            else:
                pass
                # # 造成加载文件较慢
                # img = cv2.imread(image_name)
                # if img is None:
                #     continue

            face_num = int(bbox_lines[bbox_lines_id+1].strip())  # 图片中有多少人脸

            for face_id in range(face_num):
                bbox_line = bbox_lines[bbox_lines_id+2+face_id].strip()
                bbox_line_split = bbox_line.split(' ')
                x = float(bbox_line_split[0])
                y = float(bbox_line_split[1])
                w = float(bbox_line_split[2])
                h = float(bbox_line_split[3])
                boxes.append([x, y, x+w, y+h])
                labels.append(1)
                difficult_boxes.append(0)
            self.boxes_all.append(boxes)
            self.labels_all.append(labels)
            self.difficult_boxes_all.append(difficult_boxes)

            self.files.append(image_name)
        # print('self.boxes:', self.boxes)
        # print('len(self.boxes):', len(self.boxes))

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')

        width, height = img.size
        im_info = {'height': height, 'width': width}

        boxes = self.boxes_all[index]
        labels = self.labels_all[index]
        difficult_boxes = self.difficult_boxes_all[index]

        anno = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels),
            'difficult': torch.tensor(difficult_boxes),
            'im_info': im_info,
        }

        # height, width = anno["im_info"]  # 标注中图像的宽高信息，boxes信息，并转换为BoxList格式
        # print('anno["boxes"]:', anno["boxes"])
        # print('height:', height)
        # print('width:', width)
        # print('anno["labels"]:', anno["labels"])
        # print('anno["difficult"]:', anno["difficult"])
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
        # img_path = self.files[index]
        # img = Image.open(img_path).convert('RGB')
        # !!!获取图像真实宽高!!!
        width, height = (1024, 1024)
        return {"height": height, "width": width}

    def map_class_id_to_class_name(self, class_id):
        return WiderFaceLoader.CLASSES[class_id]


if __name__ == '__main__':
    # 数据集加载
    is_train = True
    dataset_name = 'wider_train'
    data = DatasetCatalog.get(dataset_name)
    args = data["args"]
    args["use_difficult"] = not is_train
    args["transforms"] = None
    dataset = WiderFaceLoader(**args)  # 使用**可以将dict变化为参数列表
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
