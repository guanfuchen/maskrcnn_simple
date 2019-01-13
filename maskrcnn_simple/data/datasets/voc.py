# coding=utf-8
import os

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import sys
import matplotlib.pyplot as plt

# from maskrcnn_simple.data.build import make_batch_data_sampler, make_data_sampler
from maskrcnn_simple.config.paths_catalog import DatasetCatalog

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_simple.structures.bounding_box import BoxList


class PascalVOCDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",  # 背景为第一类，总共21类
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir  # 数据集data_dir
        self.image_set = split  # 数据集split，分为train，test，val
        self.keep_difficult = use_difficult  # 训练过程中不使用difficult数据
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        # 在ImageSets/Main存在train.txt，test.txt还有val.txt
        # print(self._imgsetpath % self.image_set)
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]  # 将ids中的\n去除然后加入为ids
        # print('ids:', self.ids)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}  # 将对应的图像转换为id，也就是0:000012，1:000017等等
        # print('id_to_img_map:', self.id_to_img_map)

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))  # class to indices, __background__:0, aeroplane:1
        # print('class_to_ind:', self.class_to_ind)

    def __getitem__(self, index):
        img_id = self.ids[index]  # 获取对应index的图像id
        img = Image.open(self._imgpath % img_id).convert("RGB")  # 读取并转换为RGB

        target = self.get_groundtruth(index)  # 获取groudtruth BoxList
        target = target.clip_to_image(remove_empty=True)
        # print('target:', target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)  # transform img and target, like ReSize

        # print('img.shape:', img.shape)
        return img, target, index

    def __len__(self):
        return len(self.ids)  # 所有图像id

    def get_groundtruth(self, index):
        # 获取图像目标检测标注结果
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()  # 将img_id带入%s获取对应的图像，标注和imageset
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]  # 标注中图像的宽高信息，boxes信息，并转换为BoxList格式
        print('anno["boxes"]:', anno["boxes"])
        print('height:', height)
        print('width:', width)
        print('anno["labels"]:', anno["labels"])
        print('anno["difficult"]:', anno["difficult"])
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])  # 增加labels
        target.add_field("difficult", anno["difficult"])  # 增加是否difficult
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]


def main():
    # 数据集加载
    is_train = True
    dataset_name = 'voc_2007_train'
    data = DatasetCatalog.get(dataset_name)
    args = data["args"]
    args["use_difficult"] = not is_train
    args["transforms"] = None
    dataset = PascalVOCDataset(**args)  # 使用**可以将dict变化为参数列表
    for iteration, (images, targets, _) in enumerate(dataset):
        images_np = np.array(images)
        # print('images_np.shape:', images_np.shape)
        bboxes = targets.bbox
        for bbox in bboxes:
            # print('bbox:', bbox)
            cv2.rectangle(images_np, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])), color=(255, 0, 0))
        # print('images:', images)
        # print('targets:', targets)
        # plt.imshow(images_np)
        if iteration == 0:
            break
    # plt.show()


if __name__ == '__main__':
    main()
