# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_simple.utils.comm import get_world_size
from maskrcnn_simple.utils.imports import import_file

from maskrcnn_simple.data import datasets as D
from maskrcnn_simple.data import samplers

from maskrcnn_simple.data.collate_batch import BatchCollator
from maskrcnn_simple.data.transforms import build_transforms


def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True):
    """
    构建data_list中的数据集
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)  # 从dataset_name中构建数据
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        # if data["factory"] == "COCODataset":
        #     args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train  # if is_train 那么不使用difficult
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)  # 传入data args和transform
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets  # 训练集返回数据集表格

    # for training, concatenate all datasets into a single one
    # print('len(datasets):', len(datasets))
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)
    else:
        dataset = datasets[0]  # 训练中将所有数据集变为一个，也就是concat

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    # print('num_gpus:', num_gpus)
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH  # 每一Batch采用的训练图像
        # print('images_per_batch:', images_per_batch)
        # 一般将gpus分到每一张图像训练
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        # 每一个gpus每一个batch训练图像数，如果images_per_batch=8，num_gpus=2，那么每一个gpus训练4张
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True  # 训练图像时shuffle为True
        num_iters = cfg.SOLVER.MAX_ITER  # 设置最大iterations，默认为40000
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        # 如果每一个gpu训练图像数目超过1
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []  # 是否每一个batch应该包含相同的aspect ratio的图像

    paths_catalog = import_file(
        "maskrcnn_simple.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog  # 倒入paths catalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST  # 如果is_train那么数据集配置为TRAIN的list

    transforms = build_transforms(cfg, is_train)  # 通过cfg构建对数据集图像的transforms
    # print('transforms:', transforms)
    # 将数据集列表，数据集变化transforms，DatasetCatalog和is_train传入构建datasets
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

    data_loaders = []  # for different data loders, like TRAIN: ("voc_2007_train", "voc_2007_val")包含了train和val这两个数据集用来训练
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)  # 对数据集进行采样
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )  # 对每一个batch的数据进行采样
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)  # 所有数据集加载器添加data_loader
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        # 训练期间，仅仅允许一个data_loaders
        assert len(data_loaders) == 1
        return data_loaders[0]  # 返回第一个data_loaders
    return data_loaders
