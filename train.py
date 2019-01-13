# -*- coding: utf-8 -*-
import argparse

import torch
from maskrcnn_simple.config.paths_catalog import ModelCatalog
from maskrcnn_simple.utils.c2_model_loading import load_c2_format

from maskrcnn_simple.utils.imports import import_file

from maskrcnn_simple.utils.comm import get_rank

from maskrcnn_simple.data.build import make_data_loader
from maskrcnn_simple.engine.trainer import do_train
from maskrcnn_simple.modeling.detector.detector import build_detection_model
from maskrcnn_simple.config import cfg
from maskrcnn_simple.solver import make_optimizer, make_lr_scheduler
from maskrcnn_simple.utils.logger import setup_logger
from maskrcnn_simple.utils.miscellaneous import mkdir
from maskrcnn_simple.utils.model_serialization import load_state_dict
from maskrcnn_simple.utils.model_zoo import cache_url


def load_file(f, cfg):
    # catalog lookup
    if f.startswith("catalog://"):
        catalog_f = ModelCatalog.get(f[len("catalog://"):])
        f = catalog_f
    # download url files
    if f.startswith("http"):
        # if the file is a url path, download it and cache it
        cached_f = cache_url(f)
        f = cached_f
    # convert Caffe2 checkpoint from pkl
    if f.endswith(".pkl"):
        return load_c2_format(cfg, f)
    # load native detectron.pytorch checkpoint
    loaded = torch.load(f, map_location=torch.device("cpu"))
    if "model" not in loaded:
        loaded = dict(model=loaded)
    return loaded


def train(cfg):
    model = build_detection_model(cfg)

    if cfg.MODEL.WEIGHT != '':
        checkpoint = load_file(cfg.MODEL.WEIGHT, cfg)  # load model weight
        load_state_dict(model, checkpoint.pop("model"))

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # 数据集加载
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        device,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_simple", output_dir, get_rank())

    train(cfg)


if __name__ == '__main__':
    main()
