# -*- coding: utf-8 -*-
import argparse

import torch
from maskrcnn_simple.utils.comm import get_rank

from maskrcnn_simple.data.build import make_data_loader
from maskrcnn_simple.engine.trainer import do_train
from maskrcnn_simple.modeling.detector.detector import build_detection_model
from maskrcnn_simple.config import cfg
from maskrcnn_simple.solver import make_optimizer, make_lr_scheduler
from maskrcnn_simple.utils.logger import setup_logger
from maskrcnn_simple.utils.miscellaneous import mkdir


def train(cfg):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

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
