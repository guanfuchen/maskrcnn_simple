# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    # for optimizing model
    params = []
    for key, value in model.named_parameters():
        # optimize model
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR  # solver lr
        weight_decay = cfg.SOLVER.WEIGHT_DECAY  # solver weight decay
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR  # bias lr是weight的两倍
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS  # bias的学习率衰减是0
        # for optimizing params type param+lr+weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # print('params:', params)
    # print('len(params):', len(params))
    # momentum if params exist lr so the lr is not useful
    optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    # learning scheduler
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
