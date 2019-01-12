# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
from maskrcnn_simple.utils.visdom_warpper import viz


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        # warmup multi step lr
        # print(list(milestones))
        # print(sorted(milestones))
        # milestones应该是递增
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)  # 将优化器加入到warm up lr中

    def get_lr(self):
        # 根据milestones调整学习率
        warmup_factor = 1
        # 在last_epoch在warmup迭代次数内
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch * 1.0 / self.warmup_iters  # using warmup_iters is int so alpha do not change
                # print('alpha:', alpha)
                # 也就是默认一开始warmup_factor，然后慢慢变回1
                warmup_factor = self.warmup_factor * (1 - alpha) + 1 * alpha  # 线性调整warmup
        lrs = [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)  # for step adjust lr using gamma 0.1
            for base_lr in self.base_lrs
        ]
        # because the lr is changing every iteration, so here the last_epoch is for iteration
        # print('last_epoch:', self.last_epoch)
        # print('warmup_iters:', self.warmup_iters)
        # print('gamma:', self.gamma)
        # print('warmup_factor:', warmup_factor)
        # print(bisect_right(self.milestones, self.last_epoch))  # for example milestones
        # print('len(lrs):', len(lrs))
        # viz.line(self.last_epoch, lrs[0], win='lr_iterations', opts=dict(title='lr_iterations', xlabel='iterations', ylabel='lr'))
        return lrs
