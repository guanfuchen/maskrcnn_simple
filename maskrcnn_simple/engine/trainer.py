# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_simple.utils.comm import get_world_size
from maskrcnn_simple.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()  # world size
    # print('world_size:', world_size)
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    device,
):
    logger = logging.getLogger("maskrcnn_simple.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)  # 数据loader最大迭代次数
    # start_iter = arguments["iteration"]
    start_iter = 0
    model.train()  # 设置模型训练模式
    start_training_time = time.time()  # 开始训练时间
    end = time.time()
    # here data_loader is get SOLVER.MAX_ITER to break
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        # print('iteration:{}'.format(iteration))
        data_time = time.time() - end  # data读取时间
        iteration = iteration + 1
        # arguments["iteration"] = iteration

        scheduler.step()  # every iteration change the scheduler

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end  # batch time
        end = time.time()
        meters.update(time=batch_time, data=data_time)  # meters更新time

        # eta_seconds剩余时间
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # 迭代次数20次同时iteration为max_iter时也就是one epoch
        if iteration % 20 == 0 or iteration == max_iter:

            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),  # meters
                    lr=optimizer.param_groups[0]["lr"],  # 优化器lr
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,  # cuda内存调用
                )
            )
        # if iteration % checkpoint_period == 0:
        #     checkpointer.save("model_{:07d}".format(iteration), **arguments)
        # if iteration == max_iter:
        #     checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
