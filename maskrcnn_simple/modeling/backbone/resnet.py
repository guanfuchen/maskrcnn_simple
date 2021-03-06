# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch.nn.functional as F
from torch import nn

from maskrcnn_simple.layers import FrozenBatchNorm2d
from maskrcnn_simple.layers import Conv2d
from maskrcnn_simple.utils.registry import Registry


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]  # 主干包括Conv+BN+Pool
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]  # ResNet stages
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]  # BottleneckWithFixedBatchNorm

        # Construct the stem module
        self.stem = stem_module(cfg)  # 主干模块

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS  # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP  # width per group

        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS  # stage2的输入通道数，为STEM的输出通道数，64 for ResNet50
        stage2_bottleneck_channels = num_groups * width_per_group  # stage2的bottleneck通道数，1*64 for ResNet50
        # ResNet stage2 out channels，64,128,256,512 for ResNet18和ResNet34
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)  # layer1 layer2 layer3
            stage2_relative_factor = 2 ** (stage_spec.index - 1)  # layer1为1 layer2为2 layer3为4
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor  # bottleneck channels
            out_channels = stage2_out_channels * stage2_relative_factor
            module = _make_stage(
                transformation_module,
                in_channels,  # 输入通道数
                bottleneck_channels,  # 瓶颈通道数
                out_channels,  # 输出通道数
                stage_spec.block_count,  # ResNet某一阶段的block数目
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,  # 是否使用stride=2的conv1x1代替
                first_stride=int(stage_spec.index > 1) + 1,  # block的第一层的stride，默认为1
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        # freeze backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)  # freeze backbone like stem

    def _freeze_backbone(self, freeze_at):
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]  # transformation module

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class BottleneckWithFixedBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__()

        self.downsample = None
        # 当输入和输出通道不相同时，使用downsample
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                FrozenBatchNorm2d(out_channels),  # 这里也适用FrozenBN
            )

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)  # 默认stride_in_1x1=True，那么stride_1x1=stride

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)
        # TODO: specify init for the above

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu_(out)

        return out


class StemWithFixedBatchNorm(nn.Module):
    # 固定BN的主干
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm
})

_STEM_MODULES = Registry({"StemWithFixedBatchNorm": StemWithFixedBatchNorm})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,  # 使用ResNet50 Stages4进行RPN和ROI的特征
    "R-50-C5": ResNet50StagesTo5,
})
