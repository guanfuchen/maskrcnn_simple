# -*- coding: utf-8 -*-
from collections import OrderedDict
from torch import nn

from maskrcnn_simple.modeling.backbone import resnet


def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))  # model sequential backbone body
    # print(model)
    return model


def build_backbone(cfg):
    return build_resnet_backbone(cfg)
