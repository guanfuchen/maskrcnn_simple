# -*- coding: utf-8 -*-
from torch import nn

from maskrcnn_simple.modeling.backbone.backbone import build_backbone
from maskrcnn_simple.modeling.roi_heads.roi_heads import build_roi_heads
from maskrcnn_simple.modeling.rpn.rpn import build_rpn
from maskrcnn_simple.structures.image_list import to_image_list


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)  # build backbone
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # training mode targets is bounding box
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)  # 将图像转换为list
        # print('images.tensors.shape:', images.tensors.shape)
        # print('targets:', targets)
        features = self.backbone(images.tensors)  # backbone forward the images

        proposals, proposal_losses = self.rpn(images, features, targets)  # rpn区域建议网络，训练包括建议框和损失

        if self.roi_heads:
            # 如果存在roi_heads
            x, result, detector_losses = self.roi_heads(features, proposals, targets)  # 区域建议和特征以及targets，训练包括
        else:
            # RPN-only models don't have roi_heads
            # x = features
            result = proposals
            detector_losses = {}

        # training mode return losses
        if self.training:
            losses = {}
            # 训练过程返回losses包括ROI head loss和RPN loss
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result


def build_detection_model(cfg):
    model = GeneralizedRCNN(cfg)
    return model

