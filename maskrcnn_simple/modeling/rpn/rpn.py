# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_simple.modeling.box_coder import BoxCoder
from maskrcnn_simple.modeling.rpn.anchor_generator import make_anchor_generator
from maskrcnn_simple.modeling.rpn.inference import make_rpn_postprocessor
from maskrcnn_simple.modeling.rpn.loss import make_rpn_loss_evaluator


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        # cls_logits for class logits
        # print('num_anchors:', num_anchors)
        # print('in_channels:', in_channels)
        # out whether the anchor is object
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)  # out the anchour bbox pre

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            # conv和cls_logits和bbox_pred初始化方法
            # 参考论文faster rcnn优化章节
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # 输入特征图x
        logits = []
        bbox_reg = []
        # print('len(x):', len(x))  # not FPN RPN, so feature for RPNHead calc is 1
        for feature in x:
            t = F.relu(self.conv(feature))  # ReLU+Conv_3x3
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        # logits and bbox_reg
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.

    RPN计算模块，从RPN proposals和backbone中读取特征图计算losses
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        # 锚点生成器
        anchor_generator = make_anchor_generator(cfg)  # anchor generator

        # backbone模块输出通道数，for ResNet50_C4为256*4
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        # RPNHead
        head = RPNHead(cfg, in_channels, anchor_generator.num_anchors_per_location()[0])  # generates region proposal

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))  # rpn box coder with weights

        # make rpn postprocessor is to select train box or test box
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)  # using rpn box coder for evaluate rpn loss

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)  # 将RPNHead输出的目标objectness和rpn_box_regression
        anchors = self.anchor_generator(images, features)  # 通过images和features生成锚点

        if self.training:
            # RPN train forward，其中根据targets来选择训练的anchors
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            # RPN test forward
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            # 如果模型仅仅包含RPN，那么boxes为anchors，没必要转换boxes
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            # 端到端的训练，锚点必须转换为boxes
            with torch.no_grad():
                # 选择用来train ROIHead的box，根据targets和anchors
                boxes = self.box_selector_train(anchors, objectness, rpn_box_regression, targets)

        # 根据anchors，objectness，rpn_box_regression和targets计算loss
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(anchors, objectness, rpn_box_regression, targets)
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        # 选择筛选的计算训练的boxes
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)
