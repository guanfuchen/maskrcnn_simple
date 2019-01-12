# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_simple.structures.bounding_box import BoxList


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors
    对于给定的图像大小和特征图大小，计算锚点集合
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        # anchor_strides的len==1
        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]  # anchor_stride=16 for ResNet50
            cell_anchors = [generate_anchors(anchor_stride, sizes, aspect_ratios).float()]
            # print('cell_anchors:', cell_anchors)
        # else:
        #     if len(anchor_strides) != len(sizes):
        #         raise RuntimeError("FPN should have #anchor_strides == #sizes")
        #     cell_anchors = [
        #         generate_anchors(anchor_stride, (size,), aspect_ratios).float()
        #         for anchor_stride, size in zip(anchor_strides, sizes)
        #     ]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        # 每一个位置anchor生成的数目，len(aspect_ratios)*len(sizes)
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        """
        grid anchors
        :param grid_sizes: grid sizes代表着特征图的sizes
        :return:
        """
        anchors = []
        # print('grid_sizes:', grid_sizes)
        # print('strides:', self.strides)
        # print('cell_anchors:', self.cell_anchors)
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            # cell_anchors代表着不同sizes和aspect ratios，在不同strides，grid_sizes
            grid_height, grid_width = size
            device = base_anchors.device

            # shifts_x
            shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
            # shifts_y
            shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            # print('shift_x:', shift_x)
            # print('shift_y:', shift_y)
            # print('shift_x.shape:', shift_x.shape)
            # print('shift_y.shape:', shift_y.shape)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            # print('shifts.shape:', shifts.shape)
            anchor = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            # print('anchor.shape:', anchor.shape)
            anchors.append(anchor)  # 根据base anchor进行偏移，包括

        # print('anchors:', anchors)
        # print('base_anchors:', base_anchors)
        # print('len(anchors):', len(anchors))
        return anchors

    # 针对某些boxlist不可见，设置visibility
    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox

        # straddle_thresh设置为0
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        # 根据image list和feature maps生成锚点
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]  # 每一个特征图的sizes
        # print('grid_sizes:', grid_sizes)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)  # grid sizes
        anchors = []
        # 图像宽度和高度
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                # BoxList默认锚点格式为xyxy
                # print('anchors_per_feature_map.shape:', anchors_per_feature_map.shape)
                # 将anchors_per_feature_map转换为boxlist
                boxlist = BoxList(anchors_per_feature_map, (image_width, image_height), mode="xyxy")
                # 设置boxlist中相对于原图像visible属性
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def make_anchor_generator(config):
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    # if config.MODEL.RPN.USE_FPN:
    #     assert len(anchor_stride) == len(
    #         anchor_sizes
    #     ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    # else:
    #     assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"

    assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    # 根据anchor sizes，aspect ratios，anchor stride和straddle thresh来构建锚点生成器
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh)
    return anchor_generator


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """
    Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    生成锚点矩阵
    """
    # 当前的sizes是相对于stride，当sizes=128，stride=16，也就是在卷积特征图上的sizes_conv=128/16=8
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    通过aspect ratios和scales来生成锚点窗口
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1  # 获得基础anchor，不加入ratio和scale
    # print('anchor:', anchor)
    anchors = _ratio_enum(anchor, aspect_ratios)  # 获得加入aspect ratios和anchor
    # print('anchors:', anchors)
    # 获得加入scales的anchors
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    # print('anchors:', anchors)
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    anchor using xyxy represents，then using whxy represents
    """
    w = anchor[2] - anchor[0] + 1  # 15-0+1
    h = anchor[3] - anchor[1] + 1  # 15-0+1
    x_ctr = anchor[0] + 0.5 * (w - 1)  # 0+0.5*(16-1)
    y_ctr = anchor[1] + 0.5 * (h - 1)  # 0+0.5*(16-1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)  # get anchor w h ctr_x ctr_y
    size = w * h
    size_ratios = size / ratios  # ratios=h/w ws*ratios*ws=sizes
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)  # using ws, hw, x_ctr, y_ctr get anchor xyxy represents
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)  # get scale anchor
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
