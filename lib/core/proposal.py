# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

def get_index2D(indices, shape):
    batch_size = indices.shape[0]
    num_people = indices.shape[1]
    indices_x = torch.div(indices, shape[1], rounding_mode='trunc').reshape(batch_size, num_people, -1)
    indices_y = (indices % shape[1]).reshape(batch_size, num_people, -1)
    indices = torch.cat([indices_x, indices_y], dim=2)
    return indices

def max_pool2D(inputs, kernel=3):
    padding = (kernel - 1) // 2  
    max = F.max_pool2d(inputs, kernel_size=kernel, stride=1, padding=padding)
    keep = (inputs == max).float()
    return keep * inputs  

def nms2D(prob_map, max_num):
    batch_size = prob_map.shape[0]
    prob_map_nms = max_pool2D(prob_map)
    prob_map_nms_reshape = prob_map_nms.reshape(batch_size, -1)
    topk_values, topk_flatten_index = prob_map_nms_reshape.topk(max_num)
    topk_index = get_index2D(topk_flatten_index, prob_map[0].shape)
    return topk_values, topk_index, topk_flatten_index