# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys

from typing import Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd

import bvh_distance_queries_cuda


class BVHFunction(autograd.Function):
    QUEUE_SIZE = 128
    SORT_POINTS_BY_MORTON = True

    @staticmethod
    def forward(ctx, triangles, points):
        outputs = bvh_distance_queries_cuda.distance_queries(
            triangles, points,
            queue_size=BVHFunction.QUEUE_SIZE,
            sort_points_by_morton=BVHFunction.SORT_POINTS_BY_MORTON,
        )
        ctx.save_for_backward(outputs, triangles)
        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


class BVH(nn.Module):

    def __init__(self,
                 sort_points_by_morton: bool = True,
                 queue_size: int = 128) -> None:
        super(BVH, self).__init__()
        assert queue_size in [32, 64, 128, 256, 512, 1024], (
            f'Queue/Stack size must be in {str[32, 64, 128, 256, 512, 1024]()}'
        )
        BVHFunction.QUEUE_SIZE = queue_size
        BVHFunction.SORT_POINTS_BY_MORTON = sort_points_by_morton

    def forward(
        self, triangles: torch.Tensor,
        points: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:

        distances, closest_points, closest_faces = BVHFunction.apply(
            triangles, points)

        return distances, closest_points, closest_faces
