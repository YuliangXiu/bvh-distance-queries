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
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import torch
import torch.nn as nn
import torch.autograd as autograd

import bvh_distance_queries_cuda


class BVHFunction(autograd.Function):

    @staticmethod
    def forward(ctx, triangles, points):
        outputs = bvh_distance_queries_cuda.distance_queries(triangles, points)
        ctx.save_for_backward(outputs, triangles)
        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


class BVH(nn.Module):

    def __init__(self):
        super(BVH, self).__init__()

    def forward(self, triangles: torch.Tensor, points: torch.Tensor):
        distances, closest_points, closest_faces = BVHFunction.apply(
            triangles, points)

        return distances, closest_points, closest_faces
