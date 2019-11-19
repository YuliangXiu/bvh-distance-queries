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
import os

import time

import argparse

try:
    input = raw_input
except NameError:
    pass

import open3d as o3d
import pyigl as igl
from iglhelpers import p2e, e2p

import tqdm

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import numpy as np
import tqdm

from loguru import logger

from psbody.mesh import Mesh
import bvh_distance_queries


if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-fn', type=str, dest='mesh_fn',
                        help='A mesh file (.obj, .ply, e.t.c.) to be checked' +
                        ' for collisions')
    parser.add_argument('--num-query-points', type=int, default=1,
                        dest='num_query_points',
                        help='Number of random query points')
    parser.add_argument('--timing-iters', type=int, default=1000,
                        dest='timing_iters')

    args, _ = parser.parse_known_args()

    mesh_fn = args.mesh_fn
    num_query_points = args.num_query_points
    timing_iters = args.timing_iters

    input_mesh = Mesh(filename=mesh_fn)

    torch.manual_seed(0)

    logger.info(f'Number of triangles = {input_mesh.f.shape[0]}')

    v = input_mesh.v
    v -= v.mean(keepdims=True, axis=0)

    vertices = torch.tensor(v, dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.f.astype(np.int64),
                         dtype=torch.long,
                         device=device)

    query_points = torch.rand([1, num_query_points, 3], dtype=torch.float32,
                              device=device) * 2 - 1
    query_points_np = query_points.detach().cpu().numpy().squeeze(
        axis=0).astype(np.float32).reshape(num_query_points, 3)

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    m = bvh_distance_queries.BVH()
    elapsed = 0

    for n in tqdm.tqdm(range(timing_iters)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        distances, closest_points, closest_faces = m(triangles, query_points)
        torch.cuda.synchronize()
        elapsed += (time.perf_counter() - start)

    logger.info(f'CUDA Elapsed time {elapsed / timing_iters}')
    #  print(outputs[2])
    distances = distances.detach().cpu().numpy()
    closest_points = closest_points.detach().cpu().numpy().squeeze()

    #  logger.info(f'Distances: {distances ** 2}')
    #  logger.info(f'Closest points: {closest_points}')

    #  outputs = outputs.detach().cpu().numpy().squeeze()

    sqrD = igl.eigen.MatrixXd()
    closest_faces = igl.eigen.MatrixXi()
    closest_points_eig = igl.eigen.MatrixXd()

    query_points_eigen = p2e(query_points_np)

    elapsed = 0

    for n in tqdm.tqdm(range(timing_iters)):
        start = time.perf_counter()
        # Find the closest points on the SMPL-X mesh
        igl.point_mesh_squared_distance(query_points_eigen,
                                        p2e(v),
                                        p2e(input_mesh.f.astype(np.int64)),
                                        sqrD, closest_faces, closest_points_eig)
        elapsed += (time.perf_counter() - start)
    logger.info(f'LibIGL Elapsed time {elapsed / timing_iters}')
