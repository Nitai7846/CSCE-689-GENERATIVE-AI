#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 22:31:54 2025

@author: nitaishah
"""

import os
import numpy as np
import open3d as o3d
from typing import Union

# ====== EDIT THESE ======
GT_PLY   = "/Volumes/One Touch/scenario31/unit1/lidar_data/lidar_data_6920.ply"
PRED_PLY = "/Volumes/One Touch/GEN_AI_PROJECT/phase_2/OUTPUTS_HAIKOO/pred_pointcloud2.ply"
VOXEL_SIZE: Union[float, int] = 0.05   # set 0 to disable
USE_ICP = True
ICP_MAX_ITERS = 50
ICP_DISTANCE_THRESH = 0.5   # meters
GT_COLOR   = [0.15, 0.45, 1.00]  # blue-ish
PRED_COLOR = [1.00, 0.55, 0.10]  # orange-ish
# ========================

def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    path = _norm(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points in {path}")
    return pcd

def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: Union[float, int]) -> o3d.geometry.PointCloud:
    if voxel_size and float(voxel_size) > 0:
        pcd = pcd.voxel_down_sample(float(voxel_size))
    return pcd

def run_icp(source: o3d.geometry.PointCloud,
            target: o3d.geometry.PointCloud,
            dist_thresh: float,
            max_iters: int) -> o3d.geometry.PointCloud:
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iters))
    reg = o3d.pipelines.registration.registration_icp(
        source, target, float(dist_thresh), np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )
    print(f"[ICP] fitness={reg.fitness:.4f}, inlier_rmse={reg.inlier_rmse:.4f}")
    return source.transform(reg.transformation.copy())

def main():
    gt   = load_pcd(GT_PLY)
    pred = load_pcd(PRED_PLY)

    gt   = preprocess(gt, VOXEL_SIZE)
    pred = preprocess(pred, VOXEL_SIZE)

    gt.paint_uniform_color(GT_COLOR)
    pred.paint_uniform_color(PRED_COLOR)

    if USE_ICP:
        pred = run_icp(pred, gt, ICP_DISTANCE_THRESH, ICP_MAX_ITERS)

    o3d.visualization.draw_geometries(
        [gt, pred],
        window_name="GT (blue) + Pred (orange)",
        width=1280, height=800
    )

if __name__ == "__main__":
    main()

