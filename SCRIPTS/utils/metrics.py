#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:25:19 2025

@author: nitaishah
"""

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:
    o3d = None


def rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Root mean squared error over all pixels.
    """
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """
    RMSE on masked region.
    pred/gt: (B,1,H,W), mask: (B,1,H,W) boolean or {0,1}
    """
    diff = (pred - gt) ** 2
    num = mask.sum().item() + 1e-6
    return torch.sqrt((diff * mask).sum() / num).item()


def chamfer_distance_pcd(pcd_a, pcd_b) -> float:
    """
    Chamfer distance using Open3D KD-tree (if available).
    pcd_a/b: open3d.geometry.PointCloud
    Returns mean of nearest-neighbor distances (A->B + B->A)/2
    """
    if o3d is None:
        raise ImportError("Open3D not available for Chamfer distance")

    kdt_b = o3d.geometry.KDTreeFlann(pcd_b)
    kdt_a = o3d.geometry.KDTreeFlann(pcd_a)

    a_pts = np.asarray(pcd_a.points)
    b_pts = np.asarray(pcd_b.points)

    dists_a = []
    for p in a_pts:
        _, idx, dist2 = kdt_b.search_knn_vector_3d(p, 1)
        dists_a.append(np.sqrt(dist2[0]))
    dists_b = []
    for p in b_pts:
        _, idx, dist2 = kdt_a.search_knn_vector_3d(p, 1)
        dists_b.append(np.sqrt(dist2[0]))

    return 0.5 * (float(np.mean(dists_a)) + float(np.mean(dists_b)))
