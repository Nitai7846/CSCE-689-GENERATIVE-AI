#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:24:17 2025

@author: nitaishah
"""

import os
import numpy as np
import torch

# Prefer plyfile (safe with NumPy 2.x), fall back to Open3D if available
try:
    from plyfile import PlyData  # tiny, pure-Python
    _HAS_PLYFILE = True
except Exception:
    _HAS_PLYFILE = False

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


def _read_points_from_ply(ply_path: str) -> np.ndarray:
    """
    Read XYZ points from a .ply using plyfile first (safer for NumPy 2.x),
    otherwise fall back to Open3D if present.

    Returns:
      (N, 3) float32 in meters.
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)

    if _HAS_PLYFILE:
        ply = PlyData.read(ply_path)
        if "vertex" not in ply:
            raise ValueError(f"No 'vertex' element in {ply_path}")
        v = ply["vertex"].data

        def _field(name: str):
            if name in v.dtype.names:
                return v[name]
            # case-insensitive fallback
            for n in v.dtype.names:
                if n.lower() == name:
                    return v[n]
            raise KeyError(f"Missing '{name}' in vertex fields of {ply_path}")

        x = np.asarray(_field("x"), dtype=np.float32)
        y = np.asarray(_field("y"), dtype=np.float32)
        z = np.asarray(_field("z"), dtype=np.float32)
        pts = np.stack([x, y, z], axis=1)
        # clean NaNs/Infs
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts.astype(np.float32, copy=False)

    if _HAS_O3D:
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts

    raise ImportError("Install `plyfile` (recommended) or `open3d` to read .ply files.")


def pointcloud_to_range(
    ply_path: str,
    H: int = 1088,
    W: int = 1440,
    r_clip: float = 120.0,
    fill: float = 0.0,
) -> torch.Tensor:
    """
    Project a LiDAR point cloud (.ply) to a 2D range image (on-the-fly, no disk writes).

    Mapping (spherical):
      r  = sqrt(x^2+y^2+z^2)
      az = atan2(y, x)           ∈ (-pi, pi]
      el = arcsin(z / r)         ∈ [-pi/2, pi/2]

    Discretization:
      az → [0, W-1] uniformly across 360°
      el → [0, H-1] linearly between min/max elevation in the cloud

    Args:
      ply_path: str, path to .ply
      H, W   : output resolution (paper uses 1088 × 1440)
      r_clip : clip far ranges for numerical stability (meters)
      fill   : default value for empty pixels

    Returns:
      torch.Tensor of shape (1, 1, H, W), dtype=float32
    """
    pts = _read_points_from_ply(ply_path)
    if pts.shape[0] == 0:
        img = np.full((H, W), fill, dtype=np.float32)
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y + z * z).astype(np.float32)
    r = np.clip(r, 1e-6, r_clip)

    az = np.arctan2(y, x)  # (-pi, pi]
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))  # [-pi/2, pi/2]

    # Azimuth → [0, W-1]
    az_idx = ((az + np.pi) / (2 * np.pi) * W).astype(np.int32)
    az_idx = np.clip(az_idx, 0, W - 1)

    # Elevation range (fallback if degenerate)
    el_min, el_max = el.min(), el.max()
    if not np.isfinite(el_min) or not np.isfinite(el_max) or el_max <= el_min:
        el_min, el_max = -np.pi / 4, np.pi / 4
    el_idx = ((el - el_min) / (el_max - el_min + 1e-6) * H).astype(np.int32)
    el_idx = np.clip(el_idx, 0, H - 1)

    # Keep nearest return per pixel: use a min-reduction
    # Start with +inf so np.minimum.at can accumulate mins
    img = np.full((H, W), np.inf, dtype=np.float32)
    lin = el_idx * W + az_idx
    # Reduce by pixel index
    np.minimum.at(img.ravel(), lin, r)

    # Replace untouched pixels (inf) with fill value
    if fill != np.inf:
        img[~np.isfinite(img)] = np.float32(fill)

    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
