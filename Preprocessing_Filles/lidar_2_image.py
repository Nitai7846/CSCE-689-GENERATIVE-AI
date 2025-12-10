#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 19:10:56 2025

@author: nitaishah
"""

# simple_lidar_image_pipeline.py
# Minimal, run-in-editor script (no terminal args).
# Converts one PLY -> LiDAR range image (paper layout) -> back to points.

import os
import numpy as np
import open3d as o3d

# =======================
# ======= CONFIG ========
# =======================
PLY_IN      = "/Volumes/One Touch/scenario31/unit1/lidar_data/lidar_data_173.ply"                # <- change me
OUT_PREFIX  = "/Volumes/One Touch/GEN_AI_PROJECT/LIDAR_IMAGE/s31_frame0001"           # saves *_range.npy / *_recon.ply / *_range.png
SAVE_PNG    = True                                       # quicklook image (normalized)
DO_PROJECT  = True                                       # PLY -> image
DO_RECON    = True                                       # image -> points (geometric)

# =======================
# ===== CONSTANTS =======
# =======================
H, W = 1088, 1440  # paper resolution

# elevation piecewise (degrees), rows:
# 0..219:   [-60, -5]   step 0.25
# 220..859: [-5, +5]    step 0.015625
# 860..1087:[+5, +62]   step 0.25
PHI0_MIN, STEP0, ROW0_END   = -60.0, 0.25, 219
PHI1_MIN, STEP1, ROW1_START =  -5.0, 0.015625, 220
PHI1_MAX, ROW1_END          =  +5.0, 859
PHI2_MIN, STEP2, ROW2_START =  +5.0, 0.25, 860

# =======================
# ==== ANGLE HELPERS ====
# =======================
def row_to_phi_deg(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    phi = np.empty_like(rows, dtype=np.float64)
    m0 = rows <= ROW0_END
    phi[m0] = PHI0_MIN + STEP0 * rows[m0]
    m1 = (rows >= ROW1_START) & (rows <= ROW1_END)
    phi[m1] = PHI1_MIN + STEP1 * (rows[m1] - ROW1_START)
    m2 = rows >= ROW2_START
    phi[m2] = PHI2_MIN + STEP2 * (rows[m2] - ROW2_START)
    return phi

def phi_deg_to_row(phi_deg: np.ndarray) -> np.ndarray:
    phi_deg = np.asarray(phi_deg, dtype=np.float64)
    rows = np.empty_like(phi_deg, dtype=np.int64)
    m0 = phi_deg < PHI1_MIN
    rows[m0] = np.round((phi_deg[m0] - PHI0_MIN) / STEP0).astype(np.int64)
    m1 = (phi_deg >= PHI1_MIN) & (phi_deg <= PHI1_MAX)
    rows[m1] = ROW1_START + np.round((phi_deg[m1] - PHI1_MIN) / STEP1).astype(np.int64)
    m2 = phi_deg > PHI1_MAX
    rows[m2] = ROW2_START + np.round((phi_deg[m2] - PHI2_MIN) / STEP2).astype(np.int64)
    return np.clip(rows, 0, H - 1)

def col_to_theta(cols: np.ndarray, W_: int = W) -> np.ndarray:
    cols = np.asarray(cols, dtype=np.float64)
    return ((cols + 0.5) / W_) * (2.0 * np.pi)  # center-of-pixel

def theta_to_col(theta: np.ndarray, W_: int = W) -> np.ndarray:
    theta = np.mod(theta, 2.0 * np.pi)
    cols = np.floor(theta / (2.0 * np.pi) * W_).astype(np.int64)
    return np.clip(cols, 0, W_ - 1)

# =======================
# ====== I/O HELPERS ====
# =======================
def read_ply_xyz(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)

def save_points_ply(points_xyz: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))
    o3d.io.write_point_cloud(path, pcd)

def save_range_npy_and_png(prefix: str, range_img: np.ndarray, save_png: bool):
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    np.save(prefix + "_range.npy", range_img.astype(np.float32))
    if save_png:
        try:
            from PIL import Image
            rng = range_img.copy()
            good = np.isfinite(rng) & (rng > 0)
            if good.any():
                lo, hi = np.percentile(rng[good], (1, 99))
                lo = float(lo); hi = float(hi if hi > lo else rng[good].max())
                norm = np.clip((rng - lo) / (hi - lo + 1e-12), 0, 1)
            else:
                norm = np.zeros_like(rng, dtype=np.float32)
            img = (norm * 255).astype(np.uint8)
            Image.fromarray(img).save(prefix + "_range.png")
        except Exception as e:
            print(f"[warn] PNG save skipped: {e}")

# =======================
# ====== PROJECTOR ======
# =======================
def points_to_range_image(points_xyz: np.ndarray) -> np.ndarray:
    """
    Project XYZ (N,3) to a paper-accurate range image (H=1088, W=1440).
    Per-pixel: keep the NEAREST range (min r). Others are dropped.
    """
    P = np.asarray(points_xyz, dtype=np.float64)
    assert P.ndim == 2 and P.shape[1] == 3, "points must be (N,3)"
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.sqrt(x*x + y*y + z*z) + 1e-12
    theta = np.arctan2(y, x)                       # [-pi, pi]
    phi = np.arctan2(z, np.sqrt(x*x + y*y))        # (-pi/2, pi/2)
    cols = theta_to_col(theta)                     # (N,)
    rows = phi_deg_to_row(np.degrees(phi))         # (N,)
    flat = rows * W + cols

    # Fill nearest-first
    order = np.argsort(r)
    flat_sorted = flat[order]
    r_sorted = r[order]

    range_img = np.full(H * W, np.inf, dtype=np.float32)
    filled = np.zeros(H * W, dtype=bool)
    for i, f in enumerate(flat_sorted):
        if not filled[f]:
            range_img[f] = r_sorted[i]
            filled[f] = True

    return range_img.reshape(H, W)

# =======================
# ====== INVERSE ========
# =======================
def range_image_to_points(range_img: np.ndarray) -> np.ndarray:
    """
    Geometric inverse: (1,H,W) or (H,W) range image -> XYZ (M,3).
    Only positive, finite ranges are converted.
    """
    img = np.asarray(range_img, dtype=np.float64)
    if img.ndim == 3:
        assert img.shape[0] == 1
        img = img[0]
    assert img.shape == (H, W), f"Expected {(H,W)}, got {img.shape}"

    v_idx = np.arange(H)
    u_idx = np.arange(W)
    Phi = np.deg2rad(row_to_phi_deg(v_idx))  # (H,)
    Theta = col_to_theta(u_idx)              # (W,)
    Phi, Theta = np.meshgrid(Phi, Theta, indexing="ij")  # (H,W)

    r = img
    valid = np.isfinite(r) & (r > 0.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    cos_phi = np.cos(Phi[valid]); sin_phi = np.sin(Phi[valid])
    cos_th  = np.cos(Theta[valid]); sin_th  = np.sin(Theta[valid])
    x = r[valid] * cos_phi * cos_th
    y = r[valid] * cos_phi * sin_th
    z = r[valid] * sin_phi
    return np.column_stack((x, y, z))

# =======================
# ========= RUN =========
# =======================
if __name__ == "__main__":
    if DO_PROJECT:
        xyz = read_ply_xyz(PLY_IN)
        rng = points_to_range_image(xyz)
        save_range_npy_and_png(OUT_PREFIX, rng, SAVE_PNG)
        print(f"[OK] wrote: {OUT_PREFIX}_range.npy" + (" and _range.png" if SAVE_PNG else ""))

    if DO_RECON:
        rng = np.load(OUT_PREFIX + "_range.npy")
        pts = range_image_to_points(rng)
        ply_out = OUT_PREFIX + "_recon.ply"
        save_points_ply(pts, ply_out)
        print(f"[OK] wrote: {ply_out}  (reconstructed from image)")
