#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 00:40:25 2025

@author: nitaishah
"""

import os
import numpy as np
import open3d as o3d
import torch
from typing import Optional, Tuple

from depth_anything_3.api import DepthAnything3


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
IMAGE_DIR = "/scratch/user/nitaishah/GEN_AI/Phase_3/perspective/pitch0"
ROOT = "/scratch/user/nitaishah/DA3_Project/outputs_30Nov"

DEPTH_DIR = os.path.join(ROOT, "depth")
CAM_DIR   = os.path.join(ROOT, "cameras")
PCD_DIR   = os.path.join(ROOT, "pointclouds")

os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(CAM_DIR,   exist_ok=True)
os.makedirs(PCD_DIR,   exist_ok=True)

# ------------------------------------------------------------
# SELECT PANOS + YAWS
# ------------------------------------------------------------
PANOS = [1, 2, 3]

# All 12 yaw views (30° increments) for best coverage
YAWS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
# For lighter reconstruction, you could use:
# YAWS = [0, 60, 120, 180, 240, 300]

PITCH = 0  # filenames have pitch0

# ------------------------------------------------------------
# LOAD MODEL FROM LOCAL CACHE (OFFLINE)
# ------------------------------------------------------------
LOCAL_MODEL_DIR = (
    "/scratch/user/nitaishah/.cache/huggingface/"
    "models--depth-anything--DA3NESTED-GIANT-LARGE/"
    "snapshots/8615eefb62f2db4f8d6ebaa59160086981672829"
)

print(f"Loading DA3 model from local directory:\n{LOCAL_MODEL_DIR}")

device = torch.device("cuda")
model = DepthAnything3.from_pretrained(LOCAL_MODEL_DIR)
model = model.to(device=device)


# ------------------------------------------------------------
# HELPER: BUILD IMAGE LIST FROM PANOS + YAWS
# ------------------------------------------------------------
def find_image_path(pano_id: int, yaw_deg: int, pitch_deg: int = 0) -> Optional[str]:
    base = f"pano_{pano_id}_yaw{yaw_deg}_pitch{pitch_deg}"
    cand_jpg = os.path.join(IMAGE_DIR, base + ".jpg")
    cand_png = os.path.join(IMAGE_DIR, base + ".png")

    if os.path.isfile(cand_jpg):
        return cand_jpg
    if os.path.isfile(cand_png):
        return cand_png
    return None


images = []
for pano_id in PANOS:
    for yaw in YAWS:
        path = find_image_path(pano_id, yaw, PITCH)
        if path is None:
            print(f"[WARN] Missing image for pano_{pano_id}_yaw{yaw}_pitch{PITCH}")
            continue
        images.append(path)

print(f"Selected {len(images)} images from panos {PANOS} and yaws {YAWS}")
if len(images) == 0:
    raise RuntimeError("No images found for the requested panos/yaws.")


# ------------------------------------------------------------
# GEOMETRY HELPERS
# ------------------------------------------------------------
def to_4x4(ext: np.ndarray) -> np.ndarray:
    """Convert 3x4 or 4x4 extrinsic to 4x4 homogeneous."""
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        T = np.eye(4, dtype=ext.dtype)
        T[:3, :4] = ext
        return T
    raise ValueError(f"Unexpected extrinsic shape {ext.shape}, expected (3,4) or (4,4)")


def fuse_depths_to_world(
    depths: np.ndarray,          # (N, H, W)
    Ks: np.ndarray,              # (N, 3, 3)
    Es_w2c: np.ndarray,          # (N, 3, 4) or (N, 4, 4)
    images_u8: np.ndarray,       # (N, H, W, 3)
    conf: Optional[np.ndarray] = None,   # (N, H, W) if available
    conf_thr: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-view backprojection:
    - For each view, turn (u,v,depth) into 3D camera points via K^{-1},
      then map to world coordinates using (w2c)^{-1}.
    - Return concatenated points + colors.
    """
    N, H, W = depths.shape

    # Pixel grid once: (H*W, 3) with [u, v, 1]
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u_coords, dtype=np.float32)
    pix = np.stack([u_coords, v_coords, ones], axis=-1).reshape(-1, 3)  # (H*W, 3)

    all_points = []
    all_colors = []

    for i in range(N):
        d = depths[i]  # (H, W)
        valid = np.isfinite(d) & (d > 0)

        if conf is not None and conf_thr is not None:
            valid &= conf[i] >= conf_thr

        if not np.any(valid):
            continue

        d_flat   = d.reshape(-1)
        valid_id = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(Ks[i])                # (3, 3)
        w2c_4 = to_4x4(Es_w2c[i])                   # (4, 4)
        c2w_4 = np.linalg.inv(w2c_4)                # (4, 4)

        rays = (K_inv @ pix[valid_id].T)            # (3, M)
        depths_valid = d_flat[valid_id][None, :]    # (1, M)
        pts_cam = rays * depths_valid               # (3, M)

        pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])  # (4, M)
        pts_world = (c2w_4 @ pts_cam_h)[:3, :].T.astype(np.float32)       # (M, 3)

        cols = images_u8[i].reshape(-1, 3)[valid_id].astype(np.float32) / 255.0  # (M, 3)

        all_points.append(pts_world)
        all_colors.append(cols)

    if not all_points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    return points, colors


# ------------------------------------------------------------
# MAIN: MULTI-VIEW INFERENCE + FUSED POINT CLOUD
# ------------------------------------------------------------
print("Running multi-view DA3 inference on selected images...")

pred = model.inference(
    image=images,
    extrinsics=None,
    intrinsics=None,
    align_to_input_ext_scale=True,
    infer_gs=False,
    use_ray_pose=False,
    render_exts=None,
    render_ixts=None,
    render_hw=None,
    process_res=504,
    process_res_method="upper_bound_resize",
    export_dir=None,            # no on-disk export by DA3
    export_format="mini_npz",   # return arrays via Prediction
    export_feat_layers=None,
    conf_thresh_percentile=40.0,
    num_max_points=1_000_000,
    show_cameras=True,
    feat_vis_fps=15,
    export_kwargs={},
)

depths    = pred.depth             # (N, H, W)
Ks        = pred.intrinsics        # (N, 3, 3)
Es_w2c    = pred.extrinsics        # (N, 3, 4) world->camera
images_u8 = pred.processed_images  # (N, H, W, 3) uint8
conf      = pred.conf              # (N, H, W) or None

N = depths.shape[0]
print(f"Prediction: {N} views, depth shape = {depths.shape}")

# ------------------------------------------------------------
# SAVE PER-IMAGE NPY (depth, K, E) FOR DEBUGGING
# ------------------------------------------------------------
for idx, img_path in enumerate(images):
    name = os.path.splitext(os.path.basename(img_path))[0]

    depth_i = depths[idx]
    K_i     = Ks[idx]
    E_i     = Es_w2c[idx]

    np.save(os.path.join(DEPTH_DIR, f"{name}_depth.npy"), depth_i)
    np.save(os.path.join(CAM_DIR,   f"{name}_K.npy"),     K_i)
    np.save(os.path.join(CAM_DIR,   f"{name}_E.npy"),     E_i)

print("Saved per-view depth / K / E .npy files.")

# ------------------------------------------------------------
# FUSE ALL SELECTED VIEWS INTO ONE POINT CLOUD
# ------------------------------------------------------------
conf_thr = None  # set a percentile-based threshold later if you want

points, colors = fuse_depths_to_world(depths, Ks, Es_w2c, images_u8, conf, conf_thr)
print("Fused point count:", points.shape[0])

if points.shape[0] == 0:
    raise RuntimeError("No valid points produced from fusion.")

# ------------------------------------------------------------
# BUILD OPEN3D POINT CLOUD AND SAVE SINGLE FUSED .PLY
# ------------------------------------------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

VOXEL_SIZE = 0.05  # tweak or set to None to skip
if VOXEL_SIZE is not None and VOXEL_SIZE > 0:
    print(f"Voxel downsampling fused cloud with voxel_size={VOXEL_SIZE}")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print("Downsampled fused point count:", np.asarray(pcd.points).shape[0])

fused_out = os.path.join(PCD_DIR, "fused_panos_1_2_3_all_yaws.ply")
o3d.io.write_point_cloud(fused_out, pcd)
print(f"Saved fused point cloud to: {fused_out}")

print("Finished DA3 multi-view fusion for selected panos/yaws.")
