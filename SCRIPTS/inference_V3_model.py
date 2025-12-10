#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 22:13:40 2025

@author: nitaishah
"""

#!/usr/bin/env python3
# Compare GT vs Pred: show range images, show 3D point clouds, save predicted PLY (plyfile-only)
import os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from importlib.machinery import SourceFileLoader

# ===================== EDIT THESE PATHS/VALUES =====================
CONFIG = {
    "PROJECT_ROOT": "/scratch/user/dhruvpatel144/genai/scripts",
    "FUSION_FILE": "models/fusion_encoder_paper_v3.py",
    "DECODER_FILE": "models/lidar_decoder_paper_v3.py",
    "LOSSES_FILE":  "losses.py",


    "CHECKPOINT": "/scratch/user/dhruvpatel144/genai/data/OUTPUTS/runs/model_v3/checkpoints/epoch_10.pt",
    "EMBEDDING_PT": "/scratch/user/dhruvpatel144/genai/embeddings/s31/unit1/batch_0865_embeddings.pt",
    "EMB_INDEX": 0,

    "USE_AUTO_LIDAR": False,
    "LIDAR_DIR": "/scratch/user/dhruvpatel144/genai/data/scenario31/unit1/lidar_data_renamed",
    "LIDAR_PLY":  "/scratch/user/dhruvpatel144/genai/data/scenario31/unit1/lidar_data_renamed/lidar_data_6920.ply",
    "PREPROC_BATCH_SIZE": 8,

    # Loss settings (for reporting)
    "ALPHA": 0.8, "BETA": 0.2, "CENTER_BAND": (220, 859), "W_CENTER": 10.0,

    # # Projection params (match training)
    # "H": 1088, "W": 1440, "R_CLIP": 120.0, "FILL": 0.0,

    # Projection params (must match training)
    "H": 1024, "W": 3, "R_CLIP": 120.0, "FILL": 0.0,

    # Decoder geometry (must match training)
    "UPS": 5, "STRIDE_H": 2, "STRIDE_W": 1,

    # Device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # Visualization / export
    "SUBSAMPLE_PC": 1,     # plot every k-th point to keep scatter light; set 1 for all points
    "SAVE_PRED_PLY": True,
    "PRED_PLY_OUT": "/scratch/user/dhruvpatel144/genai/data/OUTPUTS/infer_preview/predicted_pointcloud.ply",
}
# ===================================================================

# Ensure project root on path
if CONFIG["PROJECT_ROOT"] not in sys.path:
    sys.path.insert(0, CONFIG["PROJECT_ROOT"])

# Dynamic import of your exact files (avoid models/__init__.py)
def _load_symbol(path, symbol):
    abspath = os.path.join(CONFIG["PROJECT_ROOT"], path)
    modname = f"_dyn_{os.path.basename(path).replace('.', '_')}"
    mod = SourceFileLoader(modname, abspath).load_module()
    return getattr(mod, symbol)

MultimodalTransformerEncoder = _load_symbol(CONFIG["FUSION_FILE"],  "PerceiverFusionEncoderV3")
LidarDecoderPaper           = _load_symbol(CONFIG["DECODER_FILE"], "LidarDecoderV3")
weighted_mse_ssim           = _load_symbol(CONFIG["LOSSES_FILE"],   "weighted_mse_ssim")

# ---- plyfile I/O ----
try:
    from plyfile import PlyData, PlyElement
except Exception:
    print("ERROR: install plyfile: pip install plyfile", file=sys.stderr)
    raise

def save_points_as_ply(path: str, pts_xyz: np.ndarray):
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
    if pts_xyz.ndim != 2 or pts_xyz.shape[1] != 3:
        raise ValueError("pts_xyz must be (N,3)")
    verts = np.empty(pts_xyz.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4')])
    verts['x'], verts['y'], verts['z'] = pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2]
    el = PlyElement.describe(verts, 'vertex')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([el], text=False).write(path)

def _read_points_from_ply(ply_path: str) -> np.ndarray:
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"No 'vertex' element in {ply_path}")
    v = ply["vertex"].data

    def _field(name: str):
        if name in v.dtype.names: return v[name]
        for n in v.dtype.names:
            if n.lower() == name: return v[n]
        raise KeyError(f"Missing '{name}' in vertex fields of {ply_path}")

    x = np.asarray(_field("x"), dtype=np.float32)
    y = np.asarray(_field("y"), dtype=np.float32)
    z = np.asarray(_field("z"), dtype=np.float32)
    pts = np.stack([x, y, z], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts.astype(np.float32, copy=False)

# ---- Projection (forward) that also returns elevation bounds for invertibility ----
def pointcloud_to_range_with_meta(
    ply_path: str,
    H: int, W: int, r_clip: float, fill: float
):
    """
    Returns:
      img:  (1,1,H,W) float32
      meta: dict with el_min, el_max (floats)
    """
    pts = _read_points_from_ply(ply_path)
    if pts.shape[0] == 0:
        img = np.full((H, W), fill, dtype=np.float32)
        return torch.from_numpy(img)[None, None], {"el_min": -np.pi/4, "el_max": np.pi/4}

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x*x + y*y + z*z).astype(np.float32)
    r = np.clip(r, 1e-6, r_clip)

    az = np.arctan2(y, x)                                # (-pi, pi]
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))            # [-pi/2, pi/2]

    az_idx = ((az + np.pi) / (2 * np.pi) * W).astype(np.int32)
    az_idx = np.clip(az_idx, 0, W - 1)

    el_min, el_max = el.min(), el.max()
    if not np.isfinite(el_min) or not np.isfinite(el_max) or el_max <= el_min:
        el_min, el_max = -np.pi / 4, np.pi / 4
    el_idx = ((el - el_min) / (el_max - el_min + 1e-6) * H).astype(np.int32)
    el_idx = np.clip(el_idx, 0, H - 1)

    img = np.full((H, W), np.inf, dtype=np.float32)
    lin = el_idx * W + az_idx
    np.minimum.at(img.ravel(), lin, r)
    img[~np.isfinite(img)] = np.float32(fill)

    meta = {"el_min": float(el_min), "el_max": float(el_max)}
    return torch.from_numpy(img)[None, None], meta

# ---- Back-projection: range image → 3D points using known el_min/el_max ----
def range_to_pointcloud(range_img: np.ndarray, el_min: float, el_max: float, r_clip: float) -> np.ndarray:
    """
    range_img: (H,W) float32 in meters
    Returns (N,3) xyz (meters) for pixels with r>0
    """
    H, W = range_img.shape
    # Build grid of angles
    az = (np.arange(W, dtype=np.float32) + 0.5) / W * (2*np.pi) - np.pi         # center-of-bin
    el = (np.arange(H, dtype=np.float32) + 0.5) / H * (el_max - el_min) + el_min
    az_grid, el_grid = np.meshgrid(az, el)

    r = np.clip(range_img.astype(np.float32), 0.0, r_clip)
    mask = r > 0.0
    if not np.any(mask):
        return np.zeros((0,3), dtype=np.float32)

    r_sel  = r[mask]
    az_sel = az_grid[mask]
    el_sel = el_grid[mask]

    cos_el = np.cos(el_sel)
    x = r_sel * cos_el * np.cos(az_sel)
    y = r_sel * cos_el * np.sin(az_sel)
    z = r_sel * np.sin(el_sel)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts

# ---- helpers ----
def _extract_batch_idx(embedding_pt_path: str) -> int:
    m = re.search(r"batch_(\d+)_embeddings\.pt$", os.path.basename(embedding_pt_path))
    if not m:
        raise ValueError(f"Cannot parse batch index from filename: {embedding_pt_path}")
    return int(m.group(1))

def _auto_lidar_path(embedding_pt_path: str, emb_index: int) -> str:
    b = _extract_batch_idx(embedding_pt_path)
    global_idx = b * CONFIG["PREPROC_BATCH_SIZE"] + emb_index
    return os.path.join(CONFIG["LIDAR_DIR"], f"lidar_data_{global_idx:04d}.ply")

def _load_checkpoint(ckpt_path, fusion, decoder, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "fusion" in ckpt and "decoder" in ckpt:
        fusion.load_state_dict(ckpt["fusion"], strict=True)
        decoder.load_state_dict(ckpt["decoder"], strict=True)
        ep = ckpt.get("epoch", None)
    else:
        f_sd = {k.split("fusion.", 1)[-1]: v for k, v in ckpt.items() if k.startswith("fusion.")}
        d_sd = {k.split("decoder.", 1)[-1]: v for k, v in ckpt.items() if k.startswith("decoder.")}
        if f_sd: fusion.load_state_dict(f_sd, strict=False)
        if d_sd: decoder.load_state_dict(d_sd, strict=False)
        ep = None
    fusion.to(device).eval()
    decoder.to(device).eval()
    return ep

# =========================== main ===========================
def main():
    device = torch.device(CONFIG["DEVICE"])

    # models + weights
    fusion = MultimodalTransformerEncoder()
    decoder = LidarDecoderPaper(latent_dim=1024,H=CONFIG["H"], W=CONFIG["W"],ups=CONFIG["UPS"], chs=(256,128,64,64,32),stride_h=CONFIG["STRIDE_H"], stride_w=CONFIG["STRIDE_W"],out_activation="relu")
    epoch = _load_checkpoint(CONFIG["CHECKPOINT"], fusion, decoder, device)
    print(f"[INFO] Loaded checkpoint (epoch={epoch})")

    # embedding
    emb = torch.load(CONFIG["EMBEDDING_PT"], map_location="cpu")
    if emb.ndim == 1:
        idx = 0
        emb = emb.unsqueeze(0)
    elif emb.ndim == 2:
        idx = int(CONFIG["EMB_INDEX"])
        if idx < 0 or idx >= emb.shape[0]:
            raise IndexError(f"emb_index {idx} out of range [0,{emb.shape[0]-1}]")
        emb = emb[idx:idx+1]
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(emb.shape)}")
    emb = emb.to(device)

    # lidar path
    if CONFIG["USE_AUTO_LIDAR"]:
        lidar_ply = _auto_lidar_path(CONFIG["EMBEDDING_PT"], idx)
    else:
        lidar_ply = CONFIG["LIDAR_PLY"]

    # forward + losses
    with torch.no_grad():
        gt_img, meta = pointcloud_to_range_with_meta(
            lidar_ply, H=CONFIG["H"], W=CONFIG["W"],
            r_clip=CONFIG["R_CLIP"], fill=CONFIG["FILL"]
        )  # (1,1,H,W), meta has el_min/el_max
        gt_img = gt_img.to(device)

        pred_img = decoder(fusion(emb))  # (1,1,H,W)

        mse = F.mse_loss(pred_img, gt_img).item()
        wloss = weighted_mse_ssim(
            pred_img, gt_img,
            alpha=CONFIG["ALPHA"], beta=CONFIG["BETA"],
            center_band=CONFIG["CENTER_BAND"], w_center=CONFIG["W_CENTER"]
        ).item()

    print("=== SINGLE SAMPLE EVAL ===")
    print(f"checkpoint      : {CONFIG['CHECKPOINT']}")
    print(f"embedding_pt    : {CONFIG['EMBEDDING_PT']}  (index={idx})")
    print(f"lidar_ply       : {lidar_ply}")
    print(f"device          : {device}")
    print(f"MSE             : {mse:.6f}")
    print(f"Weighted MSE+SSIM: {wloss:.6f}")

    # ----- Back-project both to 3D using the SAME elevation bounds (meta) -----
    gt_np   = gt_img.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_np = pred_img.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_np_clamped = np.clip(pred_np, 0.0, CONFIG["R_CLIP"])  # for geometry sanity

    el_min, el_max = meta["el_min"], meta["el_max"]
    gt_pts   = range_to_pointcloud(gt_np,   el_min, el_max, CONFIG["R_CLIP"])
    pred_pts = range_to_pointcloud(pred_np_clamped, el_min, el_max, CONFIG["R_CLIP"])
    print(f"[INFO] GT points: {gt_pts.shape[0]} | Pred points: {pred_pts.shape[0]}")

    # ----- Save predicted point cloud if requested -----
    if CONFIG["SAVE_PRED_PLY"]:
        save_points_as_ply(CONFIG["PRED_PLY_OUT"], pred_pts)
        print(f"[INFO] Saved predicted point cloud → {CONFIG['PRED_PLY_OUT']}")

    # ----- Display: range images and 3D scatter (matplotlib) -----
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Range images
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im0 = axes[0].imshow(gt_np, cmap="viridis", vmin=0.0, vmax=CONFIG["R_CLIP"])
    axes[0].set_title("GT Range (m)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_np_clamped, cmap="viridis", vmin=0.0, vmax=CONFIG["R_CLIP"])
    axes[1].set_title("Predicted Range (m)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3D point clouds (downsample for plotting speed if needed)
    ss = max(1, int(CONFIG["SUBSAMPLE_PC"]))
    gt_plot   = gt_pts[::ss]
    pred_plot = pred_pts[::ss]

    fig2 = plt.figure(figsize=(14, 6))
    ax1 = fig2.add_subplot(121, projection='3d')
    ax2 = fig2.add_subplot(122, projection='3d')

    ax1.scatter(gt_plot[:,0], gt_plot[:,1], gt_plot[:,2], s=0.1)
    ax1.set_title("GT Point Cloud")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")

    ax2.scatter(pred_plot[:,0], pred_plot[:,1], pred_plot[:,2], s=0.1)
    ax2.set_title("Predicted Point Cloud")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")

    # equal-ish axes
    for ax in (ax1, ax2):
        try:
            xyz = np.vstack([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).astype(float)
            ctr = xyz.mean(axis=1)
            rng = (xyz[:,1]-xyz[:,0]).max()/2
            ax.set_xlim(ctr[0]-rng, ctr[0]+rng)
            ax.set_ylim(ctr[1]-rng, ctr[1]+rng)
            ax.set_zlim(ctr[2]-rng, ctr[2]+rng)
        except Exception:
            pass

    plt.show()

if __name__ == "__main__":
    main()

