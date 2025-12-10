#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 15:19:51 2025

@author: nitaishah
"""

import os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from importlib.machinery import SourceFileLoader

# ===================== EDIT THESE PATHS/VALUES =====================
CONFIG = {
    "PROJECT_ROOT": "/Volumes/One Touch/GEN_AI_PROJECT/phase_2/SCRIPTS",
    "FUSION_FILE":  "models/fusion_encoder_paper_v2.py",   # V2 code
    "DECODER_FILE": "models/lidar_decoder_paper_v2.py",    # V2 code
    "LOSSES_FILE":  "losses.py",

    "CHECKPOINT":   "/Volumes/One Touch/GEN_AI_PROJECT/phase_2/WEIGHTS/epoch_8.pt",
    "EMBEDDING_PT": "/Volumes/One Touch/GEN_AI_PROJECT/phase_2/Data/dhruv/embeddings/s31/unit1/batch_0865_embeddings.pt",
    "EMB_INDEX":    0,

    "USE_AUTO_LIDAR": False,
    "LIDAR_DIR": "/scratch/user/nitaishah/GEN_AI/DATA_31/scenario31/unit1/lidar_data",
    "LIDAR_PLY": "/Volumes/One Touch/scenario31/unit1/lidar_data/lidar_data_6920.ply",
    "PREPROC_BATCH_SIZE": 8,

    "ALPHA": 0.8, "BETA": 0.2, "CENTER_BAND": (220, 859), "W_CENTER": 10.0,

    "H": 1088, "W": 1440, "R_CLIP": 120.0, "FILL": 0.0,

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "AMP": True,

    "SUBSAMPLE_PC": 1,
    "SAVE_PRED_PLY": True,
    "PRED_PLY_OUT": "/Volumes/One Touch/GEN_AI_PROJECT/phase_2/OUTPUTS/infer_preview/predicted_pointcloud_v2.ply",
}
# ===================================================================

# Put project on sys.path
if CONFIG["PROJECT_ROOT"] not in sys.path:
    sys.path.insert(0, CONFIG["PROJECT_ROOT"])

def _load_symbol(path, symbol):
    abspath = os.path.join(CONFIG["PROJECT_ROOT"], path)
    modname = f"_dyn_{os.path.basename(path).replace('.', '_')}"
    mod = SourceFileLoader(modname, abspath).load_module()
    return getattr(mod, symbol)

MultimodalTransformerEncoderV2 = _load_symbol(CONFIG["FUSION_FILE"],  "MultimodalTransformerEncoderV2")
LidarDecoderPaperV2           = _load_symbol(CONFIG["DECODER_FILE"], "LidarDecoderPaperV2")
weighted_mse_ssim             = _load_symbol(CONFIG["LOSSES_FILE"],   "weighted_mse_ssim")

# -------- version-safe AMP (old/new PyTorch) --------
try:
    from torch.amp import autocast as _autocast_amp
    _AMP_BACKEND = "torch.amp"
except Exception:
    from torch.cuda.amp import autocast as _autocast_amp
    _AMP_BACKEND = "torch.cuda.amp"

def amp_autocast(dtype=torch.float16):
    return _autocast_amp("cuda", dtype=dtype) if _AMP_BACKEND == "torch.amp" else _autocast_amp(dtype=dtype)
# ---------------------------------------------------

# ---- ply I/O ----
from plyfile import PlyData, PlyElement

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
    if "vertex" not in ply: raise ValueError(f"No 'vertex' in {ply_path}")
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

# ---- projection with meta ----
def pointcloud_to_range_with_meta(ply_path: str, H: int, W: int, r_clip: float, fill: float):
    pts = _read_points_from_ply(ply_path)
    if pts.shape[0] == 0:
        img = np.full((H, W), fill, dtype=np.float32)
        return torch.from_numpy(img)[None, None], {"el_min": -np.pi/4, "el_max": np.pi/4}

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    r = np.sqrt(x*x + y*y + z*z).astype(np.float32)
    r = np.clip(r, 1e-6, r_clip)

    az = np.arctan2(y, x)                                # (-pi,pi]
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))            # [-pi/2, pi/2]

    az_idx = ((az + np.pi) / (2 * np.pi) * W).astype(np.int32)
    az_idx = np.clip(az_idx, 0, W - 1)

    el_min, el_max = el.min(), el.max()
    if not np.isfinite(el_min) or not np.isfinite(el_max) or el_max <= el_min:
        el_min, el_max = -np.pi/4, np.pi/4
    el_idx = ((el - el_min) / (el_max - el_min + 1e-6) * H).astype(np.int32)
    el_idx = np.clip(el_idx, 0, H - 1)

    img = np.full((H, W), np.inf, dtype=np.float32)
    lin = el_idx * W + az_idx
    np.minimum.at(img.ravel(), lin, r)
    img[~np.isfinite(img)] = np.float32(fill)

    meta = {"el_min": float(el_min), "el_max": float(el_max)}
    return torch.from_numpy(img)[None, None], meta

# ---- back-projection ----
def range_to_pointcloud(range_img: np.ndarray, el_min: float, el_max: float, r_clip: float) -> np.ndarray:
    H, W = range_img.shape
    az = (np.arange(W, dtype=np.float32) + 0.5) / W * (2*np.pi) - np.pi
    el = (np.arange(H, dtype=np.float32) + 0.5) / H * (el_max - el_min) + el_min
    az_grid, el_grid = np.meshgrid(az, el)
    r = np.clip(range_img.astype(np.float32), 0.0, r_clip)
    mask = r > 0.0
    if not np.any(mask): return np.zeros((0,3), dtype=np.float32)
    r_sel, az_sel, el_sel = r[mask], az_grid[mask], el_grid[mask]
    cos_el = np.cos(el_sel)
    x = r_sel * cos_el * np.cos(az_sel)
    y = r_sel * cos_el * np.sin(az_sel)
    z = r_sel * np.sin(el_sel)
    return np.stack([x, y, z], axis=1).astype(np.float32)

# ---- helpers ----
def _extract_batch_idx(embedding_pt_path: str) -> int:
    m = re.search(r"batch_(\d+)_embeddings\.pt$", os.path.basename(embedding_pt_path))
    if not m: raise ValueError(f"Cannot parse batch index: {embedding_pt_path}")
    return int(m.group(1))

def _auto_lidar_path(embedding_pt_path: str, emb_index: int) -> str:
    b = _extract_batch_idx(embedding_pt_path)
    global_idx = b * CONFIG["PREPROC_BATCH_SIZE"] + emb_index
    return os.path.join(CONFIG["LIDAR_DIR"], f"lidar_data_{global_idx:04d}.ply")

def _build_models_from_ckpt_config(ckpt, device):
    # Defaults in case config missing
    enc_kwargs = dict(num_modalities=4, d_token_flat=768, d_model=512, n_heads=8,
                      depth=12, mlp_ratio=4.0, use_cls=True, proj_out=1024,
                      drop=0.0, attn_drop=0.0, drop_path_rate=0.10)
    dec_kwargs = dict(latent_dim=1024, base_ch=128, use_attn_lowres=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            enc_kwargs.update(cfg.get("enc", {}))
            dec_kwargs.update(cfg.get("dec", {}))

    fusion  = MultimodalTransformerEncoderV2(**enc_kwargs).to(device).eval()
    decoder = LidarDecoderPaperV2(**dec_kwargs).to(device).eval()
    return fusion, decoder, enc_kwargs, dec_kwargs

def _load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fusion, decoder, enc_kwargs, dec_kwargs = _build_models_from_ckpt_config(ckpt, device)

    # Prefer new format
    if isinstance(ckpt, dict) and "fusion" in ckpt and "decoder" in ckpt:
        fusion.load_state_dict(ckpt["fusion"], strict=True)
        decoder.load_state_dict(ckpt["decoder"], strict=True)
        ep = ckpt.get("epoch", None)
    else:
        # Fallback: older flat keys
        f_sd = {k.split("fusion.", 1)[-1]: v for k, v in ckpt.items() if k.startswith("fusion.")}
        d_sd = {k.split("decoder.", 1)[-1]: v for k, v in ckpt.items() if k.startswith("decoder.")}
        # try strict first; if mismatch, do non-strict with warning
        try:
            if f_sd: fusion.load_state_dict(f_sd, strict=True)
            if d_sd: decoder.load_state_dict(d_sd, strict=True)
        except Exception as e:
            print(f"[WARN] strict load failed ({e}); trying non-strict ...")
            if f_sd: fusion.load_state_dict(f_sd, strict=False)
            if d_sd: decoder.load_state_dict(d_sd, strict=False)
        ep = None
    return fusion, decoder, ep

def _align_pred_target(pred: torch.Tensor, gt: torch.Tensor):
    if gt.ndim == 3: gt = gt.unsqueeze(1)
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    return pred, gt

# =========================== main ===========================
def main():
    device = torch.device(CONFIG["DEVICE"])
    torch.backends.cudnn.benchmark = True

    # models + weights from ckpt config
    fusion, decoder, epoch = _load_checkpoint(CONFIG["CHECKPOINT"], device)
    print(f"[INFO] Loaded checkpoint (epoch={epoch})")

    # embedding
    emb = torch.load(CONFIG["EMBEDDING_PT"], map_location="cpu")
    if emb.ndim == 1:
        idx = 0; emb = emb.unsqueeze(0)
    elif emb.ndim == 2:
        idx = int(CONFIG["EMB_INDEX"])
        if idx < 0 or idx >= emb.shape[0]:
            raise IndexError(f"emb_index {idx} out of range [0,{emb.shape[0]-1}]")
        emb = emb[idx:idx+1]
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(emb.shape)}")
    emb = emb.to(device)

    # lidar path
    lidar_ply = _auto_lidar_path(CONFIG["EMBEDDING_PT"], idx) if CONFIG["USE_AUTO_LIDAR"] else CONFIG["LIDAR_PLY"]

    # forward + losses
    with torch.no_grad():
        gt_img, meta = pointcloud_to_range_with_meta(
            lidar_ply, H=CONFIG["H"], W=CONFIG["W"],
            r_clip=CONFIG["R_CLIP"], fill=CONFIG["FILL"]
        )  # (1,1,H,W)
        gt_img = gt_img.to(device)

        if CONFIG["AMP"] and device.type == "cuda":
            with amp_autocast(dtype=torch.float16):
                pred_img = decoder(fusion(emb))
        else:
            pred_img = decoder(fusion(emb))

        pred_img, gt_img = _align_pred_target(pred_img, gt_img)

        mse = F.mse_loss(pred_img, gt_img).item()
        wloss = weighted_mse_ssim(
            pred_img, gt_img,
            alpha=CONFIG["ALPHA"], beta=CONFIG["BETA"],
            center_band=CONFIG["CENTER_BAND"], w_center=CONFIG["W_CENTER"]
        ).item()

    print("=== SINGLE SAMPLE EVAL (V2) ===")
    print(f"checkpoint      : {CONFIG['CHECKPOINT']}")
    print(f"embedding_pt    : {CONFIG['EMBEDDING_PT']}  (index={idx})")
    print(f"lidar_ply       : {lidar_ply}")
    print(f"device          : {device}")
    print(f"MSE             : {mse:.6f}")
    print(f"Weighted MSE+SSIM: {wloss:.6f}")

    # Back-project to 3D using same elevation bounds
    gt_np   = gt_img.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_np = pred_img.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_np_clamped = np.clip(pred_np, 0.0, CONFIG["R_CLIP"])
    el_min, el_max = meta["el_min"], meta["el_max"]
    gt_pts   = range_to_pointcloud(gt_np,   el_min, el_max, CONFIG["R_CLIP"])
    pred_pts = range_to_pointcloud(pred_np_clamped, el_min, el_max, CONFIG["R_CLIP"])
    print(f"[INFO] GT points: {gt_pts.shape[0]} | Pred points: {pred_pts.shape[0]}")

    if CONFIG["SAVE_PRED_PLY"]:
        save_points_as_ply(CONFIG["PRED_PLY_OUT"], pred_pts)
        print(f"[INFO] Saved predicted point cloud → {CONFIG['PRED_PLY_OUT']}")

    # Visuals
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im0 = axes[0].imshow(gt_np, cmap="viridis", vmin=0.0, vmax=CONFIG["R_CLIP"])
    axes[0].set_title("GT Range (m)"); axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(pred_np_clamped, cmap="viridis", vmin=0.0, vmax=CONFIG["R_CLIP"])
    axes[1].set_title("Predicted Range (m)"); axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    ss = max(1, int(CONFIG["SUBSAMPLE_PC"]))
    gt_plot, pred_plot = gt_pts[::ss], pred_pts[::ss]
    fig2 = plt.figure(figsize=(14, 6))
    ax1 = fig2.add_subplot(121, projection='3d')
    ax2 = fig2.add_subplot(122, projection='3d')
    ax1.scatter(gt_plot[:,0], gt_plot[:,1], gt_plot[:,2], s=0.1)
    ax1.set_title("GT Point Cloud")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
    ax2.scatter(pred_plot[:,0], pred_plot[:,1], pred_plot[:,2], s=0.1)
    ax2.set_title("Predicted Point Cloud")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
    for ax in (ax1, ax2):
        try:
            xyz = np.vstack([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).astype(float)
            ctr = xyz.mean(axis=1); rng = (xyz[:,1]-xyz[:,0]).max()/2
            ax.set_xlim(ctr[0]-rng, ctr[0]+rng)
            ax.set_ylim(ctr[1]-rng, ctr[1]+rng)
            ax.set_zlim(ctr[2]-rng, ctr[2]+rng)
        except Exception:
            pass
    plt.show()

if __name__ == "__main__":
    main()
