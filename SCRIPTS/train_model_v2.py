#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 10:45:57 2025

@author: nitaishah
"""
import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# --- deeper models (drop-in) ---
from models.fusion_encoder_paper_v2 import MultimodalTransformerEncoderV2 as FusionEnc
from models.lidar_decoder_paper_v2   import LidarDecoderPaperV2           as LidarDec

from utils.lidar_project import pointcloud_to_range
from losses import weighted_mse_ssim

# -------- version-safe AMP helpers (old/new PyTorch) --------
try:
    from torch.amp import autocast as _autocast_amp, GradScaler as _GradScaler_amp
    _AMP_BACKEND = "torch.amp"
except Exception:
    from torch.cuda.amp import autocast as _autocast_amp, GradScaler as _GradScaler_amp
    _AMP_BACKEND = "torch.cuda.amp"

def amp_autocast(dtype=torch.float16):
    return _autocast_amp("cuda", dtype=dtype) if _AMP_BACKEND == "torch.amp" else _autocast_amp(dtype=dtype)

def amp_scaler(enabled=True):
    return _GradScaler_amp("cuda", enabled=enabled) if _AMP_BACKEND == "torch.amp" else _GradScaler_amp(enabled=enabled)
# ------------------------------------------------------------

CONFIG = {
    "embedding_dir": "/scratch/user/nitaishah/GEN_AI/Phase_2/DATA/embeddings/s31/unit1",
    "lidar_dir":     "/scratch/user/nitaishah/GEN_AI/DATA_31/scenario31/unit1/lidar_data",
    "save_dir":      "/scratch/user/nitaishah/GEN_AI/Phase_2/OUTPUTS/runs/deeper_v2_min",
    "epochs":        8,
    "lr":            1e-4,
    "batch_size":    4,            # can be <= saved embedding batch (which is 8)
    "save_every":    2,
    "alpha":         0.8,
    "beta":          0.2,
    "center_band":   (220, 859),
    "w_center":      10.0,
    "use_batches":   876,          # number of embedding files to use
    "log_every":     50,
    "amp":           True,
    "enc": dict(num_modalities=4, d_token_flat=768, d_model=512, n_heads=8,
                depth=12, mlp_ratio=4.0, use_cls=True, proj_out=1024,
                drop=0.0, attn_drop=0.0, drop_path_rate=0.10),
    "dec": dict(latent_dim=1024, base_ch=128, use_attn_lowres=False),
}

def _discover_emb_batches(emb_dir: str) -> list[str]:
    files = [f for f in os.listdir(emb_dir) if f.endswith(".pt")]
    files.sort()
    return [os.path.join(emb_dir, f) for f in files]

def _lidar_paths_for_range(start_idx: int, count: int, base_dir: str) -> list[str]:
    # returns [lidar_data_{start_idx}.ply ... lidar_data_{start_idx+count-1}.ply]
    return [os.path.join(base_dir, f"lidar_data_{k:04d}.ply") for k in range(start_idx, start_idx + count)]

def _align_pred_target(pred: torch.Tensor, gt: torch.Tensor):
    # Ensure (B,1,H,W) and match spatial size
    if gt.ndim == 3: gt = gt.unsqueeze(1)
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    if pred.shape[0] != gt.shape[0]:
        raise RuntimeError(f"Batch mismatch: pred B={pred.shape[0]} vs gt B={gt.shape[0]}")
    return pred, gt

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    ckpt_dir = os.path.join(CONFIG["save_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    fusion  = FusionEnc(**CONFIG["enc"]).to(device)
    decoder = LidarDec(**CONFIG["dec"]).to(device)

    optimizer = optim.Adam(list(fusion.parameters()) + list(decoder.parameters()), lr=CONFIG["lr"])
    scaler = amp_scaler(enabled=(CONFIG["amp"] and device == "cuda"))

    emb_files_all = _discover_emb_batches(CONFIG["embedding_dir"])
    emb_files = emb_files_all[:CONFIG["use_batches"]]

    # Peek first file to know saved embedding batch size (likely 8)
    first_emb = torch.load(emb_files[0], map_location="cpu")
    embB_saved = int(first_emb.shape[0])  # e.g., 8
    del first_emb

    # We will process each emb file in chunks of size cfg_bsz
    cfg_bsz = int(CONFIG["batch_size"])
    chunks_per_file = math.ceil(embB_saved / cfg_bsz)
    total_chunks = len(emb_files) * chunks_per_file

    print(f"[INFO] Embedding batch saved as {embB_saved}. Running with batch_size={cfg_bsz}.")
    print(f"[INFO] Using {len(emb_files)} embedding files → {total_chunks} training chunks.", flush=True)

    global_chunk = 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        fusion.train()
        decoder.train()
        running = 0.0

        pbar = tqdm(emb_files, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for i_file, emb_path in enumerate(pbar):
            emb_full = torch.load(emb_path, map_location=device)  # (embB_saved, 3072)
            # Global base index for this file’s samples in the lidar sequence
            file_base = i_file * embB_saved

            # iterate over sub-batches: [0:cfg_bsz], [cfg_bsz:2*cfg_bsz], ...
            for s in range(0, embB_saved, cfg_bsz):
                emb = emb_full[s: s + cfg_bsz]                 # shape (chunk_b, 3072)
                chunk_b = int(emb.shape[0])                    # last chunk may be smaller

                # Build matching GT for the exact global indices
                start_idx = file_base + s
                lidar_paths = _lidar_paths_for_range(start_idx, chunk_b, CONFIG["lidar_dir"])
                with torch.no_grad():
                    gt_imgs = [pointcloud_to_range(p).to(device, non_blocking=True) for p in lidar_paths]
                    gt = torch.cat(gt_imgs, dim=0)            # (chunk_b, H, W) or (chunk_b,1,H,W)

                optimizer.zero_grad(set_to_none=True)

                if CONFIG["amp"] and device == "cuda":
                    with amp_autocast(dtype=torch.float16):
                        z = fusion(emb)
                        pred = decoder(z)
                        pred, gt = _align_pred_target(pred, gt)
                        loss = weighted_mse_ssim(
                            pred, gt,
                            alpha=CONFIG["alpha"], beta=CONFIG["beta"],
                            center_band=CONFIG["center_band"], w_center=CONFIG["w_center"]
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z = fusion(emb)
                    pred = decoder(z)
                    pred, gt = _align_pred_target(pred, gt)
                    loss = weighted_mse_ssim(
                        pred, gt,
                        alpha=CONFIG["alpha"], beta=CONFIG["beta"],
                        center_band=CONFIG["center_band"], w_center=CONFIG["w_center"]
                    )
                    loss.backward()
                    optimizer.step()

                running += float(loss)
                global_chunk += 1

                # periodic logging: first, every N, last
                if (global_chunk == 1) or (global_chunk % CONFIG["log_every"] == 0) or (global_chunk == total_chunks):
                    avg_so_far = running / global_chunk
                    print(f"[E{epoch:02d}] chunk {global_chunk}/{total_chunks} "
                          f"file={i_file+1}/{len(emb_files)} "
                          f"sub={s//cfg_bsz+1}/{chunks_per_file} "
                          f"curr_loss={float(loss):.6f} avg_loss={avg_so_far:.6f}",
                          flush=True)

                # cleanup
                del emb, z, gt, pred
                for t in gt_imgs: del t
                if device == "cuda":
                    torch.cuda.empty_cache()

            del emb_full  # free full batch tensor

        avg = running / total_chunks
        with open(os.path.join(CONFIG["save_dir"], "metrics.txt"), "a") as f:
            f.write(f"epoch={epoch}, loss={avg:.6f}\n")
        print(f"[E{epoch:02d}] epoch_avg_loss={avg:.6f}", flush=True)

        if epoch % CONFIG["save_every"] == 0:
            ck = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({
                "fusion": fusion.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": CONFIG,
                "embB_saved": embB_saved,
                "chunks_per_file": chunks_per_file,
            }, ck)
            print(f"[E{epoch:02d}] saved checkpoint: {ck}", flush=True)

if __name__ == "__main__":
    main()
