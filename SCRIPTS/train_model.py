#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:26:16 2025

@author: nitaishah
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from models.fusion_encoder_paper import MultimodalTransformerEncoder
from models.lidar_decoder_paper import LidarDecoderPaper
from utils.lidar_project import pointcloud_to_range
from losses import weighted_mse_ssim

CONFIG = {
    "embedding_dir": "/scratch/user/nitaishah/GEN_AI/Phase_2/DATA/embeddings/s31/unit1",
    "lidar_dir":     "/scratch/user/nitaishah/GEN_AI/DATA_31/scenario31/unit1/lidar_data",
    "save_dir":      "/scratch/user/nitaishah/GEN_AI/Phase_2/OUTPUTS/runs/paper_baseline",
    "epochs":        40,
    "lr":            1e-4,
    "batch_size":    8,
    "save_every":    5,
    "alpha":         0.8,
    "beta":          0.2,
    "center_band":   (220, 859),
    "w_center":      10.0,
    "use_batches":   876,     # skip last partial
    "log_every":     50,      # <-- print every N batches
}

def _discover_emb_batches(emb_dir: str) -> list[str]:
    files = [f for f in os.listdir(emb_dir) if f.endswith(".pt")]
    files.sort()
    return [os.path.join(emb_dir, f) for f in files]

def _lidar_paths_for_batch(i_batch: int, base_dir: str, batch_size: int) -> list[str]:
    start = i_batch * batch_size
    end   = start + batch_size
    return [os.path.join(base_dir, f"lidar_data_{k:04d}.ply") for k in range(start, end)]

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    ckpt_dir = os.path.join(CONFIG["save_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fusion = MultimodalTransformerEncoder().to(device)
    decoder = LidarDecoderPaper().to(device)
    optimizer = optim.Adam(list(fusion.parameters()) + list(decoder.parameters()), lr=CONFIG["lr"])

    emb_files_all = _discover_emb_batches(CONFIG["embedding_dir"])
    emb_files = emb_files_all[:CONFIG["use_batches"]]
    num_batches = len(emb_files)
    print(f"[INFO] Using {num_batches} batches (size={CONFIG['batch_size']}).", flush=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        fusion.train()
        decoder.train()
        running = 0.0

        for i, emb_path in enumerate(tqdm(emb_files, desc=f"Epoch {epoch}/{CONFIG['epochs']}")):
            emb = torch.load(emb_path, map_location=device)  # (8, 3072)

            lidar_paths = _lidar_paths_for_batch(i, CONFIG["lidar_dir"], CONFIG["batch_size"])
            gt_imgs = [pointcloud_to_range(p).to(device) for p in lidar_paths]
            gt = torch.cat(gt_imgs, dim=0)  # (8,1,H,W)

            pred = decoder(fusion(emb))
            loss = weighted_mse_ssim(
                pred, gt,
                alpha=CONFIG["alpha"], beta=CONFIG["beta"],
                center_band=CONFIG["center_band"], w_center=CONFIG["w_center"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss)

            # --- periodic per-batch logging ---
            if (i + 1) % CONFIG["log_every"] == 0 or i == 0 or (i + 1) == num_batches:
                avg_so_far = running / (i + 1)
                print(
                    f"[E{epoch:02d}] batch {i+1}/{num_batches} "
                    f"curr_loss={float(loss):.6f} avg_loss={avg_so_far:.6f}",
                    flush=True
                )

            del emb, gt, pred
            for t in gt_imgs: del t
            if device == "cuda":
                torch.cuda.empty_cache()

        avg = running / num_batches
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
            }, ck)
            print(f"[E{epoch:02d}] saved checkpoint: {ck}", flush=True)

if __name__ == "__main__":
    main()
