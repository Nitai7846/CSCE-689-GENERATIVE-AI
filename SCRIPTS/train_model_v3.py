#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Third Variant Training Script (v3)
- Encoder: PerceiverFusionEncoderV3
- Decoder: LidarDecoderV3 (flexible strides for tall-narrow outputs)
- Loss: weighted MSE + SSIM combo (your existing losses.weighted_mse_ssim)

Assumes:
- Embedding batches: N files, each torch tensor [8, 3072]
- LiDAR pointclouds: ply files; pointcloud_to_range(...) -> [1,H,W] torch.FloatTensor
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from models.fusion_encoder_paper_v3 import PerceiverFusionEncoderV3
from models.lidar_decoder_paper_v3 import LidarDecoderV3
from utils.lidar_project import pointcloud_to_range
from losses import weighted_mse_ssim

CONFIG = {
    # --- paths (edit to your actual) ---
    "embedding_dir": "/scratch/user/dhruvpatel144/genai/embeddings/s31/unit1",
    "lidar_dir":     "/scratch/user/dhruvpatel144/genai/data/scenario31/unit1/lidar_data_renamed",
    "save_dir":      "/scratch/user/dhruvpatel144/genai/data/OUTPUTS/runs/model_v3",

    # --- training ---
    "epochs":        12,
    "lr":            1e-4,
    "batch_size":    8,
    "save_every":    5,
    "log_every":     50,

    # --- loss weighting (same API as your existing) ---
    "alpha":         0.8,
    "beta":          0.2,
    "center_band":   (220, 859),
    "w_center":      10.0,

    # --- data iteration ---
    "use_batches":   876,      # cap/skip last partial if needed

    # --- output image size/strides (edit if using 1440x1088) ---
    "H":             1024,
    "W":             3,
    "ups":           5,
    "stride_h":      2,
    "stride_w":      1,
}

def _discover_emb_batches(emb_dir: str):
    files = [f for f in os.listdir(emb_dir) if f.endswith(".pt")]
    files.sort()
    return [os.path.join(emb_dir, f) for f in files]

def _lidar_paths_for_batch(i_batch: int, base_dir: str, batch_size: int):
    start = i_batch * batch_size
    end   = start + batch_size
    return [os.path.join(base_dir, f"lidar_data_{k:04d}.ply") for k in range(start, end)]

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    ckpt_dir = os.path.join(CONFIG["save_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Model (v3) -----
    fusion = PerceiverFusionEncoderV3(
        num_modalities=4, d_token=768, d_latent=768,
        n_latents=6, num_heads=12, ffn_dim=2048,
        dropout=0.1, num_blocks=2, out_dim=1024
    ).to(device)

    decoder = LidarDecoderV3(
        latent_dim=1024, H=CONFIG["H"], W=CONFIG["W"],
        ups=CONFIG["ups"], chs=(256,128,64,64,32),
        stride_h=CONFIG["stride_h"], stride_w=CONFIG["stride_w"],
        out_activation="relu"
    ).to(device)

    optimizer = optim.Adam(list(fusion.parameters()) + list(decoder.parameters()), lr=CONFIG["lr"])

    emb_files_all = _discover_emb_batches(CONFIG["embedding_dir"])
    emb_files = emb_files_all[:CONFIG["use_batches"]]
    num_batches = len(emb_files)
    print(f"[INFO] Using {num_batches} batches (size={CONFIG['batch_size']}).", flush=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        fusion.train(); decoder.train()
        running = 0.0

        for i, emb_path in enumerate(tqdm(emb_files, desc=f"Epoch {epoch}/{CONFIG['epochs']}")):
            emb = torch.load(emb_path, map_location=device)  # [8, 3072]

            lidar_paths = _lidar_paths_for_batch(i, CONFIG["lidar_dir"], CONFIG["batch_size"])
            gt_imgs = [pointcloud_to_range(p, H=CONFIG["H"], W=CONFIG["W"]).to(device) for p in lidar_paths]
            gt = torch.cat(gt_imgs, dim=0)  # [8,1,H,W]

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

            if (i + 1) % CONFIG["log_every"] == 0 or i == 0 or (i + 1) == num_batches:
                avg_so_far = running / (i + 1)
                print(f"[E{epoch:02d}] batch {i+1}/{num_batches} curr={float(loss):.6f} avg={avg_so_far:.6f}", flush=True)

            # cleanup
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
                "config": CONFIG
            }, ck)
            print(f"[E{epoch:02d}] saved checkpoint: {ck}", flush=True)

if __name__ == "__main__":
    main()
