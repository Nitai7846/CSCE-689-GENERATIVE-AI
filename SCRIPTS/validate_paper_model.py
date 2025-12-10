#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:26:56 2025

@author: nitaishah
"""

import os
import torch
from tqdm import tqdm

from models.fusion_encoder_paper import MultimodalTransformerEncoder
from models.lidar_decoder_paper import LidarDecoderPaper
from utils.lidar_project import pointcloud_to_range

def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()

CONFIG = {
    "embedding_dir": "/scratch/user/nitaishah/GEN_AI/Phase_2/DATA/embeddings/s31/unit1",
    "lidar_dir":     "/scratch/user/nitaishah/GEN_AI/Phase_2/DATA/raw/lidar_ply",
    "ckpt_path":     "/scratch/user/nitaishah/GEN_AI/Phase_2/OUTPUTS/runs/paper_baseline/checkpoints/epoch_40.pt",
    "log_path":      "/scratch/user/nitaishah/GEN_AI/Phase_2/OUTPUTS/runs/paper_baseline/val_metrics.txt",
    "batch_eval_step": 10,   # evaluate every Nth batch to keep it light
    "lidar_index_start": 1000,
    "batch_size": 8,
}

def _lidar_paths_for_batch(i_batch: int) -> list[str]:
    start = CONFIG["lidar_index_start"] + i_batch * CONFIG["batch_size"]
    return [os.path.join(CONFIG["lidar_dir"], f"lidar_data_{k}.ply")
            for k in range(start, start + CONFIG["batch_size"])]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fusion = MultimodalTransformerEncoder().to(device)
    decoder = LidarDecoderPaper().to(device)

    ck = torch.load(CONFIG["ckpt_path"], map_location=device)
    fusion.load_state_dict(ck["fusion"])
    decoder.load_state_dict(ck["decoder"])
    fusion.eval(); decoder.eval()

    emb_files = sorted([f for f in os.listdir(CONFIG["embedding_dir"]) if f.endswith(".pt")])

    metrics = []
    with torch.no_grad():
        for i, fname in enumerate(tqdm(emb_files, desc="Validate")):
            if i % CONFIG["batch_eval_step"] != 0:
                continue
            emb = torch.load(os.path.join(CONFIG["embedding_dir"], fname), map_location=device)
            lidar_paths = _lidar_paths_for_batch(i)
            gt_imgs = [pointcloud_to_range(p).to(device) for p in lidar_paths]
            gt = torch.cat(gt_imgs, dim=0)  # (B,1,H,W)

            pred = decoder(fusion(emb))
            metrics.append(rmse(pred, gt))

            del emb, gt, pred
            for t in gt_imgs: del t

    avg_rmse = float(sum(metrics) / max(1, len(metrics)))
    os.makedirs(os.path.dirname(CONFIG["log_path"]), exist_ok=True)
    with open(CONFIG["log_path"], "a") as f:
        f.write(f"Avg RMSE over sampled batches: {avg_rmse:.6f}\n")

if __name__ == "__main__":
    main()
