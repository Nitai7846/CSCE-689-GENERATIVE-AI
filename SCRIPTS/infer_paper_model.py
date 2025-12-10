#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:27:29 2025

@author: nitaishah
"""

import os
import torch
from tqdm import tqdm

from models.fusion_encoder_paper import MultimodalTransformerEncoder
from models.lidar_decoder_paper import LidarDecoderPaper

CONFIG = {
    "embedding_dir": "/scratch/user/nitaishah/GEN_AI/Phase_2/DATA/embeddings/s31/unit1",
    "ckpt_path":     "/scratch/user/nitaishah/GEN_AI/Phase_2/OUTPUTS/runs/paper_baseline/checkpoints/epoch_40.pt",
    "save_dir":      "/scratch/user/nitaishah/GEN_AI/Phase_2/OUTPUTS/runs/paper_baseline/preds",
    "num_batches":   10,   # how many batches to export predictions for
}

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fusion = MultimodalTransformerEncoder().to(device)
    decoder = LidarDecoderPaper().to(device)

    ck = torch.load(CONFIG["ckpt_path"], map_location=device)
    fusion.load_state_dict(ck["fusion"])
    decoder.load_state_dict(ck["decoder"])
    fusion.eval(); decoder.eval()

    emb_files = sorted([f for f in os.listdir(CONFIG["embedding_dir"]) if f.endswith(".pt")])[:CONFIG["num_batches"]]

    with torch.no_grad():
        for fname in tqdm(emb_files, desc="Infer"):
            emb = torch.load(os.path.join(CONFIG["embedding_dir"], fname), map_location=device)
            pred = decoder(fusion(emb))
            out_path = os.path.join(CONFIG["save_dir"], fname.replace(".pt", "_pred.pt"))
            torch.save(pred.cpu(), out_path)

            del emb, pred

if __name__ == "__main__":
    main()
