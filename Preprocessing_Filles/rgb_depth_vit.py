#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:16:32 2025

@author: nitaishah
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoImageProcessor, AutoModelForDepthEstimation,
    pipeline, ViTImageProcessor, ViTModel
)
import matplotlib.pyplot as plt

# --- tweak for quality/speed ---
MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"  # depth model
VIT_ID = "google/vit-base-patch16-224-in21k"            # ViT model (original paper)
MAX_SIZE = 512                                          # smaller for ViT input
CMAP = "gray"
PCTS = (1, 99)

# =========================================================
# 1️⃣ DEVICE + HELPERS
# =========================================================
def _pick_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available(): return "cuda:0", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps", torch.float16
    return "cpu", torch.float32

def _load_and_resize(img: Union[str, Path, Image.Image], max_side: int) -> Image.Image:
    if isinstance(img, (str, Path)): img = Image.open(img).convert("RGB")
    else: img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy(); img.thumbnail((max_side, max_side), Image.BICUBIC)
    return img

def _normalize01(arr: np.ndarray, pcts=(1, 99)) -> np.ndarray:
    lo, hi = np.percentile(arr, pcts)
    if hi <= lo: lo, hi = float(arr.min()), float(arr.max() + 1e-6)
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return x.astype(np.float32)

# =========================================================
# 2️⃣ LOAD MODELS
# =========================================================
_DEVICE, _DTYPE = _pick_device_and_dtype()

# --- depth model ---
_PROC_DEPTH = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
_MODEL_DEPTH = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=_DTYPE)
_PIPE_DEPTH = pipeline("depth-estimation", model=_MODEL_DEPTH, image_processor=_PROC_DEPTH, device=_DEVICE)

# --- vision transformer (for RGB + depth) ---
_VIT_PROCESSOR = ViTImageProcessor.from_pretrained(VIT_ID)
_VIT_MODEL = ViTModel.from_pretrained(VIT_ID).to(_DEVICE)

# =========================================================
# 3️⃣ INFERENCE FUNCTIONS
# =========================================================
def depth_anything_infer_gray(image_path: Union[str, Path]) -> np.ndarray:
    """Run depth estimation and return normalized grayscale depth."""
    rgb = _load_and_resize(image_path, MAX_SIZE)
    out = _PIPE_DEPTH(rgb)
    depth = np.asarray(out["depth"]).astype(np.float32)
    return _normalize01(depth, PCTS)

def vit_extract_features(img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """Return CLS embedding (768-dim for ViT-B/16)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8))  # convert 0–1 to 0–255
    inputs = _VIT_PROCESSOR(images=img, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        outputs = _VIT_MODEL(**inputs)
    # CLS token representation
    cls_emb = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden_dim]
    return cls_emb.squeeze(0).cpu().numpy()

# =========================================================
# 4️⃣ MAIN PIPELINE
# =========================================================
img_path = "/Volumes/One Touch/scenario31/unit1/camera_data/image_193.jpg"
rgb_img = _load_and_resize(img_path, MAX_SIZE)
depth01 = depth_anything_infer_gray(img_path)

# --- visualize depth ---
plt.figure(figsize=(10, 6))
plt.imshow(depth01, cmap=CMAP, vmin=0, vmax=1)
plt.axis("off")
plt.tight_layout()
plt.show()

# --- apply ViT ---
rgb_feat = vit_extract_features(rgb_img)
# Convert depth to 3 channels (gray → RGB for ViT)
depth_rgb = np.stack([depth01]*3, axis=-1)
depth_feat = vit_extract_features(depth_rgb)

print("RGB feature vector shape:", rgb_feat.shape)
print("Depth feature vector shape:", depth_feat.shape)
