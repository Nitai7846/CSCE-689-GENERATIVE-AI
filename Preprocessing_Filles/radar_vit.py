#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:22:46 2025

@author: nitaishah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 9 2025
Applies Vision Transformer (ViT) to DeepSense radar Range–Velocity & Range–Angle maps
@author: nitaishah
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

# -------------------------------------------------------
# Helper: normalize between 0–1
# -------------------------------------------------------
def minmax(arr):
    arr = np.real(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

# -------------------------------------------------------
# Compute Range–Velocity and Range–Angle maps
# -------------------------------------------------------
def compute_deepsense_maps(radar_data, radar_params,
                           min_range=5, max_range=15,
                           fft_size_angle=64, plot=True):
    C = 3e8
    N_rx = radar_params['rx']
    N_samples = radar_params['samples']
    N_chirps = radar_params['chirps']
    f_s = radar_params['adc_sampling']
    slope = radar_params['chirp_slope']
    f_c = radar_params['start_freq']
    idle = radar_params['idle_time'] * 1e-6
    ramp = radar_params['ramp_end_time'] * 1e-6
    T_c = idle + ramp

    # Derived resolutions
    RANGE_RES = (C * f_s) / (2 * N_samples * slope)
    VEL_RES_KMPH = 3.6 * C / (2 * f_c * T_c * N_chirps)

    acquired_range = N_samples * RANGE_RES
    first_idx = int(np.ceil(N_samples * min_range / acquired_range))
    last_idx = int(np.ceil(N_samples * max_range / acquired_range))
    range_axis = np.linspace(0, acquired_range, N_samples)
    cropped_range = range_axis[first_idx:last_idx]

    vel_extent = VEL_RES_KMPH * N_chirps / 2
    vel_axis = np.linspace(-vel_extent, vel_extent, N_chirps)
    ang_lim = 75
    ang_axis = np.linspace(-ang_lim, ang_lim, fft_size_angle)

    # ----- Range–Velocity -----
    data_rv = np.fft.fft(radar_data, axis=1)
    data_rv = np.fft.fft(data_rv, axis=2)
    data_rv = np.fft.fftshift(data_rv, axes=2)
    data_rv = np.abs(data_rv).sum(axis=0)
    data_rv = np.log1p(data_rv)
    rv_map = minmax(data_rv)[first_idx:last_idx, :]

    # ----- Range–Angle -----
    data_ra = np.fft.fft(radar_data, axis=1)
    data_ra -= np.mean(data_ra, axis=2, keepdims=True)
    data_ra = np.fft.fft(data_ra, fft_size_angle, axis=0)
    data_ra = np.fft.fftshift(data_ra, axes=0)
    data_ra = np.abs(data_ra).sum(axis=2)
    ra_map = minmax(data_ra.T)[first_idx:last_idx, :]

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        im0 = axs[0].imshow(ra_map, aspect='auto', cmap='seismic', origin='lower',
                            extent=[-ang_lim, +ang_lim, cropped_range[0], cropped_range[-1]])
        axs[0].set_xlabel('Angle (°)')
        axs[0].set_ylabel('Range (m)')
        axs[0].set_title('Range–Angle Map')
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(rv_map, aspect='auto', cmap='seismic', origin='lower',
                            extent=[-vel_extent, +vel_extent, cropped_range[0], cropped_range[-1]])
        axs[1].set_xlabel('Velocity (km/h)')
        axs[1].set_ylabel('Range (m)')
        axs[1].set_title('Range–Velocity Map')
        plt.colorbar(im1, ax=axs[1])
        plt.show()

    axes = {"range": cropped_range, "velocity": vel_axis, "angle": ang_axis}
    return rv_map, ra_map, axes

# -------------------------------------------------------
# Vision Transformer Feature Extractor
# -------------------------------------------------------
def extract_vit_features(image_array, vit_processor, vit_model, device="cpu"):
    """Converts 2D array to 3-channel RGB image, extracts ViT [CLS] embedding."""
    # Normalize and convert to RGB image
    img = (minmax(image_array) * 255).astype(np.uint8)
    img_rgb = np.stack([img] * 3, axis=-1)
    img_pil = Image.fromarray(img_rgb)

    # Preprocess and infer
    inputs = vit_processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls_emb

# -------------------------------------------------------
# Main Script
# -------------------------------------------------------
if __name__ == "__main__":
    radar_path = "/Volumes/One Touch/scenario31/unit1/radar_data/radar_data_193.npy"
    radar_data = np.load(radar_path)  # shape (4, 256, 128)
    print("Radar data shape:", radar_data.shape)

    RADAR_PARAMS = {
        'chirps': 128,
        'tx': 1,
        'rx': 4,
        'samples': 256,
        'adc_sampling': 5e6,
        'chirp_slope': 15.015e12,
        'start_freq': 77e9,
        'idle_time': 5,
        'ramp_end_time': 60
    }

    # --- Compute Radar Maps ---
    rv_map, ra_map, axes = compute_deepsense_maps(radar_data, RADAR_PARAMS,
                                                  min_range=5, max_range=15,
                                                  fft_size_angle=64, plot=True)

    # --- Load Vision Transformer ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)

    # --- Extract ViT Features ---
    rv_feat = extract_vit_features(rv_map, vit_processor, vit_model, device)
    ra_feat = extract_vit_features(ra_map, vit_processor, vit_model, device)

    print(f"RV feature shape: {rv_feat.shape}")
    print(f"RA feature shape: {ra_feat.shape}")

    # --- Optional: Feature Fusion Example ---
    fused_feat = np.concatenate([rv_feat, ra_feat])
    print(f"Fused radar embedding: {fused_feat.shape}")
