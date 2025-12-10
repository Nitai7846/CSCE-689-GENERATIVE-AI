#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:25:57 2025

@author: nitaishah
"""

import torch
import torch.nn.functional as F


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, center_band=(220, 859), w_center=10.0):
    """
    Paper-style weighted MSE emphasizing central elevation rows.
    pred/target: (B,1,H,W)
    """
    assert pred.shape == target.shape, "pred/target must have same shape"
    B, C, H, W = pred.shape
    w = torch.ones((H,), device=pred.device, dtype=pred.dtype)
    y0, y1 = center_band
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H - 1, y1))
    if y1 > y0:
        w[y0:y1] = w_center

    w = w.view(1, 1, H, 1)
    loss = ((pred - target) ** 2 * w).mean()
    return loss


def _ssim_single(x, y, C1=0.01 ** 2, C2=0.03 ** 2, win_size=11):
    """
    Single-scale SSIM (grayscale), computed per-batch, channel-wise mean.
    x,y: (B,1,H,W) in [0,1] (recommend normalizing)
    """
    # simple Gaussian-like box filter via avg_pool (fast, no deps)
    pad = win_size // 2
    mu_x = F.avg_pool2d(x, win_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, win_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(x * x, win_size, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, win_size, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, win_size, 1, pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_y + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean()


def weighted_mse_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
    beta: float = 0.2,
    center_band=(220, 859),
    w_center=10.0,
):
    """
    Composite objective from the paper summary:
      L = alpha * weighted_MSE + beta * (1 - SSIM)
    """
    # normalize to [0,1] for SSIM stability (use per-batch min/max)
    with torch.no_grad():
        p_min, p_max = pred.amin(dim=(2, 3), keepdim=True), pred.amax(dim=(2, 3), keepdim=True)
        t_min, t_max = target.amin(dim=(2, 3), keepdim=True), target.amax(dim=(2, 3), keepdim=True)
        pred_n = (pred - p_min) / (p_max - p_min + 1e-6)
        targ_n = (target - t_min) / (t_max - t_min + 1e-6)

    wmse = weighted_mse(pred, target, center_band=center_band, w_center=w_center)
    ssim_val = _ssim_single(pred_n, targ_n)
    return alpha * wmse + beta * (1.0 - ssim_val)
