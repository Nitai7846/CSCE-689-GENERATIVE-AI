#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def _base_hw(H: int, W: int, ups: int, stride_h: int, stride_w: int):
    """Compute base (H0,W0) so that H0*(stride_h**ups)=H and W0*(stride_w**ups)=W."""
    if stride_h == 1:
        H0 = H
    else:
        sfh = stride_h ** ups
        assert H % sfh == 0, f"H={H} must be divisible by stride_h**ups={sfh}"
        H0 = H // sfh
    if stride_w == 1:
        W0 = W
    else:
        sfw = stride_w ** ups
        assert W % sfw == 0, f"W={W} must be divisible by stride_w**ups={sfw}"
        W0 = W // sfw
    return H0, W0

class LidarDecoderV3(nn.Module):
    """
    Flexible LiDAR decoder (v3)
    ---------------------------
    - latent(1024) -> FC -> [B, C0, H0, W0]
    - `ups` stages of ConvTranspose2d with stride=(stride_h, stride_w)
    - BN+ReLU after all but final stage
    Defaults mirror paper behavior (5 stages; 2x2 stride). For tall-narrow
    outputs (e.g., 1024x3), set stride_w=1 to keep width fixed.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        H: int = 1440,
        W: int = 1088,
        ups: int = 5,
        chs=(256, 128, 64, 64, 32),
        stride_h: int = 2,
        stride_w: int = 2,
        out_activation: str = "relu"
    ):
        super().__init__()
        assert len(chs) == ups, "len(chs) must equal number of upsampling stages"
        self.H, self.W = H, W
        self.ups = ups
        self.stride_h = stride_h
        self.stride_w = stride_w

        H0, W0 = _base_hw(H, W, ups, stride_h, stride_w)
        self.H0, self.W0 = H0, W0
        C0 = chs[0]

        self.fc = nn.Linear(latent_dim, C0 * H0 * W0)

        blocks = []
        in_c = C0
        for i in range(ups):
            out_c = chs[i+1] if i + 1 < len(chs) else 1
            is_last = (i == ups - 1)
            k_h = 4 if stride_h > 1 else 3
            k_w = 4 if stride_w > 1 else 3
            blocks.append(nn.ConvTranspose2d(
                in_c, out_c,
                kernel_size=(k_h, k_w),
                stride=(stride_h, stride_w),
                padding=(1, 1)
            ))
            if not is_last:
                blocks.append(nn.BatchNorm2d(out_c))
                blocks.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.deconv = nn.Sequential(*blocks)

        if out_activation == "relu":
            self.out_act = nn.ReLU()
        elif out_activation == "none":
            self.out_act = nn.Identity()
        else:
            raise ValueError("out_activation must be 'relu' or 'none'")

        # Init (DCGAN-style + BN gamma)
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x = self.fc(z).view(B, -1, self.H0, self.W0)  # [B,C0,H0,W0]
        x = self.deconv(x)                             # [B,1,H,W]
        return self.out_act(x)
