#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:11:12 2025

@author: nitaishah
"""

# SCRIPTS/models/lidar_decoder_paper.py
import torch
import torch.nn as nn


class LidarDecoderPaper(nn.Module):
    """
    Paper-style LiDAR Decoder: 1024-D latent → (1, 1088, 1440) range image.

    Strategy:
    - Map 1024 → (C0=64, H0=34, W0=45)
    - Five ×2 upsampling stages: 34×45 → 68×90 → 136×180 → 272×360 → 544×720 → 1088×1440
    - Final 1-channel output (meters). No activation enforced here; loss handles scaling/normalization.
    """

    def __init__(self, latent_dim: int = 1024, base_ch: int = 64):
        super().__init__()
        self.h0, self.w0, self.c0 = 34, 45, base_ch

        # latent → seed feature map
        self.fc = nn.Linear(latent_dim, self.c0 * self.h0 * self.w0)

        # upsampling stack (ConvTranspose2d, stride=2)
        self.up = nn.Sequential(
            self._up(self.c0,   128),   # 34×45   → 68×90
            self._up(128,       128),   # 68×90   → 136×180
            self._up(128,        64),   # 136×180 → 272×360
            self._up(64,         32),   # 272×360 → 544×720
            self._up(32,         16),   # 544×720 → 1088×1440
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # (B,1,1088,1440)
        )

        # init
        self._init_weights()

    @staticmethod
    def _up(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _init_weights(self):
        # Kaiming init for convs; zeros for biases
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, 1024)
        returns: (B, 1, 1088, 1440)
        """
        B = z.size(0)
        x = self.fc(z).view(B, self.c0, self.h0, self.w0)
        x = self.up(x)
        return x
