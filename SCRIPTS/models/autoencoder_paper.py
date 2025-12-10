#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:11:34 2025

@author: nitaishah
"""

# SCRIPTS/train_paper_model.py
import torch
import torch.nn as nn
from .fusion_encoder_paper import MultimodalTransformerEncoder
from .lidar_decoder_paper import LidarDecoderPaper


class LidarAutoencoderPaper(nn.Module):
    """
    Paper-faithful wrapper: (B,3072) → Encoder → (B,1024) → Decoder → (B,1,1088,1440)

    Exposes:
    - forward(x3072)        : full pass
    - encode(x3072)         : returns latent 1024
    - decode(latent1024)    : returns range image
    """

    def __init__(
        self,
        enc_kwargs: dict | None = None,
        dec_kwargs: dict | None = None,
    ):
        super().__init__()
        enc_kwargs = enc_kwargs or {}
        dec_kwargs = dec_kwargs or {}

        self.encoder = MultimodalTransformerEncoder(**enc_kwargs)
        self.decoder = LidarDecoderPaper(**dec_kwargs)

    @torch.no_grad()
    def encode(self, x3072: torch.Tensor) -> torch.Tensor:
        return self.encoder(x3072)

    @torch.no_grad()
    def decode(self, z1024: torch.Tensor) -> torch.Tensor:
        return self.decoder(z1024)

    def forward(self, x3072: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x3072)       # (B,1024)
        out = self.decoder(z)         # (B,1,1088,1440)
        return out
