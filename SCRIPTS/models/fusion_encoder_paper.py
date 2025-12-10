#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:10:47 2025

@author: nitaishah
"""

# SCRIPTS/models/fusion_encoder_paper.py
import torch
import torch.nn as nn


class MultimodalTransformerEncoder(nn.Module):
    """
    Paper-faithful Multimodal Transformer Encoder.

    Input : (B, 3072)  = concat of 4 modality embeddings (each 768-D)
            Assumes order [RGB, Depth, Radar-RA, Radar-RV] but is agnostic to it.
    Output: (B, 1024)  fused latent

    Notes:
    - Uses a learnable [CLS] token and per-modality embeddings.
    - Single encoder layer by default (set num_layers>1 if you later ablate depth).
    """

    def __init__(
        self,
        num_modalities: int = 4,
        d_token: int = 768,
        hidden_dim: int = 1024,
        num_heads: int = 12,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_cls: bool = True,
    ):
        super().__init__()
        assert num_modalities * d_token == 3072, \
            "Default expects 4 modalities × 768 = 3072 input width."

        self.num_modalities = num_modalities
        self.d_token = d_token
        self.use_cls = use_cls

        # modality embeddings (tiny positional cue per token)
        self.mod_embed = nn.Parameter(torch.zeros(1, num_modalities, d_token))
        nn.init.trunc_normal_(self.mod_embed, std=0.02)

        # optional [CLS] token
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # transformer encoder (batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # project pooled token to latent 1024
        self.proj = nn.Linear(d_token, hidden_dim)

    def forward(self, x3072: torch.Tensor) -> torch.Tensor:
        """
        x3072: (B, 3072) → reshape to (B, 4, 768)
        returns: (B, 1024)
        """
        B = x3072.size(0)
        x = x3072.view(B, self.num_modalities, self.d_token)   # (B, 4, 768)
        x = x + self.mod_embed                                 # (B, 4, 768)

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)             # (B, 1, 768)
            x = torch.cat([cls, x], dim=1)                     # (B, 5, 768)

        z = self.encoder(x)                                    # (B, 5/4, 768)
        pooled = z[:, 0] if self.use_cls else z.mean(dim=1)    # (B, 768)
        return self.proj(pooled)                                # (B, 1024)
