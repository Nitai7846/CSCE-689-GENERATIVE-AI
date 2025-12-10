#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class PerceiverFusionEncoderV3(nn.Module):
    """
    Perceiver-Style Fusion Encoder (v3)
    -----------------------------------
    Input:  x in R[B, 3072]  (concat of 4 modality CLS tokens × 768)
    Tokens: reshape -> [B, 4, 768]
    Idea:   Maintain a small set of learnable 'latent' vectors that
            attend to the modality tokens via cross-attention; then
            do self-attention over latents, repeat; finally pool and
            project to 1024-D.
    Output: fused latent in R[B, 1024]
    """

    def __init__(
        self,
        num_modalities: int = 4,
        d_token: int = 768,
        d_latent: int = 768,
        n_latents: int = 6,
        num_heads: int = 12,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        num_blocks: int = 2,     # [CrossAttn -> SelfAttn] repeated
        out_dim: int = 1024
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.d_token = d_token
        self.d_latent = d_latent
        self.n_latents = n_latents

        # Optional per-modality projection (keeps dim=768; adds capacity)
        self.mod_proj = nn.ModuleList([nn.Linear(d_token, d_token) for _ in range(num_modalities)])

        # Learnable latent array [1, n_latents, d_latent]
        self.latents = nn.Parameter(torch.zeros(1, n_latents, d_latent))
        nn.init.trunc_normal_(self.latents, std=0.02)

        # Cross-attn: latents (queries) attend to modality tokens (keys/values)
        def cross_block():
            return nn.ModuleDict({
                "q_proj": nn.Linear(d_latent, d_latent),
                "k_proj": nn.Linear(d_token,  d_latent),
                "v_proj": nn.Linear(d_token,  d_latent),
                "attn":   nn.MultiheadAttention(embed_dim=d_latent, num_heads=num_heads, dropout=dropout, batch_first=True),
                "ff":     nn.Sequential(
                    nn.LayerNorm(d_latent),
                    nn.Linear(d_latent, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, d_latent),
                    nn.Dropout(dropout),
                ),
                "norm":   nn.LayerNorm(d_latent),
            })

        # Self-attn over latents
        def latent_sa_block():
            layer = nn.TransformerEncoderLayer(
                d_model=d_latent, nhead=num_heads, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True
            )
            return nn.TransformerEncoder(layer, num_layers=1)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({"cross": cross_block(), "self": latent_sa_block()})
            for _ in range(num_blocks)
        ])

        # Final projection to 1024
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, out_dim)
        )

        # Learnable modality type embeddings (pos-like bias per token)
        self.mod_type_embed = nn.Parameter(torch.zeros(1, num_modalities, d_token))
        nn.init.trunc_normal_(self.mod_type_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3072] -> view as [B, 4, 768]
        """
        B = x.size(0)
        tokens = x.view(B, self.num_modalities, self.d_token)  # [B,4,768]
        tokens = torch.stack([proj(tokens[:, i, :]) for i, proj in enumerate(self.mod_proj)], dim=1)
        tokens = tokens + self.mod_type_embed.expand(B, -1, -1)  # bias per modality

        # Latents per batch
        lat = self.latents.expand(B, -1, -1)  # [B, n_latents, d_latent]

        for blk in self.blocks:
            # Cross-attn: Q=lat, K/V=tokens   -> lat'
            q = blk["cross"]["q_proj"](lat)
            k = blk["cross"]["k_proj"](tokens)
            v = blk["cross"]["v_proj"](tokens)
            lat_upd, _ = blk["cross"]["attn"](q, k, v, need_weights=False)
            lat = blk["cross"]["norm"](lat + lat_upd)
            lat = lat + blk["cross"]["ff"](lat)

            # Self-attn over latents
            lat = blk["self"](lat)

        # Pool latents (mean) then project
        lat_mean = lat.mean(dim=1)  # [B, d_latent]
        z = self.out_proj(lat_mean)  # [B, 1024]
        return z
