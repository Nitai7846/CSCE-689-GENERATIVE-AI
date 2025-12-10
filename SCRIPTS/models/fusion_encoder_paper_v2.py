#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 02:36:25 2025

@author: nitaishah
"""

# fusion_encoder_paper_v2.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class GEGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.proj = nn.Linear(dim, hidden * 2)
    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            GEGLU(d_model, self.mlp_hidden),
            nn.Dropout(drop),
            nn.Linear(self.mlp_hidden, d_model)
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        # pre-norm attention
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        # pre-norm MLP
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class MultimodalTransformerEncoderV2(nn.Module):
    """
    Drop-in deeper encoder:
      input: (B, 3072) = num_modalities * d_token_flat  (defaults: 4*768)
      output: (B, 1024)
    """
    def __init__(
        self,
        num_modalities=4,
        d_token_flat=768,
        d_model=512,
        n_heads=8,
        depth=12,
        mlp_ratio=4.0,
        use_cls=True,
        proj_out=1024,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        assert num_modalities * d_token_flat == 3072, "Adjust d_token_flat/num_modalities to keep input 3072."

        self.num_modalities = num_modalities
        self.use_cls = use_cls

        # modality-specific tokenizer (small MLP) → d_model
        self.tokenizers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_token_flat),
                nn.Linear(d_token_flat, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_modalities)
        ])

        # optional CLS token
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)

        # learned modality embeddings (positional over modalities)
        self.mod_embed = nn.Parameter(torch.zeros(1, num_modalities, d_model))
        nn.init.trunc_normal_(self.mod_embed, std=0.02)

        # drop-path schedule across depth
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, drop, attn_drop, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

        # pooled projection → 1024
        self.out = nn.Sequential(
            nn.Linear(d_model, proj_out),
            nn.GELU(),
            nn.Linear(proj_out, proj_out)
        )

    def forward(self, x3072):
        B = x3072.shape[0]
        # split per modality
        xs = x3072.view(B, self.num_modalities, -1).unbind(dim=1)  # list of [B,768]
        tokens = []
        for m, x in enumerate(xs):
            tokens.append(self.tokenizers[m](x))                    # [B, d_model]
        x = torch.stack(tokens, dim=1) + self.mod_embed            # [B, M, d_model]

        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)                       # [B,1,d_model]
            x = torch.cat([cls, x], dim=1)                         # [B,1+M,d_model]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.use_cls:
            pooled = x[:, 0]                                       # CLS pooling
        else:
            pooled = x.mean(dim=1)

        return self.out(pooled)                                    # [B,1024]
