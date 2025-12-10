#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 02:37:36 2025

@author: nitaishah
"""

# lidar_decoder_paper_v2.py
import torch
import math 
import torch.nn as nn
import torch.nn.functional as F

def make_coord_grid(B, H, W, device):
    ys = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([xs, ys], dim=1)  # [B,2,H,W]

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, g=16, eps=1e-5):
        """
        g is the *max* desired number of groups. We adapt it so that
        num_groups divides num_channels (out_ch) using gcd(out_ch, g).
        """
        super().__init__()
        pad = k // 2
        # choose group count that divides out_ch
        ng = math.gcd(out_ch, g)
        if ng == 0:
            ng = 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, k, padding=pad)
        self.gn1   = nn.GroupNorm(num_groups=ng, num_channels=out_ch, eps=eps)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=pad)
        self.gn2   = nn.GroupNorm(num_groups=ng, num_channels=out_ch, eps=eps)
        self.act   = nn.GELU()
        self.proj  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        idt = self.proj(x)
        x = self.conv1(x); x = self.gn1(x); x = self.act(x)
        x = self.conv2(x); x = self.gn2(x)
        x = self.act(x + idt)
        return x

class AttnBlock(nn.Module):
    def __init__(self, ch, n_heads=4, g_hint=8, eps=1e-5):
        super().__init__()
        # adapt groups so it divides ch (helps if you change base_ch)
        ng = math.gcd(ch, g_hint)
        if ng == 0:
            ng = 1
        self.norm = nn.GroupNorm(ng, ch, eps=eps)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.norm(x).view(B, C, H * W).transpose(1, 2)  # [B,HW,C]
        y = self.attn(y, y, y, need_weights=False)[0]
        y = y.transpose(1, 2).view(B, C, H, W)
        return x + y

class LidarDecoderPaperV2(nn.Module):
    """
    Drop-in deeper decoder:
      input:  (B,1024)
      output: (B,1,1088,1440)
    """
    def __init__(self, latent_dim=1024, base_ch=256, use_attn_lowres=False):
        super().__init__()
        # target grid: 34x45 upsampled by 2^5 = 32 → 1088x1440
        self.H0, self.W0 = 34, 45
        self.fc = nn.Linear(latent_dim, base_ch * self.H0 * self.W0)

        # 1/32
        self.s32_a = ResBlock(base_ch + 2, base_ch)   # +2 for CoordConv
        self.s32_b = ResBlock(base_ch, base_ch)
        self.att32 = AttnBlock(base_ch) if use_attn_lowres else nn.Identity()
        # 1/16
        self.up16  = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.s16_a = ResBlock(base_ch + 2, base_ch)   # CoordConv at every scale
        self.s16_b = ResBlock(base_ch, base_ch)
        self.att16 = AttnBlock(base_ch) if use_attn_lowres else nn.Identity()
        # 1/8
        self.up8   = nn.ConvTranspose2d(base_ch, base_ch//2, 2, stride=2)
        self.s8_a  = ResBlock(base_ch//2 + 2, base_ch//2)
        self.s8_b  = ResBlock(base_ch//2, base_ch//2)
        # 1/4
        self.up4   = nn.ConvTranspose2d(base_ch//2, base_ch//4, 2, stride=2)
        self.s4_a  = ResBlock(base_ch//4 + 2, base_ch//4)
        self.s4_b  = ResBlock(base_ch//4, base_ch//4)
        # 1/2
        self.up2   = nn.ConvTranspose2d(base_ch//4, base_ch//8, 2, stride=2)
        self.s2_a  = ResBlock(base_ch//8 + 2, base_ch//8)
        self.s2_b  = ResBlock(base_ch//8, base_ch//8)
        # 1/1
        self.up1   = nn.ConvTranspose2d(base_ch//8, base_ch//16, 2, stride=2)
        self.s1_a  = ResBlock(base_ch//16 + 2, base_ch//16)
        self.s1_b  = ResBlock(base_ch//16, base_ch//16)

        self.head  = nn.Sequential(
            nn.Conv2d(base_ch//16, base_ch//16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch//16, 1, 1)
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z).view(B, -1, self.H0, self.W0)                 # [B,C,34,45]

        # 1/32
        coords = make_coord_grid(B, self.H0, self.W0, z.device)
        x = torch.cat([x, coords], dim=1)
        x = self.s32_a(x); x = self.s32_b(x); x = self.att32(x)

        # 1/16
        x = self.up16(x)
        coords = make_coord_grid(B, x.shape[2], x.shape[3], z.device)
        x = torch.cat([x, coords], dim=1)
        x = self.s16_a(x); x = self.s16_b(x); x = self.att16(x)

        # 1/8
        x = self.up8(x)
        coords = make_coord_grid(B, x.shape[2], x.shape[3], z.device)
        x = torch.cat([x, coords], dim=1)
        x = self.s8_a(x); x = self.s8_b(x)

        # 1/4
        x = self.up4(x)
        coords = make_coord_grid(B, x.shape[2], x.shape[3], z.device)
        x = torch.cat([x, coords], dim=1)
        x = self.s4_a(x); x = self.s4_b(x)

        # 1/2
        x = self.up2(x)
        coords = make_coord_grid(B, x.shape[2], x.shape[3], z.device)
        x = torch.cat([x, coords], dim=1)
        x = self.s2_a(x); x = self.s2_b(x)

        # 1/1 (1088x1440)
        x = self.up1(x)
        coords = make_coord_grid(B, x.shape[2], x.shape[3], z.device)
        x = torch.cat([x, coords], dim=1)
        x = self.s1_a(x); x = self.s1_b(x)

        return self.head(x)                                          # [B,1,1088,1440]
