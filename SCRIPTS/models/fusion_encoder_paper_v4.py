import torch
import torch.nn as nn

class CrossModalTransformerEncoder(nn.Module):
    """
    Cross-Modal Transformer Encoder (v3)
    ------------------------------------
    Input: (B, 3072) = 4 modality embeddings × 768 each.
    Architecture:
      - Splits input into 4 modality tokens (B,4,768)
      - Uses *pairwise cross-attention* to learn relationships
      - Adds learnable fusion token that aggregates all modalities
      - Output: fused latent (B, 1024)
    """

    def __init__(
        self,
        num_modalities=4,
        d_token=768,
        hidden_dim=1024,
        num_heads=8,
        ffn_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.d_token = d_token

        # Per-modality linear projections (learn modality embeddings)
        self.mod_proj = nn.ModuleList([
            nn.Linear(d_token, d_token) for _ in range(num_modalities)
        ])

        # Learnable fusion token
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.fusion_token, std=0.02)

        # Cross-modal attention blocks
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(2)  # two-layer cross fusion
        ])

        # Adaptive modality weighting
        self.gate = nn.Sequential(
            nn.Linear(num_modalities * d_token, num_modalities),
            nn.Softmax(dim=-1)
        )

        # Final projection
        self.proj = nn.Linear(d_token, hidden_dim)

    def forward(self, x):
        """
        x: (B, 3072)
        """
        B = x.size(0)
        # Split into modality tokens
        tokens = x.view(B, self.num_modalities, self.d_token)
        tokens = torch.stack([proj(tokens[:, i, :]) for i, proj in enumerate(self.mod_proj)], dim=1)

        # Add fusion token
        fusion_tok = self.fusion_token.expand(B, -1, -1)
        tokens = torch.cat([fusion_tok, tokens], dim=1)  # (B, 1+4, 768)

        # Pass through cross-attention layers
        for layer in self.cross_attn_layers:
            tokens = layer(tokens)

        # Separate fusion and modality tokens
        fusion_tok, mod_tokens = tokens[:, 0], tokens[:, 1:]

        # Adaptive gating between modalities
        gates = self.gate(mod_tokens.reshape(B, -1)).unsqueeze(-1)  # (B,4,1)
        fused_modality = (mod_tokens * gates).sum(dim=1)  # (B,768)

        # Combine fusion + gated modalities
        fused = fusion_tok + fused_modality
        return self.proj(fused)  # (B,1024)
