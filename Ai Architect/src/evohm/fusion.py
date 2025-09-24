import torch
import torch.nn as nn


class MultiHeadCrossModalFusion(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        img_seq, text_seq = image_features.unsqueeze(1), text_features.unsqueeze(1)
        attended_features, _ = self.cross_attn(query=img_seq, key=text_seq, value=text_seq)
        fused = self.norm1(image_features + self.dropout(attended_features.squeeze(1)))
        fused = self.norm2(fused + self.dropout(self.ffn(fused)))
        return fused
