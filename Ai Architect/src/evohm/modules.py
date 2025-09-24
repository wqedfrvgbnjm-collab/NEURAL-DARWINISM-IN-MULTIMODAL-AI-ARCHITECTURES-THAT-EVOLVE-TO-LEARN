from typing import Dict
import random
import torch
import torch.nn as nn


class ModuleType:
    MLP, TRANSFORMER, RESNET = "mlp", "transformer", "resnet"


class EnhancedMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, n_layers: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.LayerNorm([dim, 1]),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.LayerNorm([dim, 1]),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            residual = x
            out = x.unsqueeze(2)
            out = block(out)
            out = out.squeeze(2)
            x = residual + out
        return x


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, dim: int, nhead: int = 8, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x = self.norm1(x + attn_out.squeeze(1))
        x = self.norm2(x + self.ffn(x))
        return x


class NeuralModule(nn.Module):
    def __init__(self, module_type: str, dim: int, module_id: int, creation_reason: str, config: Dict):
        super().__init__()
        self.module_type, self.module_id, self.dim = module_type, module_id, dim
        self.creation_reason = creation_reason
        self.config = config
        self.epochs_existed = 0
        self.epochs_with_zero_contribution = 0
        module_map = {
            ModuleType.MLP: EnhancedMLPBlock,
            ModuleType.TRANSFORMER: EnhancedTransformerBlock,
            ModuleType.RESNET: ResNetBlock,
        }
        self.module = module_map[module_type](dim, **config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
