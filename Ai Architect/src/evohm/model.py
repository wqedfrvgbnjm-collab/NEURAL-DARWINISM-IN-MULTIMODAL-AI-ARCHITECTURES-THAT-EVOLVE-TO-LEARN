from typing import Dict, Tuple
import torch
import torch.nn as nn

from .extractors import VisionTransformerExtractor, TextTransformerExtractor
from .fusion import MultiHeadCrossModalFusion
from .router import GraphAttentionRouter
from .modules import NeuralModule, ModuleType


class EvolutionaryNeuralArchitecture(nn.Module):
    def __init__(self, dim: int, text_model_name: str, image_model_name: str):
        super().__init__()
        self.dim = dim
        self.neural_modules = nn.ModuleList()
        self.router = GraphAttentionRouter(dim)
        self.image_extractor = VisionTransformerExtractor(image_model_name, dim)
        self.text_extractor = TextTransformerExtractor(text_model_name, dim)
        self.fusion = MultiHeadCrossModalFusion(dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.5),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.LayerNorm(dim // 2),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, 2),
        )
        self.generation = 0
        self.module_contributions: Dict[int, float] = {}
        self.probation_info = None

    def add_module(self, module_type: str, reason: str, config: Dict):
        module_id = (max((m.module_id for m in self.neural_modules), default=-1) + 1)
        new_module = NeuralModule(module_type, self.dim, module_id, reason, config)
        self.neural_modules.append(new_module)
        return new_module

    def remove_module_by_id(self, module_id: int, reason: str):
        target_idx = -1
        for i, module in enumerate(self.neural_modules):
            if module.module_id == module_id:
                target_idx = i
                break
        if target_idx != -1:
            del self.neural_modules[target_idx]
            print(f"Removed module {module_id} ({reason})")

    def forward(self, batch: Dict[str, torch.Tensor], return_attn: bool = False):
        image_features = self.image_extractor(batch['image'])
        text_features = self.text_extractor(batch['text_input_ids'], batch['text_attention_mask'])
        fused_features = self.fusion(image_features, text_features)
        module_outputs = [module(fused_features) for module in self.neural_modules]
        routed_features, attn_weights = self.router(fused_features, module_outputs, self.probation_info)
        if self.training and attn_weights.numel() > 0:
            avg_attn = attn_weights.mean(dim=0)
            for i, module in enumerate(self.neural_modules):
                if i < len(avg_attn):
                    self.module_contributions[module.module_id] = float(avg_attn[i].item())
        logits = self.classifier(routed_features)
        if return_attn:
            return logits, attn_weights
        return logits
