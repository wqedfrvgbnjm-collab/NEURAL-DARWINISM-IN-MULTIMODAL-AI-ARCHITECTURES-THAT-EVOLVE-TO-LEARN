from typing import Dict, Tuple
import torch
import torch.nn as nn


class GraphAttentionRouter(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, module_outputs, probation_info: Dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not module_outputs:
            return x, torch.empty(0, device=x.device)
        module_stack = torch.stack(module_outputs, dim=1)
        routed_features, attn_weights = self.attn(x.unsqueeze(1), module_stack, module_stack)
        if probation_info and probation_info['id'] < attn_weights.shape[-1]:
            prob_id, prob_weight = probation_info['id'], probation_info['weight']
            attn_weights[:, :, prob_id] += prob_weight
            attn_weights /= attn_weights.sum(dim=-1, keepdim=True)
        gate_val = torch.sigmoid(self.gate(x))
        final_features = gate_val * x + (1 - gate_val) * routed_features.squeeze(1)
        return final_features, attn_weights.squeeze(1)
