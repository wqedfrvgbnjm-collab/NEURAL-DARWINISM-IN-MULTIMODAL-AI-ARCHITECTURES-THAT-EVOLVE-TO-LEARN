import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel


class VisionTransformerExtractor(nn.Module):
    def __init__(self, image_model_name: str, out_dim: int):
        super().__init__()
        self.vit = CLIPVisionModel.from_pretrained(image_model_name)
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        for layer in self.vit.vision_model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.projection = nn.Linear(self.vit.config.hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.vit(x).pooler_output)


class TextTransformerExtractor(nn.Module):
    def __init__(self, text_model_name: str, out_dim: int):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for param in self.text_model.parameters():
            param.requires_grad = False
        for layer in self.text_model.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.projection = nn.Linear(self.text_model.config.dim, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.projection(self.text_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])
