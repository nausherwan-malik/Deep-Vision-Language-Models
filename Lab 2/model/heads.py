from __future__ import annotations

import torch
from torch import nn


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Backbone config must expose hidden_size.")
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    @property
    def config(self):
        return self.backbone.config

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self) -> None:
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        return self.value_head(hidden_states).squeeze(-1)
