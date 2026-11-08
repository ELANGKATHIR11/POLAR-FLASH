"""
POLAR-FLASH: Parameter-Efficient LoRA on Quantized Compressed Layers
Implements Part IV & Milestone 6 of the POLAR-FLASH Specification:
1. Freezes base quantized weights W_hat.
2. Injects trainable low-rank adapters: W = W_hat + (alpha / r) * (B @ A).
3. Ultra-low training memory overhead: gradients only computed for rank-r adapters.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class PolarLoRALinear(nn.Module):
    """
    Combines a frozen POLAR-FLASH base linear layer with trainable low-rank LoRA matrices.
    Forward: Y = X @ (W_base + (alpha / r) * B @ A).T + bias
    """
    def __init__(
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for A, zero for B so initial adapter contribution is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Base forward pass (frozen quantized weights)
        base_out = self.base_layer(x)

        # 2. LoRA adapter forward pass
        # x: (..., in_features)
        # lora_A: (r, in_features), lora_B: (out_features, r)
        dropped_x = self.dropout(x)
        lora_intermediate = torch.matmul(dropped_x, self.lora_A.t())  # (..., r)
        lora_out = torch.matmul(lora_intermediate, self.lora_B.t()) * self.scaling  # (..., out_features)

        return base_out + lora_out

    def merge_weights(self) -> torch.Tensor:
        """Merges LoRA adapter into base weights for zero-overhead inference."""
        adapter_delta = (self.lora_B @ self.lora_A) * self.scaling
        if hasattr(self.base_layer, "dequantize"):
            base_w = self.base_layer.dequantize()
            return base_w + adapter_delta
        elif hasattr(self.base_layer, "weight"):
            return self.base_layer.weight + adapter_delta
        return adapter_delta

