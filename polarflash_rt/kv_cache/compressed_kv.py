"""
POLAR-FLASH: Compressed KV-Cache & Long-Context Physical Memory Engine
Implements Part III Section 13 of the POLAR-FLASH Specification:
1. Exact Grouped-Query Attention (GQA) memory footprint: M_KV = 2 * L * T * H_kv * D * q
2. FP16 baseline (q = 2.0 bytes), INT8 dynamic quantized cache (q = 1.0 byte), and 2-Bit packed cache (q = 0.25 bytes).
3. Sliding-window attention eviction with sink tokens.
4. Numerical fidelity tracking: Attention logit divergence and cosine similarity.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List


class CompressedKVCache:
    """
    Production-grade compressed KV-cache supporting FP16, INT8, and 2-Bit quantization
    with sliding-window eviction and attention score error monitoring.
    """
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_context_len: int = 8192,
        mode: str = "int8",  # "fp16", "int8", "2bit"
        window_size: Optional[int] = None,
        num_sink_tokens: int = 4
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_context_len = max_context_len
        self.mode = mode.lower()
        self.window_size = window_size
        self.num_sink_tokens = num_sink_tokens

        # Bytes per cache element
        self.q_bytes = {
            "fp16": 2.0,
            "int8": 1.0,
            "2bit": 0.25
        }.get(self.mode, 1.0)

        # Internal layer caches: list of dicts per layer
        self.layers: List[Dict[str, Any]] = [
            {"k_data": None, "v_data": None, "k_scales": None, "v_scales": None, "seq_len": 0}
            for _ in range(num_layers)
        ]

    def theoretical_memory_bytes(self, num_tokens: int) -> int:
        """Calculates theoretical memory consumption in bytes."""
        # M_KV = 2 * L * T * H_kv * D * q
        return int(2 * self.num_layers * num_tokens * self.num_kv_heads * self.head_dim * self.q_bytes)

    def _quantize_int8(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token affine symmetric INT8 quantization."""
        # x: (batch, heads, seq, dim)
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / 127.0
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
        return q, scale

    def _dequantize_int8(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantizes INT8 to floating point."""
        return q.to(scale.dtype) * scale

    def _quantize_2bit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        2-Bit uniform quantization (4 centroids: -1.5, -0.5, 0.5, 1.5 * delta).
        Packs 4 values per uint8 byte.
        """
        # delta = max(|x|) / 1.75
        max_val = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        delta = torch.clamp(max_val / 1.75, min=1e-8)
        norm_x = x / delta
        # Index in {0, 1, 2, 3}: levels [-1.5, -0.5, 0.5, 1.5]
        indices = torch.clamp(torch.floor(norm_x + 2.0), 0, 3).to(torch.uint8)

        # Bit-packing: 4 2-bit values into 1 byte along last dimension
        orig_dim = indices.shape[-1]
        pad_len = (4 - (orig_dim % 4)) % 4
        if pad_len > 0:
            pad = torch.zeros(*indices.shape[:-1], pad_len, dtype=torch.uint8, device=indices.device)
            indices = torch.cat([indices, pad], dim=-1)

        reshaped = indices.view(*indices.shape[:-1], -1, 4)
        packed = (
            (reshaped[..., 0] << 6) |
            (reshaped[..., 1] << 4) |
            (reshaped[..., 2] << 2) |
            reshaped[..., 3]
        )
        return packed, delta

    def _dequantize_2bit(self, packed: torch.Tensor, delta: torch.Tensor, orig_dim: int) -> torch.Tensor:
        """Unpacks and dequantizes 2-bit values."""
        v0 = (packed >> 6) & 0x03
        v1 = (packed >> 4) & 0x03
        v2 = (packed >> 2) & 0x03
        v3 = packed & 0x03

        unpacked = torch.stack([v0, v1, v2, v3], dim=-1).view(*packed.shape[:-1], -1)[..., :orig_dim]
        # Reconstruct: centroid = (index - 1.5) * delta
        recon = (unpacked.to(delta.dtype) - 1.5) * delta
        return recon

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates KV cache for a layer and returns the reconstructed full context key & value.
        key_states, value_states: shape (batch, num_kv_heads, seq_len, head_dim)
        """
        layer = self.layers[layer_idx]
        b, h, s, d = key_states.shape

        if self.mode == "fp16":
            if layer["k_data"] is None:
                layer["k_data"] = key_states.half()
                layer["v_data"] = value_states.half()
            else:
                layer["k_data"] = torch.cat([layer["k_data"], key_states.half()], dim=2)
                layer["v_data"] = torch.cat([layer["v_data"], value_states.half()], dim=2)
            k_out = layer["k_data"]
            v_out = layer["v_data"]

        elif self.mode == "int8":
            k_q, k_sc = self._quantize_int8(key_states)
            v_q, v_sc = self._quantize_int8(value_states)

            if layer["k_data"] is None:
                layer["k_data"] = k_q
                layer["v_data"] = v_q
                layer["k_scales"] = k_sc
                layer["v_scales"] = v_sc
            else:
                layer["k_data"] = torch.cat([layer["k_data"], k_q], dim=2)
                layer["v_data"] = torch.cat([layer["v_data"], v_q], dim=2)
                layer["k_scales"] = torch.cat([layer["k_scales"], k_sc], dim=2)
                layer["v_scales"] = torch.cat([layer["v_scales"], v_sc], dim=2)

            k_out = self._dequantize_int8(layer["k_data"], layer["k_scales"])
            v_out = self._dequantize_int8(layer["v_data"], layer["v_scales"])

        elif self.mode == "2bit":
            k_q, k_sc = self._quantize_2bit(key_states)
            v_q, v_sc = self._quantize_2bit(value_states)

            if layer["k_data"] is None:
                layer["k_data"] = k_q
                layer["v_data"] = v_q
                layer["k_scales"] = k_sc
                layer["v_scales"] = v_sc
            else:
                layer["k_data"] = torch.cat([layer["k_data"], k_q], dim=2)
                layer["v_data"] = torch.cat([layer["v_data"], v_q], dim=2)
                layer["k_scales"] = torch.cat([layer["k_scales"], k_sc], dim=2)
                layer["v_scales"] = torch.cat([layer["v_scales"], v_sc], dim=2)

            k_out = self._dequantize_2bit(layer["k_data"], layer["k_scales"], d)
            v_out = self._dequantize_2bit(layer["v_data"], layer["v_scales"], d)

        # Sliding-window eviction with sink tokens
        total_seq = k_out.shape[2]
        if self.window_size and total_seq > self.window_size:
            keep_indices = list(range(self.num_sink_tokens)) + list(range(total_seq - self.window_size + self.num_sink_tokens, total_seq))
            k_out = k_out[:, :, keep_indices, :]
            v_out = v_out[:, :, keep_indices, :]

        layer["seq_len"] = k_out.shape[2]
        return k_out, v_out

    def compute_attention_error(
        self,
        query: torch.Tensor,
        fp_key: torch.Tensor,
        quant_key: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes attention logit error and cosine similarity between exact and quantized KV.
        query: (batch, heads, 1, dim)
        fp_key, quant_key: (batch, heads, seq, dim)
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        exact_logits = torch.matmul(query, fp_key.transpose(-1, -2)) * scale
        quant_logits = torch.matmul(query, quant_key.transpose(-1, -2)) * scale

        logit_mse = float(torch.mean((exact_logits - quant_logits) ** 2).item())
        cos_sim = float(torch.nn.functional.cosine_similarity(exact_logits.flatten(), quant_logits.flatten(), dim=0).item())

        exact_attn = torch.softmax(exact_logits, dim=-1)
        quant_attn = torch.softmax(quant_logits, dim=-1)
        kl_div = float(torch.sum(exact_attn * (torch.log(exact_attn + 1e-9) - torch.log(quant_attn + 1e-9))).item())

        return {
            "logit_mse": logit_mse,
            "cosine_similarity": cos_sim,
            "kl_divergence": max(0.0, kl_div)
        }
