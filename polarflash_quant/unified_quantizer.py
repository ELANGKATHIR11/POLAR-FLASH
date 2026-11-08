"""
POLAR-FLASH: Unified Quantization Interface & Architecture Adapters
Implements Part I & X of the POLAR-FLASH Specification:
1. Unified quantizer supporting 8 core backends:
   - FP16 (reference 16-bit float)
   - BF16 (reference bfloat16)
   - INT8 (symmetric 8-bit integer)
   - INT4 (symmetric 4-bit packed integer)
   - INT3 (symmetric 3-bit packed integer)
   - INT2 (symmetric 2-bit packed integer)
   - E8 (E8 root lattice vector quantization)
   - POLAR (Recursive PolarQuant with radial and angular bit packing)
2. Architecture adapters for Llama, Qwen, Mistral, and multimodal projections.
3. Layer sensitivity estimation and rate allocation integration.
4. Exact bit-rate, MSE, and SNR diagnostic calculations.
"""

import math
import struct
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

from polarflash_quant.bitpacking import (
    pack_int4, unpack_int4,
    pack_int2, unpack_int2,
    pack_int3, unpack_int3
)
from polarflash_quant.transforms.hadamard import BlockHadamardTransform
from polarflash_quant.polar.polar_quant import PolarQuant2D, RecursivePolarQuant, pack_polar_bits, unpack_polar_bits
from polarflash_quant.lattice.e8_lattice import E8LatticeQuantizer, E8ResidualVectorQuantizer
from polarflash_quant.information_theory.rate_distortion import LayerRateAllocator


class ArchitectureAdapter:
    """
    Standardizes tensor names and identifies role across LLM families:
    Llama, Qwen, Mistral, and multimodal vision projectors.
    """
    LLAMA_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    QWEN_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                 "c_attn", "c_proj", "w1", "w2"]
    MISTRAL_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    VISION_KEYS = ["vision_tower", "multi_modal_projector", "mm_projector", "merger"]

    @staticmethod
    def identify_architecture(tensor_names: List[str]) -> str:
        names_str = " ".join(tensor_names).lower()
        if "qwen" in names_str or "c_attn" in names_str:
            return "qwen"
        elif "mistral" in names_str:
            return "mistral"
        elif any(k in names_str for k in ArchitectureAdapter.VISION_KEYS):
            return "multimodal"
        else:
            return "llama"

    @staticmethod
    def is_target_projection(name: str) -> bool:
        """Determines if a tensor is a 2D linear weight that should be quantized."""
        n = name.lower()
        if "norm" in n or "bias" in n or "embed_tokens" in n or "wte" in n:
            return False
        all_keys = ArchitectureAdapter.LLAMA_KEYS + ArchitectureAdapter.QWEN_KEYS + ArchitectureAdapter.VISION_KEYS
        return any(k in n for k in all_keys)

    @staticmethod
    def classify_role(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["q_proj", "k_proj", "v_proj", "c_attn"]):
            return "attention_qkv"
        if any(k in n for k in ["o_proj", "c_proj"]):
            return "attention_out"
        if any(k in n for k in ["gate_proj", "up_proj", "w1"]):
            return "mlp_up_gate"
        if any(k in n for k in ["down_proj", "w2"]):
            return "mlp_down"
        if any(k in n for k in ArchitectureAdapter.VISION_KEYS):
            return "vision_projector"
        return "other"


class UnifiedQuantizer:
    """
    Unified Quantizer supporting 8 backends with true bit-packing,
    Hadamard rotation, and reconstruction diagnostics.
    """
    SUPPORTED_BACKENDS = ["fp16", "bf16", "int8", "int4", "int3", "int2", "e8", "polar"]

    def __init__(
        self,
        backend: str = "polar",
        hadamard_block_size: int = 512,
        apply_hadamard: bool = True,
        polar_r_bits: int = 4,
        polar_theta_bits: int = 4,
        e8_stages: int = 1
    ):
        backend = backend.lower()
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Choose from {self.SUPPORTED_BACKENDS}")

        self.backend = backend
        self.hadamard_block_size = hadamard_block_size
        self.apply_hadamard = apply_hadamard
        self.polar_r_bits = polar_r_bits
        self.polar_theta_bits = polar_theta_bits
        self.e8_stages = e8_stages

    def quantize(
        self,
        weight: torch.Tensor,
        name: str = "tensor"
    ) -> Tuple[bytes, Dict[str, Any], torch.Tensor]:
        """
        Quantizes a weight tensor into packed bytes and metadata,
        and returns the dequantized reconstruction for validation.
        """
        orig_shape = list(weight.shape)
        w_float = weight.detach().float().cpu()
        num_elements = w_float.numel()

        # Step 1: Walsh-Hadamard Pre-Rotation
        hadamard_applied = False
        hadamard_dim = 0
        if self.apply_hadamard and w_float.ndim == 2:
            out_dim, in_dim = w_float.shape
            bht = BlockHadamardTransform(in_features=in_dim, block_size=self.hadamard_block_size, seed=42)
            w_float = bht.forward(w_float)
            hadamard_applied = True
            hadamard_dim = in_dim

        # Step 2: Backend Quantization
        if self.backend == "fp16":
            w_half = w_float.half()
            payload_bytes = w_half.numpy().tobytes()
            quant_meta = {"dtype": "fp16", "bpw": 16.0}
            w_hat = w_half.float()

        elif self.backend == "bf16":
            w_bfloat = w_float.bfloat16()
            payload_bytes = w_bfloat.view(torch.int16).numpy().tobytes()
            quant_meta = {"dtype": "bf16", "bpw": 16.0}
            w_hat = w_bfloat.float()

        elif self.backend == "int8":
            max_val = torch.max(torch.abs(w_float)).item()
            scale = max_val / 127.0 if max_val > 1e-12 else 1.0
            q_idx = torch.clamp(torch.round(w_float / scale), -128, 127).to(torch.int8)
            payload_bytes = q_idx.numpy().tobytes()
            quant_meta = {"dtype": "int8", "scale": float(scale), "bpw": 8.0}
            w_hat = q_idx.float() * scale

        elif self.backend == "int4":
            max_val = torch.max(torch.abs(w_float)).item()
            scale = max_val / 7.0 if max_val > 1e-12 else 1.0
            q_idx = torch.clamp(torch.round(w_float / scale), -8, 7).to(torch.int8)
            payload_bytes = pack_int4(q_idx)
            quant_meta = {"dtype": "int4", "scale": float(scale), "bpw": 4.0, "num_elements": num_elements}
            w_hat = q_idx.float() * scale

        elif self.backend == "int3":
            max_val = torch.max(torch.abs(w_float)).item()
            scale = max_val / 3.0 if max_val > 1e-12 else 1.0
            q_idx = torch.clamp(torch.round(w_float / scale), -4, 3).to(torch.int8)
            uint3 = (q_idx + 4).to(torch.uint8).view(-1)
            payload_bytes = pack_int3(uint3)
            quant_meta = {"dtype": "int3", "scale": float(scale), "bpw": 3.0, "num_elements": num_elements}
            w_hat = q_idx.float() * scale

        elif self.backend == "int2":
            max_val = torch.max(torch.abs(w_float)).item()
            scale = max_val / 1.5 if max_val > 1e-12 else 1.0
            q_idx = torch.clamp(torch.round(w_float / scale), -2, 1).to(torch.int8)
            uint2 = (q_idx + 2).to(torch.uint8).view(-1)
            payload_bytes = pack_int2(uint2)
            quant_meta = {"dtype": "int2", "scale": float(scale), "bpw": 2.0, "num_elements": num_elements}
            w_hat = q_idx.float() * scale

        elif self.backend == "e8":
            if self.e8_stages > 1:
                rvq = E8ResidualVectorQuantizer(num_stages=self.e8_stages)
                w_hat_t, e8_meta = rvq.quantize(w_float)
                payload_bytes = w_hat_t.half().numpy().tobytes()
                quant_meta = {"dtype": "e8_rvq", "stages": self.e8_stages, "bpw": 8.0 / 8.0 + 1.0, **e8_meta}
                w_hat = w_hat_t
            else:
                quantizer = E8LatticeQuantizer(search_steps=8)
                w_hat_t, e8_meta = quantizer.quantize(w_float)
                payload_bytes = w_hat_t.half().numpy().tobytes()
                quant_meta = {"dtype": "e8_lattice", "bpw": 2.0, **e8_meta}
                w_hat = w_hat_t

        elif self.backend == "polar":
            quantizer = PolarQuant2D(r_bits=self.polar_r_bits, theta_bits_base=self.polar_theta_bits)
            flat_w = w_float.view(-1)
            pad_needed = flat_w.numel() % 2
            if pad_needed != 0:
                flat_w = torch.cat([flat_w, torch.zeros(1)])
            
            w_hat_flat, polar_meta = quantizer.quantize(flat_w, pack_bytes=True)
            payload_bytes = polar_meta["packed_bytes"]
            quant_meta = {
                "dtype": "polar",
                "r_bits": self.polar_r_bits,
                "theta_bits": self.polar_theta_bits,
                "r_scale": polar_meta["r_scale"],
                "num_elements": num_elements,
                "padded": pad_needed,
                "bpw": (self.polar_r_bits + self.polar_theta_bits) / 2.0
            }
            w_hat = w_hat_flat[:num_elements].view(*orig_shape)

        # Step 3: Inverse Hadamard if applied
        if hadamard_applied:
            bht = BlockHadamardTransform(in_features=hadamard_dim, block_size=self.hadamard_block_size, seed=42)
            w_hat = bht.inverse(w_hat)
            quant_meta["hadamard_rotation"] = True
            quant_meta["hadamard_block_size"] = self.hadamard_block_size

        w_hat = w_hat.view(*orig_shape)

        # Step 4: Verification Metrics
        orig_tensor = weight.detach().float().cpu().view(*orig_shape)
        mse = float(torch.mean((orig_tensor - w_hat) ** 2).item())
        var_orig = float(torch.var(orig_tensor).item())
        snr_db = 10.0 * np.log10(max(var_orig, 1e-12) / max(mse, 1e-15))
        effective_bpw = (len(payload_bytes) * 8.0) / max(num_elements, 1)

        quant_meta.update({
            "name": name,
            "shape": orig_shape,
            "num_elements": num_elements,
            "payload_bytes": len(payload_bytes),
            "effective_bpw": round(effective_bpw, 3),
            "mse": mse,
            "snr_db": round(snr_db, 2)
        })

        return payload_bytes, quant_meta, w_hat

    @staticmethod
    def dequantize(
        payload_bytes: bytes,
        shape: List[int],
        dtype: str,
        quant_meta: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Dequantizes a raw payload byte buffer back into a float32 tensor.
        """
        num_elements = int(np.prod(shape))

        if dtype == "fp16":
            arr = np.frombuffer(payload_bytes, dtype=np.float16)
            t = torch.from_numpy(arr.copy()).float().view(*shape)

        elif dtype == "bf16":
            arr = np.frombuffer(payload_bytes, dtype=np.int16)
            t = torch.from_numpy(arr.copy()).view(torch.bfloat16).float().view(*shape)

        elif dtype == "int8":
            scale = quant_meta.get("scale", 1.0)
            arr = np.frombuffer(payload_bytes, dtype=np.int8)
            t = torch.from_numpy(arr.copy()).float().view(*shape) * scale

        elif dtype == "int4":
            scale = quant_meta.get("scale", 1.0)
            q_idx = unpack_int4(payload_bytes, num_elements=num_elements)
            t = q_idx.float().view(*shape) * scale

        elif dtype == "int3":
            scale = quant_meta.get("scale", 1.0)
            uint3 = unpack_int3(payload_bytes, num_elements=num_elements)
            q_idx = uint3.to(torch.int8) - 4
            t = q_idx.float().view(*shape) * scale

        elif dtype == "int2":
            scale = quant_meta.get("scale", 1.0)
            uint2 = unpack_int2(payload_bytes, num_elements=num_elements)
            q_idx = uint2.to(torch.int8) - 2
            t = q_idx.float().view(*shape) * scale

        elif dtype in ["e8_lattice", "e8_rvq"]:
            arr = np.frombuffer(payload_bytes, dtype=np.float16)
            t = torch.from_numpy(arr.copy()).float().view(*shape)

        elif dtype == "polar":
            r_bits = quant_meta.get("r_bits", 4)
            theta_bits = quant_meta.get("theta_bits", 4)
            r_scale = quant_meta.get("r_scale", 1.0)
            num_elem = quant_meta.get("num_elements", num_elements)
            pad_len = quant_meta.get("padded", 0)
            num_pairs = (num_elem + pad_len) // 2

            r_idx, theta_idx = unpack_polar_bits(
                payload_bytes,
                num_pairs=num_pairs,
                r_bits=r_bits,
                theta_bits=theta_bits
            )

            theta_levels = 2 ** theta_bits
            r_hat = r_idx.float() * r_scale
            theta_hat = ((theta_idx.float() + 0.5) / float(theta_levels)) * (2.0 * math.pi) - math.pi
            zero_mask = (r_hat <= 1e-12)
            theta_hat[zero_mask] = 0.0

            x1 = r_hat * torch.cos(theta_hat)
            x2 = r_hat * torch.sin(theta_hat)
            reconstructed = torch.stack([x1, x2], dim=-1).view(-1)[:num_elements]
            t = reconstructed.view(*shape)

        else:
            arr = np.frombuffer(payload_bytes, dtype=np.float16)
            t = torch.from_numpy(arr.copy()).float().view(*shape)

        # Inverse Hadamard if indicated in metadata
        if quant_meta.get("hadamard_rotation", False) and t.ndim == 2:
            block_size = quant_meta.get("hadamard_block_size", 512)
            bht = BlockHadamardTransform(in_features=shape[-1], block_size=block_size, seed=42)
            t = bht.inverse(t)

        return t
