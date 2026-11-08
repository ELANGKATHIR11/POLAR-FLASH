"""
POLAR-FLASH: Model Ingestion & End-to-End Compression Engine
Implements Part I & X of the POLAR-FLASH Specification:
1. Ingests Hugging Face Safetensors checkpoints and PyTorch model directories.
2. Extracts linear projection weights (Q, K, V, O, Gate, Up, Down).
3. Applies Walsh-Hadamard orthogonal rotation to eliminate outlier coordinate spikes.
4. Compresses layers with Polar, E8, or INT4 backends.
5. Emits page-aligned, memory-mappable .pflash binary archives with full verification checksums.
"""

import os
import glob
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from safetensors.torch import load_file

from polarflash_core.formats.pflash_format import PFlashWriter
from polarflash_quant.transforms.hadamard import RandomizedHadamardTransform
from polarflash_quant.polar.polar_quant import PolarQuant2D
from polarflash_quant.lattice.e8_lattice import E8LatticeQuantizer


class ModelIngestionEngine:
    """
    Ingests full transformer checkpoints and compiles them into production-ready .pflash containers.
    """
    def __init__(self, page_alignment: int = 4096):
        self.alignment = page_alignment

    def find_safetensors_files(self, model_dir: str) -> List[str]:
        """Discovers safetensors shards within a model folder."""
        shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
        return sorted(shards)

    def ingest_and_quantize(
        self,
        model_dir: str,
        output_pflash: str,
        backend: str = "polar",
        max_layers: Optional[int] = None,
        apply_hadamard: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end ingestion, quantization, and .pflash serialization.
        """
        t0 = time.time()
        backend = backend.lower()
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Source model directory not found: {model_dir}")

        shards = self.find_safetensors_files(model_dir)
        writer = PFlashWriter(page_alignment=self.alignment)

        # Ingest config if available
        config_path = os.path.join(model_dir, "config.json")
        model_cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                model_cfg = json.load(f)

        writer.set_metadata({
            "source_model": os.path.basename(os.path.abspath(model_dir)),
            "backend": backend,
            "architecture": model_cfg.get("architectures", ["Unknown"])[0] if model_cfg else "Unknown",
            "hidden_size": model_cfg.get("hidden_size", 2048),
            "hadamard_rotation": apply_hadamard,
            "creation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })

        total_parameters = 0
        total_layers_processed = 0

        # Target 2D projection tensors
        target_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        for shard_idx, shard_path in enumerate(shards):
            print(f"[Ingestion] Scanning shard {shard_idx+1}/{len(shards)}: {os.path.basename(shard_path)} ...")
            tensors = load_file(shard_path)

            for key, tensor in tensors.items():
                if max_layers is not None and total_layers_processed >= max_layers:
                    break

                # Filter for linear projection weights (2D tensors)
                if tensor.ndim == 2 and any(tk in key for tk in target_keys):
                    h_out, h_in = tensor.shape
                    w_float = tensor.float()

                    # Walsh-Hadamard pre-rotation if dimension is power of 2
                    if apply_hadamard and (h_in & (h_in - 1) == 0):
                        rht = RandomizedHadamardTransform(dim=h_in, seed=42)
                        rotated = rht.forward(w_float)
                    else:
                        rotated = w_float

                    # Quantize according to backend
                    if backend == "polar":
                        # Even length check
                        flat_len = rotated.numel()
                        if flat_len % 2 != 0:
                            rotated = rotated.view(-1)[:-1]
                        quantizer = PolarQuant2D(r_bits=4, theta_bits_base=4)
                        w_hat, q_meta = quantizer.quantize(rotated)
                        payload = w_hat.half().numpy().tobytes()
                        dtype = "polar4"

                    elif backend == "e8":
                        quantizer = E8LatticeQuantizer(search_steps=8)
                        w_hat, q_meta = quantizer.quantize(rotated)
                        payload = w_hat.half().numpy().tobytes()
                        dtype = "e8_lattice"

                    else:  # Symmetric INT4
                        scale = torch.max(torch.abs(rotated)) / 7.0
                        q_idx = torch.clamp(torch.round(rotated / scale), -8, 7).to(torch.int8)
                        payload = q_idx.numpy().tobytes()
                        dtype = "int4"
                        q_meta = {"scale": float(scale.item())}

                    writer.add_tensor(
                        name=key,
                        data_bytes=payload,
                        shape=[h_out, h_in],
                        dtype=dtype,
                        quant_meta=q_meta
                    )

                    total_parameters += tensor.numel()
                    total_layers_processed += 1

            if max_layers is not None and total_layers_processed >= max_layers:
                break

        # Save container
        save_stats = writer.save(output_pflash)
        total_time = time.time() - t0

        summary = {
            "source_model_dir": model_dir,
            "output_pflash": output_pflash,
            "backend": backend,
            "total_tensors_quantized": total_layers_processed,
            "total_parameters": total_parameters,
            "archive_size_bytes": save_stats["total_bytes"],
            "archive_size_mb": round(save_stats["total_bytes"] / (1024 ** 2), 2),
            "effective_bpw": save_stats["effective_bpw"],
            "elapsed_seconds": round(total_time, 2)
        }
        return summary

