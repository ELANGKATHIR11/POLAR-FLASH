"""
POLAR-FLASH: Model Ingestion & Unified Serialization Engine
Implements Part I & X of the POLAR-FLASH Specification:
1. Ingests Hugging Face Safetensors checkpoints and PyTorch model directories.
2. Architecture adapters for Llama, Qwen, Mistral, and Multimodal Vision Projectors.
3. Quantizes tensors using UnifiedQuantizer supporting all 8 backends (fp16, bf16, int8, int4, int3, int2, e8, polar).
4. Verifies tensor reconstruction with MSE / SNR metrics.
5. Emits page-aligned, memory-mappable .pflash binary archives with CRC32 checksums and storage reports.
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
from polarflash_quant.unified_quantizer import UnifiedQuantizer, ArchitectureAdapter


class ModelIngestionEngine:
    """
    Ingests full transformer checkpoints and compiles them into production-ready .pflash containers.
    Supports streaming shard-by-shard ingestion to keep RAM overhead minimal.
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
        apply_hadamard: bool = True,
        polar_r_bits: int = 4,
        polar_theta_bits: int = 4
    ) -> Dict[str, Any]:
        """
        Executes end-to-end streaming ingestion, quantization, validation, and .pflash serialization.
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

        quantizer = UnifiedQuantizer(
            backend=backend,
            apply_hadamard=apply_hadamard,
            polar_r_bits=polar_r_bits,
            polar_theta_bits=polar_theta_bits
        )

        detected_arch = model_cfg.get("architectures", ["Unknown"])[0] if model_cfg else "Unknown"

        writer.set_metadata({
            "source_model": os.path.basename(os.path.abspath(model_dir)),
            "backend": backend,
            "architecture": detected_arch,
            "hidden_size": model_cfg.get("hidden_size", 2048),
            "hadamard_rotation": apply_hadamard,
            "creation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })

        total_parameters = 0
        total_layers_processed = 0
        layer_diagnostics = []

        for shard_idx, shard_path in enumerate(shards):
            shard_name = os.path.basename(shard_path)
            print(f"[Ingestion] Scanning shard {shard_idx+1}/{len(shards)}: {shard_name} ...")
            
            # Load shard into memory (discarded after shard loop to prevent memory accumulation)
            tensors = load_file(shard_path)

            for key, tensor in tensors.items():
                if max_layers is not None and total_layers_processed >= max_layers:
                    break

                # Filter for linear projection weights (2D tensors)
                if tensor.ndim == 2 and ArchitectureAdapter.is_target_projection(key):
                    h_out, h_in = tensor.shape

                    payload_bytes, q_meta, w_hat = quantizer.quantize(tensor, name=key)

                    writer.add_tensor(
                        name=key,
                        data_bytes=payload_bytes,
                        shape=[h_out, h_in],
                        dtype=q_meta.get("dtype", backend),
                        quant_meta=q_meta
                    )

                    layer_diagnostics.append({
                        "name": key,
                        "shape": [h_out, h_in],
                        "params": tensor.numel(),
                        "effective_bpw": q_meta["effective_bpw"],
                        "mse": q_meta["mse"],
                        "snr_db": q_meta["snr_db"]
                    })

                    total_parameters += tensor.numel()
                    total_layers_processed += 1

            del tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if max_layers is not None and total_layers_processed >= max_layers:
                break

        # Save container with page alignment and CRC32 verification
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
            "elapsed_seconds": round(total_time, 2),
            "diagnostics": layer_diagnostics
        }

        return summary
