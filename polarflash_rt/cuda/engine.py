"""
POLAR-FLASH: CUDA Accelerated GPU Runtime Engine
Implements Part III & VI of the POLAR-FLASH Specification:
1. Pinned-memory streaming and asynchronous host-to-device transfers via CUDA streams.
2. Tensor-Core accelerated FP16/BF16 matrix multiplication.
3. GPU memory management avoiding VRAM fragmentation.
4. Automatic fallback when CUDA is not present.
"""

import os
import gc
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

from polarflash_core.formats.pflash_format import PFlashReader
from polarflash_rt.cpu.engine import PolarLinearLayer


class PFlashCUDAEngine:
    """
    High-performance CUDA runtime for POLAR-FLASH.
    Execution Architecture:
    - 2-Stage Execution: On-the-fly dequantization of packed .pflash weights followed by CuBLAS Tensor-Core GEMM.
    - Transparent Hardware Classification: Weights remain packed in container and host storage,
      and are unpacked into working VRAM buffers during inference. This is not a low-bit fused kernel.
    - Zero-copy memoryview storage access with lazy on-demand instantiation and LRU eviction.
    - Pinned memory transfers via CUDA streams.
    """
    def __init__(
        self,
        pflash_path: str,
        device: str = "cuda:0",
        lazy: bool = True,
        max_cached_layers: Optional[int] = None,
        memory_budget_mb: Optional[float] = None,
        keep_cached: bool = True
    ):
        self.pflash_path = pflash_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.reader = PFlashReader(pflash_path)
        self.lazy = lazy
        self.max_cached_layers = max_cached_layers
        self.memory_budget_mb = memory_budget_mb
        self.keep_cached = keep_cached

        # LRU layer cache
        self.layers: OrderedDict[str, PolarLinearLayer] = OrderedDict()
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

        if not self.lazy:
            self._build_all_layers()

    def _build_all_layers(self):
        """Eagerly instantiates all linear layers from catalog."""
        for tensor_name in self.reader.tensor_catalog.keys():
            self.get_layer(tensor_name)

    def get_layer(self, layer_name: str) -> PolarLinearLayer:
        """
        Retrieves layer with on-demand zero-copy memoryview instantiation
        and LRU cache eviction when capacity or VRAM limits are reached.
        """
        if layer_name in self.layers:
            self.layers.move_to_end(layer_name)
            return self.layers[layer_name]

        if layer_name not in self.reader.tensor_catalog:
            raise KeyError(f"Layer '{layer_name}' not registered in .pflash catalog.")

        # Evict oldest layers if cache limit reached
        if self.max_cached_layers is not None and len(self.layers) >= self.max_cached_layers:
            self.evict_oldest_layer()

        # Evict if VRAM budget is exceeded
        if self.memory_budget_mb is not None and self.device.type == "cuda":
            current_vram = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            while current_vram > self.memory_budget_mb and self.layers:
                self.evict_oldest_layer()
                current_vram = torch.cuda.memory_allocated(self.device) / (1024 ** 2)

        meta = self.reader.tensor_catalog[layer_name]
        shape = meta["shape"]
        if len(shape) != 2:
            raise ValueError(f"Tensor '{layer_name}' is not 2D linear weight (shape: {shape})")

        out_f, in_f = shape[0], shape[1]
        # Zero-copy memoryview slice directly referencing mmap
        tensor_view = self.reader.get_tensor_view(layer_name, verify_checksum=False)

        layer = PolarLinearLayer(
            in_features=in_f,
            out_features=out_f,
            dtype=meta["dtype"],
            quant_meta=meta["quant_meta"],
            raw_bytes=tensor_view,
            keep_cached=self.keep_cached
        )
        self.layers[layer_name] = layer
        return layer

    def evict_oldest_layer(self) -> Optional[str]:
        """Evicts the least recently used layer to reclaim VRAM."""
        if not self.layers:
            return None
        oldest_name, oldest_layer = self.layers.popitem(last=False)
        oldest_layer.clear_cache()
        del oldest_layer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return oldest_name

    def evict_layer(self, layer_name: str):
        """Explicitly evicts a specified layer."""
        if layer_name in self.layers:
            layer = self.layers.pop(layer_name)
            layer.clear_cache()
            del layer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def forward_layer(self, layer_name: str, x: torch.Tensor) -> torch.Tensor:
        """Executes a layer forward pass on the target device."""
        layer = self.get_layer(layer_name)
        x_dev = x.to(self.device)
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                out = layer(x_dev)
            self.stream.synchronize()
        else:
            out = layer(x_dev)

        if not self.keep_cached:
            layer.clear_cache()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return out

    def get_resident_memory_stats(self) -> Dict[str, Any]:
        """Measures actual resident Host RAM and CUDA VRAM allocations."""
        vram_allocated_mb = 0.0
        vram_peak_mb = 0.0
        if self.device.type == "cuda":
            vram_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            vram_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        rss_mb = 0.0
        if psutil is not None:
            try:
                rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            except Exception:
                pass

        return {
            "device": str(self.device),
            "resident_host_ram_mb": round(rss_mb, 2),
            "cuda_vram_allocated_mb": round(vram_allocated_mb, 2),
            "cuda_vram_peak_mb": round(vram_peak_mb, 2),
            "num_active_cached_layers": len(self.layers),
            "total_catalog_layers": len(self.reader.tensor_catalog),
            "cached_layer_names": list(self.layers.keys())
        }

    def reclaim_memory(self):
        """Flushes all decoded weight buffers, drops layer handles, and triggers CUDA cache release."""
        for layer in self.layers.values():
            layer.clear_cache()
        self.layers.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def benchmark_gpu_throughput(
        self,
        batch_size: int = 1,
        seq_len: int = 256,
        iterations: int = 20
    ) -> Dict[str, Any]:
        """Measures GPU latency, memory footprint, and achieved TFLOPS."""
        if not self.reader.tensor_catalog:
            return {"error": "No linear layers found."}

        first_layer_name = list(self.reader.tensor_catalog.keys())[0]
        layer = self.get_layer(first_layer_name)
        dummy_input = torch.randn(batch_size, seq_len, layer.in_features, device=self.device)

        # Warmup
        for _ in range(5):
            _ = layer(dummy_input)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = layer(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        avg_latency = sum(latencies) / len(latencies)
        tokens_per_sec = (batch_size * seq_len) / max(avg_latency, 1e-6)
        flops = 2 * batch_size * seq_len * layer.in_features * layer.out_features
        tflops = (flops / max(avg_latency, 1e-6)) / 1e12

        vram_used_mb = 0.0
        vram_peak_mb = 0.0
        if self.device.type == "cuda":
            vram_used_mb = torch.cuda.memory_allocated(device=self.device) / (1024 ** 2)
            vram_peak_mb = torch.cuda.max_memory_allocated(device=self.device) / (1024 ** 2)

        packed_storage_bytes = len(layer._raw_bytes)
        unpacked_working_bytes = layer.in_features * layer.out_features * 4  # Float32 elements

        return {
            "device": str(self.device),
            "execution_architecture": "2-stage on-the-fly dequantize + Tensor Core GEMM (PyTorch / CuBLAS)",
            "is_fused_lowbit_kernel": False,
            "weights_packed_in_storage": True,
            "layer_evaluated": first_layer_name,
            "packed_storage_bytes": packed_storage_bytes,
            "unpacked_working_bytes": unpacked_working_bytes,
            "compression_ratio": unpacked_working_bytes / max(packed_storage_bytes, 1),
            "average_latency_ms": avg_latency * 1000.0,
            "tokens_per_second": tokens_per_sec,
            "achieved_tflops": tflops,
            "vram_allocated_mb": vram_used_mb,
            "vram_peak_mb": vram_peak_mb
        }

    def close(self):
        self.reclaim_memory()
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
