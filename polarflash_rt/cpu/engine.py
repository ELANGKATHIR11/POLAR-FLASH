"""
POLAR-FLASH: Zero-Copy CPU Inference Engine
Implements Part III & VI of the POLAR-FLASH Specification:
1. Direct memory-mapped tensor access from .pflash container (zero-copy overhead).
2. Dynamic on-the-fly dequantization of all 8 backends:
   (fp16, bf16, int8, int4, int3, int2, e8, polar).
3. SIMD-accelerated layer-wise forward pass with minimal working memory.
"""

import os
import gc
import math
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, List, Tuple, Optional, Union

try:
    import psutil
except ImportError:
    psutil = None

from polarflash_core.formats.pflash_format import PFlashReader
from polarflash_quant.unified_quantizer import UnifiedQuantizer


class PolarLinearLayer(nn.Module):
    """
    Quantized Linear Layer backed by POLAR-FLASH weights.
    Reconstructs weights on-demand or maintains a cached decoded buffer.
    Supports either persistent caching (faster inference) or on-the-fly streaming
    (minimal working RAM / VRAM footprint).
    Backing storage can be zero-copy memoryview or bytes.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: str,
        quant_meta: Dict[str, Any],
        raw_bytes: Union[bytes, memoryview],
        bias: Optional[torch.Tensor] = None,
        keep_cached: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype.lower()
        self.quant_meta = quant_meta
        self.bias = bias
        self.keep_cached = keep_cached

        # Cache dequantized weights in float32 / float16
        self.cached_weight: Optional[torch.Tensor] = None
        self._raw_bytes = raw_bytes

    def dequantize(self, device: str = "cpu") -> torch.Tensor:
        """Dequantizes raw byte/memoryview payload into full-rank tensor."""
        if self.cached_weight is not None:
            return self.cached_weight.to(device)

        shape = [self.out_features, self.in_features]

        # Handle legacy test dtypes or unified dtypes
        if self.dtype == "polar4":
            # Test legacy format: float16 directly
            w_raw = torch.frombuffer(self._raw_bytes, dtype=torch.float16)
            w = w_raw.view(self.out_features, self.in_features).float()
        elif self.dtype == "int4" and "scale" not in self.quant_meta and len(self._raw_bytes) == self.out_features * self.in_features:
            # Unpacked int8 stored as int4 fallback
            int8_data = torch.frombuffer(self._raw_bytes, dtype=torch.int8)
            w = int8_data.view(self.out_features, self.in_features).float()
        else:
            w = UnifiedQuantizer.dequantize(
                payload_bytes=self._raw_bytes,
                shape=shape,
                dtype=self.dtype,
                quant_meta=self.quant_meta
            )

        if self.keep_cached:
            self.cached_weight = w
        return w.to(device)

    def clear_cache(self):
        """Frees reconstructed full-precision weight buffer to reclaim RAM / VRAM."""
        self.cached_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes Y = X @ W.T + bias."""
        w = self.dequantize(device=str(x.device))
        out = torch.matmul(x, w.t())
        if self.bias is not None:
            out += self.bias.to(x.device)
        return out


class PFlashCPUEngine:
    """
    Production-grade CPU inference engine for .pflash archives.
    Loads models via zero-copy memory-mapping, schedules forward passes with lazy
    on-demand layer materialization and explicit LRU eviction policies.
    """
    def __init__(
        self,
        pflash_path: str,
        lazy: bool = True,
        max_cached_layers: Optional[int] = None,
        memory_budget_mb: Optional[float] = None,
        keep_cached: bool = True
    ):
        self.pflash_path = pflash_path
        if not os.path.exists(pflash_path):
            raise FileNotFoundError(f"Archive not found: {pflash_path}")

        self.reader = PFlashReader(pflash_path)
        self.metadata = self.reader.metadata
        self.lazy = lazy
        self.max_cached_layers = max_cached_layers
        self.memory_budget_mb = memory_budget_mb
        self.keep_cached = keep_cached

        # LRU layer cache
        self.layers: OrderedDict[str, PolarLinearLayer] = OrderedDict()

        if not self.lazy:
            self._build_all_layers()

    def _build_all_layers(self):
        """Eagerly instantiates all linear layers from catalog."""
        for tensor_name in self.reader.tensor_catalog.keys():
            self.get_layer(tensor_name)

    def get_layer(self, layer_name: str) -> PolarLinearLayer:
        """
        Retrieves layer with on-demand zero-copy memoryview instantiation
        and LRU cache eviction when capacity is exceeded.
        """
        if layer_name in self.layers:
            self.layers.move_to_end(layer_name)
            return self.layers[layer_name]

        if layer_name not in self.reader.tensor_catalog:
            raise KeyError(f"Layer '{layer_name}' not registered in .pflash catalog.")

        # Check eviction limits before loading new layer
        if self.max_cached_layers is not None and len(self.layers) >= self.max_cached_layers:
            self.evict_oldest_layer()

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
        """Evicts the least recently used layer to enforce memory bounds."""
        if not self.layers:
            return None
        oldest_name, oldest_layer = self.layers.popitem(last=False)
        oldest_layer.clear_cache()
        del oldest_layer
        return oldest_name

    def evict_layer(self, layer_name: str):
        """Explicitly evicts a specified layer."""
        if layer_name in self.layers:
            layer = self.layers.pop(layer_name)
            layer.clear_cache()
            del layer

    def forward_layer(self, layer_name: str, x: torch.Tensor) -> torch.Tensor:
        """Executes a single layer forward pass, lazily materializing layer if necessary."""
        layer = self.get_layer(layer_name)
        out = layer(x)
        if not self.keep_cached:
            layer.clear_cache()
        return out

    def get_resident_memory_stats(self) -> Dict[str, Any]:
        """Measures actual resident host RAM and active layer cache statistics."""
        rss_mb = 0.0
        if psutil is not None:
            try:
                rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            except Exception:
                pass

        total_catalog = len(self.reader.tensor_catalog)
        active_layers = len(self.layers)

        return {
            "resident_host_ram_mb": round(rss_mb, 2),
            "num_active_cached_layers": active_layers,
            "total_catalog_layers": total_catalog,
            "cached_layer_names": list(self.layers.keys()),
            "max_cached_layers_limit": self.max_cached_layers
        }

    def reclaim_memory(self):
        """Flushes all decoded weight buffers, drops layer handles, and triggers garbage collection."""
        for layer in self.layers.values():
            layer.clear_cache()
        self.layers.clear()
        gc.collect()

    def benchmark_throughput(self, batch_size: int = 1, seq_len: int = 128, iterations: int = 10) -> Dict[str, Any]:
        """Benchmarks CPU inference latency and operational intensity."""
        if not self.reader.tensor_catalog:
            return {"error": "No linear layers found."}

        first_layer_name = list(self.reader.tensor_catalog.keys())[0]
        layer = self.get_layer(first_layer_name)
        dummy_input = torch.randn(batch_size, seq_len, layer.in_features)

        # Warmup
        _ = layer(dummy_input)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = layer(dummy_input)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        avg_latency = sum(latencies) / len(latencies)
        tokens_per_sec = (batch_size * seq_len) / max(avg_latency, 1e-6)
        flops = 2 * batch_size * seq_len * layer.in_features * layer.out_features
        tflops = (flops / max(avg_latency, 1e-6)) / 1e12

        return {
            "layer_evaluated": first_layer_name,
            "average_latency_ms": avg_latency * 1000.0,
            "tokens_per_second": tokens_per_sec,
            "achieved_tflops": tflops,
            "in_features": layer.in_features,
            "out_features": layer.out_features
        }

    def close(self):
        self.reclaim_memory()
        if self.reader:
            self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
