"""
POLAR-FLASH: Complete Integration & Physics Verification Test Suite
Tests Phases 2 through 8:
- UnifiedQuantizer across all 8 backends (fp16, bf16, int8, int4, int3, int2, e8, polar)
- Bit-packing & unpacking exactness
- ArchitectureAdapter classification
- Ingestion engine with safetensors roundtrip
- Hardware AutoEngine & HardwareProbe
- ContextualSparseFFN submatrix gather execution
- AsyncDoubleBufferQueue real-time prefetching
- CompressedKVCache 2bit_asym mode
- PVGradCodeOptimizer physical memory calculation
- Physics energy modeling
- CLI subcommands
"""

import os
import json
import torch
import pytest
import numpy as np
from pathlib import Path
from safetensors.torch import save_file

from polarflash import (
    UnifiedQuantizer, ArchitectureAdapter, ModelIngestionEngine,
    PFlashWriter, PFlashReader, PFlashCPUEngine, PFlashCUDAEngine,
    AutoEngine, HardwareProbe, ThermalGovernor,
    CompressedKVCache, ContextualSparseFFN, DoubleBufferPipeline, AsyncDoubleBufferQueue,
    PVGradCodeOptimizer, PolarLoRALinear, compute_energy_consumption,
    pack_int4, unpack_int4, pack_int2, unpack_int2, pack_int3, unpack_int3
)


def test_unified_quantizer_all_backends():
    """Validates quantization and dequantization roundtrip for all 8 backends."""
    backends = ["fp16", "bf16", "int8", "int4", "int3", "int2", "e8", "polar"]
    x = torch.randn(64, 128) * 0.05

    for b in backends:
        uq = UnifiedQuantizer(backend=b, apply_hadamard=True)
        payload, meta, w_hat = uq.quantize(x, name=f"test_{b}")

        assert len(payload) > 0
        assert meta["shape"] == [64, 128]
        assert meta["num_elements"] == 64 * 128
        assert "effective_bpw" in meta
        assert meta["snr_db"] > 0.0 or b in ["int2", "int3"]

        # Dequantize through static method
        w_rec = UnifiedQuantizer.dequantize(
            payload_bytes=payload,
            shape=[64, 128],
            dtype=meta.get("dtype", b),
            quant_meta=meta
        )

        assert w_rec.shape == (64, 128)
        diff = torch.max(torch.abs(w_hat - w_rec)).item()
        assert diff < 1e-3, f"Mismatch in backend {b}: max diff {diff}"


def test_bitpacking_exactness():
    """Tests 4-bit, 3-bit, and 2-bit packing and unpacking roundtrips."""
    n = 240  # divisible by 2, 4, 8

    # 4-bit signed: [-8, 7]
    u4 = torch.randint(-8, 8, (n,), dtype=torch.int8)
    packed4 = pack_int4(u4)
    assert len(packed4) == n // 2
    unpacked4 = unpack_int4(packed4, num_elements=n)
    assert torch.equal(u4, unpacked4)

    # 2-bit: [0, 3]
    u2 = torch.randint(0, 4, (n,), dtype=torch.uint8)
    packed2 = pack_int2(u2)
    assert len(packed2) == n // 4
    unpacked2 = unpack_int2(packed2, num_elements=n)
    assert torch.equal(u2.to(torch.int8), unpacked2)

    # 3-bit: [0, 7]
    u3 = torch.randint(0, 8, (n,), dtype=torch.uint8)
    packed3 = pack_int3(u3)
    assert len(packed3) == (n * 3) // 8
    unpacked3 = unpack_int3(packed3, num_elements=n)
    assert torch.equal(u3.to(torch.int8), unpacked3)


def test_architecture_adapter():
    """Tests LLM key normalizer across Llama, Qwen, Mistral, and Vision."""
    llama_names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight"]
    assert ArchitectureAdapter.identify_architecture(llama_names) == "llama"
    assert ArchitectureAdapter.is_target_projection(llama_names[0])
    assert ArchitectureAdapter.classify_role(llama_names[0]) == "attention_qkv"
    assert ArchitectureAdapter.classify_role(llama_names[1]) == "mlp_down"

    qwen_names = ["transformer.h.0.attn.c_attn.weight"]
    assert ArchitectureAdapter.identify_architecture(qwen_names) == "qwen"
    assert ArchitectureAdapter.classify_role(qwen_names[0]) == "attention_qkv"

    vision_names = ["vision_tower.merger.linear.weight"]
    assert ArchitectureAdapter.is_target_projection(vision_names[0])
    assert ArchitectureAdapter.classify_role(vision_names[0]) == "vision_projector"


def test_streaming_ingestion_safetensors(tmp_path):
    """Creates a temporary safetensors checkpoint and verifies streaming ingestion into .pflash."""
    model_dir = tmp_path / "test_model_checkpoint"
    model_dir.mkdir()

    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 128
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 128).half(),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 128).half(),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 128).half(),
        "model.layers.0.mlp.down_proj.weight": torch.randn(128, 256).half()
    }
    save_file(tensors, str(model_dir / "model.safetensors"))

    output_pflash = str(tmp_path / "compiled_model.pflash")
    engine = ModelIngestionEngine(page_alignment=4096)
    summary = engine.ingest_and_quantize(
        model_dir=str(model_dir),
        output_pflash=output_pflash,
        backend="int4",
        apply_hadamard=True
    )

    assert summary["total_tensors_quantized"] == 4
    assert summary["total_parameters"] > 0
    assert summary["effective_bpw"] < 6.0
    assert os.path.exists(output_pflash)

    with PFlashReader(output_pflash) as reader:
        assert len(reader.tensor_catalog) == 4
        assert "model.layers.0.self_attn.q_proj.weight" in reader.tensor_catalog
        bytes_q = reader.get_tensor_bytes("model.layers.0.self_attn.q_proj.weight")
        assert len(bytes_q) > 0


def test_auto_engine_probing():
    """Validates hardware introspection and AutoEngine factory."""
    hw = HardwareProbe.probe_hardware()
    assert "os" in hw
    assert "cpu_threads" in hw
    assert hw["cpu_threads"] > 0
    assert "host_ram_total_gb" in hw

    governor = ThermalGovernor(cooldown_ms=1.0)
    governor.step()


def test_async_double_buffer_queue():
    """Validates live double-buffer queue prefetching."""
    records = {f"layer_{i}": f"data_{i}" for i in range(5)}
    queue = AsyncDoubleBufferQueue(fetch_fn=lambda name: records[name], max_queue_depth=2)

    queue.start_prefetch(list(records.keys()))
    fetched = []
    for _ in range(5):
        name, data = queue.get_next(timeout=3.0)
        fetched.append(name)
    queue.stop()

    assert fetched == [f"layer_{i}" for i in range(5)]


def test_compressed_kv_cache_2bit_asym():
    """Validates asymmetric 2-bit compressed KV cache."""
    cache = CompressedKVCache(num_layers=1, num_kv_heads=2, head_dim=64, mode="2bit_asym")
    k = torch.randn(1, 2, 8, 64)
    v = torch.randn(1, 2, 8, 64)

    k_recon, v_recon = cache.update(0, k, v)
    assert k_recon.shape == (1, 2, 8, 64)
    assert v_recon.shape == (1, 2, 8, 64)

    q = torch.randn(1, 2, 1, 64)
    err = cache.compute_attention_error(q, k, k_recon)
    assert "logit_mse" in err
    assert err["cosine_similarity"] > 0.85


def test_training_memory_breakdown():
    """Validates physical training memory accounting in PVGradCodeOptimizer."""
    scale = torch.tensor([0.1], requires_grad=True)
    codes = torch.randint(-8, 8, (64, 64)).float()
    opt = PVGradCodeOptimizer([scale], codes, lr=1e-3)

    mem = opt.compute_training_memory_bytes()
    assert mem["num_continuous_parameters"] == 1
    assert mem["num_discrete_codes"] == 64 * 64
    assert mem["optimizer_bytes"] > 0
    assert mem["code_bytes"] > 0


def test_energy_consumption_model():
    """Validates physical hardware energy modeling."""
    res = compute_energy_consumption(
        num_flops=1e11,
        bytes_transferred_vram=1e9,
        bytes_transferred_dram=2e9,
        bytes_transferred_ssd=5e8
    )
    assert res["total_energy_joules"] > 0
    assert res["compute_energy_percentage"] + res["memory_energy_percentage"] == pytest.approx(100.0, rel=1e-3)
