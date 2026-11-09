"""
POLAR-FLASH: Runtime Memory Architecture & Eviction Validation Tests
Verifies:
1. Zero-copy memoryview-backed tensor storage.
2. Lazy layer loading (no upfront bulk tensor materialization).
3. Explicit LRU cache eviction policy with max_cached_layers constraint.
4. Memory reclamation and resident RAM/VRAM accounting.
5. Explicit lifetime and closure safety.
"""

import os
import pytest
import torch
from polarflash_core.formats.pflash_format import PFlashWriter, PFlashReader
from polarflash_rt.cpu.engine import PFlashCPUEngine, PolarLinearLayer


@pytest.fixture
def sample_pflash_archive(tmp_path):
    """Creates a temporary multi-layer .pflash archive with 4 linear layers."""
    archive_path = str(tmp_path / "multi_layer_model.pflash")
    writer = PFlashWriter()

    torch.manual_seed(42)
    for i in range(4):
        # 64x64 FP16 weight = 8192 bytes
        w = torch.randn(64, 64, dtype=torch.float16)
        raw_bytes = w.numpy().tobytes()
        writer.add_tensor(
            name=f"model.layers.{i}.linear",
            data_bytes=raw_bytes,
            shape=[64, 64],
            dtype="fp16",
            quant_meta={"bpw": 16.0}
        )

    writer.set_metadata({"architecture": "test_transformer", "num_layers": 4})
    writer.save(archive_path)
    return archive_path


def test_lazy_loading_prevents_bulk_materialization(sample_pflash_archive):
    """Verifies that engine initialization does not instantiate all layers eagerly."""
    engine = PFlashCPUEngine(sample_pflash_archive, lazy=True)
    assert len(engine.layers) == 0, "No layers should be materialized upfront under lazy mode"

    stats = engine.get_resident_memory_stats()
    assert stats["num_active_cached_layers"] == 0
    assert stats["total_catalog_layers"] == 4

    # Now access layer 0
    l0 = engine.get_layer("model.layers.0.linear")
    assert isinstance(l0, PolarLinearLayer)
    assert len(engine.layers) == 1
    assert "model.layers.0.linear" in engine.layers

    engine.close()


def test_lru_cache_and_layer_eviction(sample_pflash_archive):
    """Verifies that max_cached_layers enforces strict LRU eviction."""
    engine = PFlashCPUEngine(sample_pflash_archive, lazy=True, max_cached_layers=2)

    # Load layer 0 and layer 1
    _ = engine.get_layer("model.layers.0.linear")
    _ = engine.get_layer("model.layers.1.linear")
    assert len(engine.layers) == 2
    assert list(engine.layers.keys()) == ["model.layers.0.linear", "model.layers.1.linear"]

    # Access layer 0 again to make it MRU (most recently used)
    _ = engine.get_layer("model.layers.0.linear")
    assert list(engine.layers.keys()) == ["model.layers.1.linear", "model.layers.0.linear"]

    # Load layer 2 -> layer 1 should be evicted!
    _ = engine.get_layer("model.layers.2.linear")
    assert len(engine.layers) == 2
    assert "model.layers.1.linear" not in engine.layers, "Layer 1 should have been evicted by LRU policy"
    assert "model.layers.0.linear" in engine.layers
    assert "model.layers.2.linear" in engine.layers

    engine.close()


def test_memoryview_storage_access_and_reclamation(sample_pflash_archive):
    """Verifies zero-copy memoryview backing and full memory reclamation."""
    engine = PFlashCPUEngine(sample_pflash_archive, lazy=True)
    layer0 = engine.get_layer("model.layers.0.linear")

    # Verify layer raw_bytes is a memoryview directly
    assert isinstance(layer0._raw_bytes, memoryview)

    # Forward pass executes and creates cached_weight
    x = torch.randn(2, 64)
    out = layer0(x)
    assert out.shape == (2, 64)
    assert layer0.cached_weight is not None

    # Test reclamation
    engine.reclaim_memory()
    assert len(engine.layers) == 0
    assert layer0.cached_weight is None, "Dequantized weight cache must be freed upon reclamation"

    engine.close()

