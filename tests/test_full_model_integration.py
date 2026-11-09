"""
POLAR-FLASH: Full Model Integration Test Suite
Validates Objective 4:
1. Ingestion of open-weight architecture checkpoint into .pflash container.
2. Compression of transformer linear projections (q, k, v, o, gate, up, down).
3. Loading and reconstruction of PFlashModelForCausalLM from memory map.
4. Tokenizer integration & autoregressive token generation.
5. Resident host RAM and GPU VRAM accounting.
6. Perplexity evaluation over evaluation text sequences.
"""

import os
import json
import math
import pytest
import torch
from safetensors.torch import save_file

from polarflash_core.model_integration import (
    ingest_checkpoint_to_pflash,
    load_pflash_causal_lm,
    PFlashModelForCausalLM
)


@pytest.fixture(scope="module")
def mock_open_weight_model(tmp_path_factory):
    """Creates a lightweight pretrained open-weight model matching Qwen2 architecture."""
    model_dir = str(tmp_path_factory.mktemp("open_weight_model"))
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    tensors = {
        "model.embed_tokens.weight": torch.randn(1000, 256).half(),
        "model.norm.weight": torch.ones(256).half()
    }
    for i in range(2):
        tensors[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(256).half()
        tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(256).half()
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(256, 256).half()
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(128, 256).half()
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(128, 256).half()
        tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(256, 256).half()
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(512, 256).half()
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(512, 256).half()
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(256, 512).half()

    save_file(tensors, os.path.join(model_dir, "model.safetensors"))
    return model_dir


def test_checkpoint_ingestion_and_quantization(mock_open_weight_model, tmp_path):
    """Tests ingestion and compression of transformer linear projections into .pflash container."""
    output_pflash = str(tmp_path / "model.pflash")
    stats = ingest_checkpoint_to_pflash(
        model_dir=mock_open_weight_model,
        output_pflash=output_pflash,
        backend="polar",
        max_layers=2,
        target_bpw=2.0
    )

    assert os.path.exists(output_pflash)
    assert stats["total_bytes"] > 0
    assert stats["quantized_layers_count"] == 14  # 7 projections x 2 layers
    assert stats["effective_bpw"] > 0


def test_model_loading_and_autoregressive_forward(mock_open_weight_model, tmp_path):
    """Tests zero-copy loading of PFlashModelForCausalLM and causal attention forward pass."""
    output_pflash = str(tmp_path / "model.pflash")
    ingest_checkpoint_to_pflash(mock_open_weight_model, output_pflash, backend="polar", max_layers=2)

    model = load_pflash_causal_lm(output_pflash, device="cpu")
    assert isinstance(model, PFlashModelForCausalLM)
    assert len(model.layers) == 2

    # Forward pass with causal attention
    input_ids = torch.randint(0, 1000, (2, 16))
    logits, _ = model(input_ids)
    assert logits.shape == (2, 16, 1000)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_memory_measurement(mock_open_weight_model, tmp_path):
    """Tests exact resident Host RAM and VRAM measurement."""
    output_pflash = str(tmp_path / "model.pflash")
    ingest_checkpoint_to_pflash(mock_open_weight_model, output_pflash, backend="polar", max_layers=2)

    model = load_pflash_causal_lm(output_pflash, device="cpu")
    mem_stats = model.measure_memory()

    assert "resident_host_ram_mb" in mem_stats
    assert mem_stats["resident_host_ram_mb"] > 0
    assert "total_parameters" in mem_stats
    assert mem_stats["total_parameters"] > 0
    assert "uncompressed_fp16_baseline_mb" in mem_stats


def test_autoregressive_token_generation(mock_open_weight_model, tmp_path):
    """Tests token-by-token generation with greedy decoding."""
    output_pflash = str(tmp_path / "model.pflash")
    ingest_checkpoint_to_pflash(mock_open_weight_model, output_pflash, backend="polar", max_layers=2)
    model = load_pflash_causal_lm(output_pflash, device="cpu")

    class SimpleTokenizer:
        eos_token_id = 999
        def __call__(self, text, return_tensors="pt"):
            # Dummy tokens from string
            tokens = [abs(hash(w)) % 990 for w in text.split()]
            return {"input_ids": torch.tensor([tokens])}
        def decode(self, tokens, skip_special_tokens=True):
            return " ".join([f"tok_{t.item()}" for t in tokens])

    tok = SimpleTokenizer()
    gen = model.generate("Hello artificial intelligence", tokenizer=tok, max_new_tokens=5, top_k=0)
    assert isinstance(gen, str)
    assert len(gen.split()) > 3


def test_perplexity_evaluation(mock_open_weight_model, tmp_path):
    """Tests cross-entropy loss and perplexity calculation."""
    output_pflash = str(tmp_path / "model.pflash")
    ingest_checkpoint_to_pflash(mock_open_weight_model, output_pflash, backend="polar", max_layers=2)
    model = load_pflash_causal_lm(output_pflash, device="cpu")

    class SimpleTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
            tokens = [abs(hash(w)) % 990 for w in text.split()]
            return {"input_ids": torch.tensor([tokens])}

    tok = SimpleTokenizer()
    ppl = model.evaluate_perplexity(["test prompt sequence for evaluation", "second sentence here"], tokenizer=tok)
    assert "mean_cross_entropy" in ppl
    assert "perplexity" in ppl
    assert ppl["perplexity"] > 0
    assert ppl["total_tokens_evaluated"] > 0

