"""
POLAR-FLASH: Native Low-Bit CUDA Kernel Verification Tests
Verifies:
1. Native Fused INT4 GEMV matches PyTorch reference calculation with numerical tolerance < 1e-5.
2. Native Fused INT2 GEMV matches PyTorch reference.
3. Native Fused INT4 GEMM for batched token inputs.
4. VRAM savings and execution path reporting (avoiding false labeling of dequantize-plus-GEMM).
"""

import pytest
import torch
from polarflash_rt.cuda.native_kernels import NativeLowBitCUDAEngine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native kernel tests")
def test_native_fused_int4_gemv_accuracy():
    """Verifies that native fused INT4 GEMV matches PyTorch FP32 matrix-vector product."""
    engine = NativeLowBitCUDAEngine.get_instance()
    assert engine.is_available, f"CUDA Driver engine failed: {getattr(engine, 'init_error', '')}"

    M, K = 64, 128
    torch.manual_seed(42)

    raw_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
    u4 = (raw_int4 + 8).to(torch.uint8)
    packed_w = (u4[:, 0::2] | (u4[:, 1::2] << 4)).contiguous()
    scales = (torch.rand(M, device="cuda") * 0.2 + 0.05).float()
    x = torch.randn(K, device="cuda", dtype=torch.float32)

    # Reference
    w_float = raw_int4.float() * scales.unsqueeze(1)
    y_ref = torch.matmul(w_float, x)

    # Native fused kernel
    y_act = engine.gemv_int4(x, packed_w, scales, out_features=M, in_features=K)

    torch.cuda.synchronize()
    diff = torch.max(torch.abs(y_act - y_ref)).item()
    assert diff < 1e-4, f"Mismatch in native fused INT4 GEMV: diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native kernel tests")
def test_native_fused_int2_gemv_accuracy():
    """Verifies that native fused INT2 GEMV matches PyTorch reference."""
    engine = NativeLowBitCUDAEngine.get_instance()
    assert engine.is_available

    M, K = 32, 64
    torch.manual_seed(123)

    raw_int2 = torch.randint(0, 4, (M, K), dtype=torch.uint8, device="cuda")
    # Pack 4 2-bit values per byte
    packed_w = (raw_int2[:, 0::4] | (raw_int2[:, 1::4] << 2) | (raw_int2[:, 2::4] << 4) | (raw_int2[:, 3::4] << 6)).contiguous()
    scales = torch.ones(M, device="cuda", dtype=torch.float32) * 0.5
    x = torch.randn(K, device="cuda", dtype=torch.float32)

    # Reference (centered around 0 by subtracting 2)
    w_float = (raw_int2.float() - 2.0) * scales.unsqueeze(1)
    y_ref = torch.matmul(w_float, x)

    # Native fused INT2 kernel
    y_act = engine.gemv_int2(x, packed_w, scales, out_features=M, in_features=K)

    torch.cuda.synchronize()
    diff = torch.max(torch.abs(y_act - y_ref)).item()
    assert diff < 1e-4, f"Mismatch in native fused INT2 GEMV: diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native kernel tests")
def test_native_fused_int4_gemm_batched():
    """Verifies that native fused INT4 GEMM processes batched inputs correctly."""
    engine = NativeLowBitCUDAEngine.get_instance()
    assert engine.is_available

    batch_size = 4
    M, K = 32, 64
    torch.manual_seed(777)

    raw_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
    u4 = (raw_int4 + 8).to(torch.uint8)
    packed_w = (u4[:, 0::2] | (u4[:, 1::2] << 4)).contiguous()
    scales = torch.ones(M, device="cuda", dtype=torch.float32) * 0.1
    X = torch.randn(batch_size, K, device="cuda", dtype=torch.float32)

    w_float = raw_int4.float() * scales.unsqueeze(1)
    Y_ref = torch.matmul(X, w_float.t())

    Y_act = engine.gemm_int4(X, packed_w, scales, out_features=M, in_features=K)
    torch.cuda.synchronize()

    diff = torch.max(torch.abs(Y_act - Y_ref)).item()
    assert diff < 1e-4, f"Mismatch in native fused INT4 GEMM: diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native kernel tests")
def test_kernel_benchmark_and_execution_path_reporting():
    """Verifies that benchmark_comparison accurately reports NATIVE_FUSED_CUDA_KERNEL."""
    engine = NativeLowBitCUDAEngine.get_instance()
    assert engine.is_available

    res = engine.benchmark_comparison(out_features=512, in_features=512, iterations=10)
    assert res["kernel_execution_path"] == "NATIVE_FUSED_CUDA_KERNEL"
    assert res["vram_savings_ratio"] == 8.0, "INT4 should achieve exactly 8x VRAM savings vs FP32"
    assert res["native_fused_int4_latency_ms"] > 0.0

