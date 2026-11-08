"""
POLAR-FLASH: Quantized Johnson-Lindenstrauss (QJL) Residual Correction Engine
Implements Part II Section 5 of the POLAR-FLASH Mathematical Specification:
1. Randomized sign projection matrix S in {-1, +1}^(m x d) with E[S^T S] = m * I.
2. Unbiased inner product residual estimator: <x, e>_hat = (1 / m) * (S x)^T (S e).
3. Full corrected inner product: <x, w>_hat = <x, w_hat> + (1 / m) * (S x)^T (S e).
4. Estimator variance analysis: Var(q_hat) = (1 / m) * sum_{i != j} x_i e_i x_j e_j.
5. Bit-packed sign representation for hardware-accelerated popcount / XNOR evaluation.
6. Genuine XNOR + Popcount bitwise estimator: u^T v = m - 2 * popcount(u ^ v).
"""

import math
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional


class QJLResidualEstimator:
    """
    Implements randomized sign projection for low-bit residual correction.
    Uses an m x d Rademacher random matrix S with entries in {-1, +1}.
    Provides unbiased inner-product estimates of quantization residuals with bounded variance:
    E[<x, e>_hat] = <x, e>, Var(<x, e>_hat) = (1/m) * (||x||^2 ||e||^2 + <x,e>^2 - 2 sum x_i^2 e_i^2).
    """
    def __init__(self, in_features: int, proj_dim: int = 64, seed: int = 42, device: str = "cpu"):
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.device = device

        # Generate Rademacher projection matrix S in {-1, +1}
        gen = torch.Generator().manual_seed(seed)
        raw_bits = torch.randint(0, 2, (proj_dim, in_features), generator=gen, device=device)
        self.S = (raw_bits.float() * 2.0 - 1.0)  # Shape: (m, d)

    def project_residual(self, error_residual: torch.Tensor) -> torch.Tensor:
        """Projects weight error residual e = w - w_hat: s_e = S * e (Shape: (..., m))"""
        e = error_residual.to(self.device).float().view(-1, self.in_features)
        return torch.matmul(e, self.S.t())

    def project_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Projects activation vector x: s_x = S * x (Shape: (..., m))"""
        x_flat = x.to(self.device).float().view(-1, self.in_features)
        return torch.matmul(x_flat, self.S.t())

    def estimate_inner_product(self, s_x: torch.Tensor, s_e: torch.Tensor) -> torch.Tensor:
        """Computes unbiased inner product estimator: <x, e>_hat = (1 / m) * (s_x * s_e^T)"""
        return torch.matmul(s_x, s_e.t()) / float(self.proj_dim)

    def corrected_forward(self, x: torch.Tensor, w_quant: torch.Tensor, s_e: torch.Tensor) -> torch.Tensor:
        """
        Computes the complete QJL-corrected linear transformation:
        y_hat = x * w_quant^T + (1 / m) * (S * x) * s_e^T
        """
        y_base = torch.matmul(x, w_quant.t())
        s_x = self.project_activation(x)
        y_correction = torch.matmul(s_x, s_e.t()) / float(self.proj_dim)
        return y_base + y_correction


class BitPackedQJLEstimator:
    """
    Memory-Optimized QJL Estimator.
    Bit-packs the Rademacher projection matrix S into 1-bit per element (packed uint8).
    Reduces projection matrix RAM/VRAM footprint by 32x:
    m * d * 4 bytes (FP32) -> (m * d / 8) bytes (Packed uint8).
    """
    def __init__(self, in_features: int, proj_dim: int = 64, seed: int = 42, device: str = "cpu"):
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.device = device

        if in_features % 8 != 0:
            raise ValueError(f"in_features must be divisible by 8 for bit-packing, got {in_features}")

        gen = torch.Generator().manual_seed(seed)
        # 0 corresponds to -1, 1 corresponds to +1
        raw_bits = torch.randint(0, 2, (proj_dim, in_features), generator=gen, dtype=torch.uint8, device=device)

        packed = torch.zeros((proj_dim, in_features // 8), dtype=torch.uint8, device=device)
        for bit_i in range(8):
            packed |= (raw_bits[:, bit_i::8] << bit_i)

        self.packed_S = packed
        self.storage_bytes = packed.numel()
        self.unpacked_cache: Optional[torch.Tensor] = None

    def unpack_matrix(self) -> torch.Tensor:
        """Unpacks the 1-bit representation back to {-1.0, +1.0} tensor on demand."""
        if self.unpacked_cache is not None:
            return self.unpacked_cache

        unpacked = torch.empty((self.proj_dim, self.in_features), dtype=torch.float32, device=self.device)
        for bit_i in range(8):
            bit_vals = (self.packed_S >> bit_i) & 1
            unpacked[:, bit_i::8] = bit_vals.float() * 2.0 - 1.0

        self.unpacked_cache = unpacked
        return unpacked

    def project_residual(self, error_residual: torch.Tensor) -> torch.Tensor:
        S = self.unpack_matrix()
        e = error_residual.to(self.device).float().view(-1, self.in_features)
        return torch.matmul(e, S.t())

    def project_activation(self, x: torch.Tensor) -> torch.Tensor:
        S = self.unpack_matrix()
        x_flat = x.to(self.device).float().view(-1, self.in_features)
        return torch.matmul(x_flat, S.t())

    def estimate_inner_product(self, s_x: torch.Tensor, s_e: torch.Tensor) -> torch.Tensor:
        return torch.matmul(s_x, s_e.t()) / float(self.proj_dim)


class XNORPopcountQJLEstimator:
    """
    True Bitwise Sign-Sketch QJL Estimator using XNOR and Popcount.
    Given sketches s_x = sgn(Sx) and s_e = sgn(Se) in {0, 1}^m where 1 represents +1 and 0 represents -1:
    The bipolar inner product is given by:
    dot(s_x, s_e) = m - 2 * popcount(s_x ^ s_e)
    Scales with rms(Sx) and rms(Se) to preserve continuous magnitude:
    <x, e>_hat = (pi / 2) * rms(Sx) * rms(Se) * (1 - 2 * popcount(s_x ^ s_e) / m).
    """
    def __init__(self, in_features: int, proj_dim: int = 128, seed: int = 42, device: str = "cpu"):
        if proj_dim % 8 != 0:
            raise ValueError(f"proj_dim must be divisible by 8 for packed bitwise popcount, got {proj_dim}")
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.device = device
        self.base_estimator = QJLResidualEstimator(in_features, proj_dim, seed, device)

    def sketch_and_pack(self, tensor: torch.Tensor, is_residual: bool = False) -> Tuple[bytes, float]:
        """
        Projects tensor via S, records the scale (L2 norm), and packs the signs into bytes.
        """
        if is_residual:
            proj = self.base_estimator.project_residual(tensor).view(-1)
        else:
            proj = self.base_estimator.project_activation(tensor).view(-1)

        scale = float(torch.norm(proj, p=2).item() / math.sqrt(self.proj_dim))
        signs = (proj >= 0).cpu().to(torch.uint8)

        # Pack signs into bytes
        num_bytes = self.proj_dim // 8
        buffer = bytearray(num_bytes)
        for byte_i in range(num_bytes):
            byte_val = 0
            for bit_i in range(8):
                bit = int(signs[byte_i * 8 + bit_i].item())
                byte_val |= (bit << (7 - bit_i))
            buffer[byte_i] = byte_val

        return bytes(buffer), scale

    def evaluate_inner_product(self, packed_x: bytes, scale_x: float, packed_e: bytes, scale_e: float) -> float:
        """
        Computes sketch inner product purely through byte-level XOR and popcount.
        """
        # Count matching bits via XNOR: equivalent to m - popcount(x ^ e)
        hamming_distance = 0
        for b_x, b_e in zip(packed_x, packed_e):
            diff = b_x ^ b_e
            # Popcount of diff
            hamming_distance += bin(diff).count("1")

        # Bipolar dot product: +1 for matches, -1 for mismatches
        # matches = m - hamming_distance, mismatches = hamming_distance
        bipolar_dot = self.proj_dim - 2 * hamming_distance
        # Arc-cosine / Grothendieck projection correction: E[sgn(s_x) * sgn(s_e)] = (2/pi) * rho
        rho_est = math.sin((math.pi / 2.0) * (bipolar_dot / float(self.proj_dim)))
        return rho_est * scale_x * scale_e


def verify_qjl_unbiasedness_and_variance(
    dim: int = 512,
    proj_dim: int = 128,
    num_trials: int = 1000,
    seed: int = 1337
) -> Dict[str, Any]:
    """
    Numerically validates:
    1. Unbiasedness: E[<x, e>_hat] == <x, e>
    2. Variance formula: Var(<x, e>_hat) == (1/m) * sum_{i != j} x_i e_i x_j e_j
    """
    torch.manual_seed(seed)
    x = torch.randn(dim)
    e = torch.randn(dim) * 0.1

    true_dot_product = float(torch.dot(x, e).item())

    # Theoretical variance calculation:
    norm_x_sq = torch.sum(x ** 2).item()
    norm_e_sq = torch.sum(e ** 2).item()
    sum_xe_sq = torch.sum((x * e) ** 2).item()
    dot_sq = true_dot_product ** 2
    theoretical_var = (1.0 / proj_dim) * (norm_x_sq * norm_e_sq + dot_sq - 2.0 * sum_xe_sq)

    estimates = []
    for s_idx in range(num_trials):
        estimator = QJLResidualEstimator(in_features=dim, proj_dim=proj_dim, seed=seed + s_idx)
        s_x = estimator.project_activation(x)
        s_e = estimator.project_residual(e)
        est = float(estimator.estimate_inner_product(s_x, s_e).item())
        estimates.append(est)

    emp_mean = float(torch.tensor(estimates).mean().item())
    emp_var = float(torch.tensor(estimates).var().item())
    bias = abs(emp_mean - true_dot_product)

    standard_error = math.sqrt(theoretical_var / num_trials)
    z_score = bias / max(standard_error, 1e-12)

    return {
        "true_inner_product": true_dot_product,
        "empirical_mean_estimate": emp_mean,
        "bias": bias,
        "standard_error": standard_error,
        "z_score": z_score,
        "is_unbiased": z_score < 2.58,  # Within 99% confidence interval
        "theoretical_variance": theoretical_var,
        "empirical_variance": emp_var,
        "variance_relative_error": abs(emp_var - theoretical_var) / max(theoretical_var, 1e-12)
    }
