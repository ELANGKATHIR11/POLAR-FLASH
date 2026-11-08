"""
POLAR-FLASH: E8 Lattice Quantization Engine
Implements Part II Section 4 of the POLAR-FLASH Mathematical Specification:
1. Conway-Sloane fast O(d) nearest lattice point projection for Gosset Lattice E8:
   E8 = D8 U (D8 + 0.5 * 1), where D8 = {x in Z^8 : sum(x_i) is even}.
2. Optimal scale factor search: s* = argmin_s sum ||x_i - s * Q_E8(x_i / s)||_2^2.
3. Quantization error, SNR, and effective bitrate accounting.
"""

import math
import torch
from typing import Dict, Any, Tuple


def quantize_d8(x: torch.Tensor) -> torch.Tensor:
    """
    Finds the nearest lattice point in D8 = {x in Z^8 : sum(x_i) = 0 mod 2}.
    Fast algorithm (Conway & Sloane, 1982):
    1. Round each coordinate to the nearest integer f(x_i).
    2. If the sum of coordinates is even, f(x) is the closest point in D8.
    3. If the sum is odd, flip the coordinate with the largest rounding error |x_i - f(x_i)|
       to the next nearest integer (i.e. add or subtract 1 depending on sign of error).
    Input shape: (..., 8)
    """
    f = torch.round(x)
    diff = x - f  # rounding error in [-0.5, 0.5]
    abs_diff = torch.abs(diff)

    # Check parity of coordinate sums
    sum_f = torch.sum(f, dim=-1, keepdim=True).long()
    is_odd = (sum_f % 2 != 0)

    # Find the index of the coordinate with maximum rounding error along last dim
    worst_idx = torch.argmax(abs_diff, dim=-1, keepdim=True)

    # Delta to flip coordinate: sign of error (or +1 if diff >= 0, -1 if diff < 0)
    flip_delta = torch.where(diff >= 0, torch.ones_like(diff), -torch.ones_like(diff))

    # Apply flip only to the worst coordinate where parity is odd
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, worst_idx, True)

    adjusted = torch.where(is_odd & mask, f + flip_delta, f)
    return adjusted


def quantize_e8(x: torch.Tensor) -> torch.Tensor:
    """
    Finds the nearest lattice point in Gosset lattice E8:
    E8 = D8 U (D8 + 0.5 * 1_8).
    Input shape: (..., 8)
    Complexity: O(8) operations per vector.
    """
    # Candidate 1: Nearest point in D8
    cand1 = quantize_d8(x)
    dist1 = torch.sum((x - cand1) ** 2, dim=-1, keepdim=True)

    # Candidate 2: Nearest point in D8 + 0.5 * 1_8
    shift = 0.5
    cand2 = quantize_d8(x - shift) + shift
    dist2 = torch.sum((x - cand2) ** 2, dim=-1, keepdim=True)

    # Choose candidate with minimum Euclidean distance
    best = torch.where(dist1 <= dist2, cand1, cand2)
    return best


class E8LatticeQuantizer:
    """
    E8 Lattice Vector Quantizer with optimal scale factor search.
    Groups weights into 8-dimensional blocks and maps each block to the scaled E8 lattice:
    x_hat = s * Q_E8(x / s)
    """
    def __init__(self, search_steps: int = 16, epsilon: float = 1e-12):
        self.search_steps = search_steps
        self.epsilon = epsilon

    def optimize_scale(self, x_blocks: torch.Tensor) -> float:
        """
        Finds optimal scale s* = argmin_s ||x - s * Q_E8(x / s)||^2
        via golden section / grid search around rms(x).
        x_blocks shape: (num_blocks, 8)
        """
        rms = float(torch.sqrt(torch.mean(x_blocks ** 2)).item())
        if rms <= self.epsilon:
            return 1.0

        # Optimal lattice scale typically lies between [0.2 * rms, 2.5 * rms]
        best_scale = rms
        best_loss = float("inf")

        scales = torch.linspace(0.2 * rms, 2.5 * rms, steps=self.search_steps, device=x_blocks.device)
        for s_val in scales:
            s = float(s_val.item())
            if s <= self.epsilon:
                continue
            scaled_x = x_blocks / s
            q_scaled = quantize_e8(scaled_x)
            reconstructed = q_scaled * s
            loss = float(torch.mean((x_blocks - reconstructed) ** 2).item())
            if loss < best_loss:
                best_loss = loss
                best_scale = s

        return best_scale

    def quantize(self, x: torch.Tensor, scale: float = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantizes tensor x by reshaping into 8-dimensional blocks.
        Length must be divisible by 8 (zero-padded if necessary).
        """
        orig_shape = x.shape
        flat = x.view(-1)
        rem = flat.numel() % 8
        if rem != 0:
            pad = 8 - rem
            flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
        else:
            pad = 0

        blocks = flat.view(-1, 8).float()

        if scale is None:
            scale = self.optimize_scale(blocks)

        scaled_blocks = blocks / max(scale, self.epsilon)
        q_blocks = quantize_e8(scaled_blocks)
        x_hat_blocks = q_blocks * scale

        x_hat_flat = x_hat_blocks.view(-1)
        if pad > 0:
            x_hat_flat = x_hat_flat[:-pad]

        x_hat = x_hat_flat.reshape(orig_shape)

        # Metrics
        mse = float(torch.mean((x - x_hat) ** 2).item())
        var_x = float(torch.var(x).item())
        snr_db = 10.0 * math.log10(max(var_x, 1e-15) / max(mse, 1e-15))

        metadata = {
            "optimal_scale": scale,
            "mse_distortion": mse,
            "snr_db": snr_db,
            "dimension": 8,
            "sphere_packing_density_e8": math.pi ** 4 / 384.0,  # ~0.25367
            "padding_added": pad
        }

        return x_hat, metadata
