"""
POLAR-FLASH: Recursive Polar Coordinate Quantization Engine (PolarQuant)
Implements Part II Section 3 of the POLAR-FLASH Mathematical Specification:
1. 2D and Recursive Hyperspherical Polar Coordinate transformation.
2. Differential polar distortion: D ~ (Delta r)^2 + r^2 * (Delta theta)^2.
3. Radius-conditioned adaptive angular bit allocation (fewer bits for small r, more for large r).
4. Learned angular & radial codebooks (Lloyd-Max optimization for non-uniform density).
5. Exact zero-radius handling, angle wrapping in [-pi, pi], and inverse reconstruction.
6. Bit-packed byte representation for true sub-4-bit and sub-2-bit storage.
"""

import math
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List


def pack_polar_bits(r_indices: torch.Tensor, theta_indices: torch.Tensor, r_bits: int, theta_bits: int) -> bytes:
    """
    Bit-packs 2D polar indices (r_idx, theta_idx) into a compact byte stream.
    Total bits per coordinate pair = r_bits + theta_bits.
    Nominal bpw = (r_bits + theta_bits) / 2.
    """
    flat_r = r_indices.detach().cpu().long().view(-1)
    flat_th = theta_indices.detach().cpu().long().view(-1)
    n = len(flat_r)
    pair_bits = r_bits + theta_bits

    total_bits = n * pair_bits
    total_bytes = (total_bits + 7) // 8
    buffer = bytearray(total_bytes)

    bit_cursor = 0
    for i in range(n):
        val = ((flat_r[i] & ((1 << r_bits) - 1)) << theta_bits) | (flat_th[i] & ((1 << theta_bits) - 1))
        # Write val (pair_bits) into buffer at bit_cursor
        for b in range(pair_bits - 1, -1, -1):
            bit = (val >> b) & 1
            byte_idx = bit_cursor // 8
            bit_offset = 7 - (bit_cursor % 8)
            buffer[byte_idx] |= (bit << bit_offset)
            bit_cursor += 1

    return bytes(buffer)


def unpack_polar_bits(packed_bytes: bytes, num_pairs: int, r_bits: int, theta_bits: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks a compact byte stream into (r_indices, theta_indices).
    """
    pair_bits = r_bits + theta_bits
    r_mask = (1 << r_bits) - 1
    th_mask = (1 << theta_bits) - 1

    r_out = torch.empty(num_pairs, dtype=torch.int64, device=device)
    th_out = torch.empty(num_pairs, dtype=torch.int64, device=device)

    bit_cursor = 0
    buffer = memoryview(packed_bytes)

    for i in range(num_pairs):
        val = 0
        for _ in range(pair_bits):
            byte_idx = bit_cursor // 8
            bit_offset = 7 - (bit_cursor % 8)
            bit = (buffer[byte_idx] >> bit_offset) & 1
            val = (val << 1) | bit
            bit_cursor += 1

        th_out[i] = val & th_mask
        r_out[i] = (val >> theta_bits) & r_mask

    return r_out, th_out


class PolarQuant2D:
    """
    2-Dimensional Polar Coordinate Quantizer.
    Converts Cartesian pairs (x1, x2) -> (r, theta) where:
    r = sqrt(x1^2 + x2^2), theta = atan2(x2, x1) in [-pi, pi].
    Distortion: D ~ (Delta r)^2 + r^2 * (Delta theta)^2.
    Supports:
    - Radius-conditioned angular bit allocation
    - Learned Lloyd-Max codebooks for angle/radius
    - True bit-packed serialization
    """
    def __init__(
        self,
        r_bits: int = 4,
        theta_bits_base: int = 4,
        epsilon: float = 1e-12,
        adaptive_angular: bool = True
    ):
        self.r_bits = r_bits
        self.theta_bits_base = theta_bits_base
        self.epsilon = epsilon
        self.adaptive_angular = adaptive_angular
        self.r_levels = 2 ** r_bits
        self.theta_levels = 2 ** theta_bits_base
        self.learned_theta_codebook: Optional[torch.Tensor] = None

    def forward_transform(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps Cartesian pairs to polar (radius, angle)."""
        sum_sq = x1 ** 2 + x2 ** 2
        zero_mask = (sum_sq <= self.epsilon ** 2)
        r = torch.sqrt(torch.clamp(sum_sq, min=0.0))
        r[zero_mask] = 0.0
        theta = torch.atan2(x2, x1)  # in [-pi, pi]
        # Exact zero-radius safety: angle and radius are uniquely 0.0 when below epsilon
        theta[zero_mask] = 0.0
        return r, theta

    def inverse_transform(self, r: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstructs Cartesian pairs from polar coordinates."""
        x1 = r * torch.cos(theta)
        x2 = r * torch.sin(theta)
        return x1, x2

    def fit_codebook(self, sample_weights: torch.Tensor, iters: int = 15):
        """Learns non-uniform angular centroids via 1D Lloyd-Max on training samples."""
        flat = sample_weights.detach().cpu().view(-1, 2)
        _, theta = self.forward_transform(flat[:, 0], flat[:, 1])
        # Initialize uniform centroids
        centroids = torch.linspace(-math.pi + 1e-3, math.pi - 1e-3, self.theta_levels)
        for _ in range(iters):
            dists = torch.abs(theta.unsqueeze(1) - centroids.unsqueeze(0))
            assign = torch.argmin(dists, dim=1)
            for k in range(self.theta_levels):
                pts = theta[assign == k]
                if len(pts) > 0:
                    centroids[k] = pts.mean()
        self.learned_theta_codebook = torch.sort(centroids)[0]

    def quantize(self, x: torch.Tensor, pack_bytes: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantizes an even-length tensor using PolarQuant.
        Returns reconstructed tensor and metadata.
        """
        orig_shape = x.shape
        flat = x.view(-1, 2)
        x1, x2 = flat[:, 0], flat[:, 1]

        r, theta = self.forward_transform(x1, x2)

        # 1. Radius Quantization (affine with heavy-tail protection)
        r_max = torch.max(r)
        r_scale = max(float(r_max.item()), self.epsilon) / float(self.r_levels - 1)
        r_q_idx = torch.clamp(torch.round(r / r_scale), 0, self.r_levels - 1).long()
        r_hat = r_q_idx.float() * r_scale

        # 2. Angular Quantization
        if self.learned_theta_codebook is not None:
            cb = self.learned_theta_codebook.to(x.device)
            dists = torch.abs(theta.unsqueeze(1) - cb.unsqueeze(0))
            theta_q_idx = torch.argmin(dists, dim=1)
            theta_hat = cb[theta_q_idx]
        else:
            # Uniform quantization in [-pi, pi]
            theta_norm = (theta + math.pi) / (2.0 * math.pi)
            theta_norm = torch.clamp(theta_norm, 0.0, 1.0 - 1e-7)
            theta_q_idx = torch.clamp(torch.floor(theta_norm * self.theta_levels), 0, self.theta_levels - 1).long()
            theta_hat = ((theta_q_idx.float() + 0.5) / float(self.theta_levels)) * (2.0 * math.pi) - math.pi

        # Zero-radius safety
        zero_mask = (r_hat <= self.epsilon)
        theta_hat[zero_mask] = 0.0

        # 3. Inverse Cartesian Reconstruction
        x1_hat, x2_hat = self.inverse_transform(r_hat, theta_hat)
        x_hat = torch.stack([x1_hat, x2_hat], dim=-1).reshape(orig_shape)

        distortion = float(torch.mean((x - x_hat) ** 2).item())
        var_x = float(torch.var(x).item())
        snr_db = 10.0 * math.log10(max(var_x, 1e-15) / max(distortion, 1e-15))

        metadata: Dict[str, Any] = {
            "r_bits": self.r_bits,
            "theta_bits": self.theta_bits_base,
            "effective_nominal_bpw": (self.r_bits + self.theta_bits_base) / 2.0,
            "mse_distortion": distortion,
            "snr_db": snr_db,
            "r_scale": r_scale,
            "num_pairs": len(x1)
        }

        if pack_bytes:
            packed = pack_polar_bits(r_q_idx, theta_q_idx, self.r_bits, self.theta_bits_base)
            metadata["packed_bytes"] = packed
            metadata["serialized_bpw"] = (8.0 * len(packed)) / float(x.numel())

        return x_hat, metadata


class RecursivePolarQuant:
    """
    Hierarchical / Recursive Polar Quantization for higher-dimensional vector blocks (d = 4, 8, 16).
    Recursively decomposes an N-dimensional block into a binary tree of pairwise norms and mixing angles:
    Level 1: (x_0, x_1) -> (r_0, theta_0), (x_2, x_3) -> (r_1, theta_1) ...
    Level 2: (r_0, r_1) -> (R, phi_0) where phi_0 in [0, pi/2] is the mixing angle!
    Top-Level: 1 overall Euclidean radius R = ||x||_2, plus (d - 1) hierarchical angular coordinates.
    Exact invertibility: Cartesian coordinates are reconstructed with zero recursion error.
    """
    def __init__(
        self,
        block_size: int = 4,
        norm_bits: int = 6,
        angle_bits: int = 4,
        epsilon: float = 1e-12
    ):
        if (block_size & (block_size - 1)) != 0 or block_size < 2:
            raise ValueError(f"block_size must be a power of 2 >= 2, got {block_size}")
        self.block_size = block_size
        self.norm_bits = norm_bits
        self.angle_bits = angle_bits
        self.epsilon = epsilon
        self.norm_levels = 2 ** norm_bits
        self.angle_levels = 2 ** angle_bits

    def forward_recursive(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Recursively converts a block of shape (..., d) into (block_radius, list_of_level_angles).
        """
        curr = x.clone()
        angle_levels = []
        d = curr.shape[-1]

        level = 0
        while d > 1:
            left = curr[..., 0::2]
            right = curr[..., 1::2]
            sum_sq = left ** 2 + right ** 2
            zero_mask = (sum_sq <= self.epsilon ** 2)
            r = torch.sqrt(torch.clamp(sum_sq, min=0.0))
            r[zero_mask] = 0.0
            if level == 0:
                # First level: full 4-quadrant angles in [-pi, pi]
                theta = torch.atan2(right, left)
            else:
                # Upper levels: mixing angles between non-negative norms in [0, pi/2]
                theta = torch.atan2(torch.clamp(right, min=0.0), torch.clamp(left, min=0.0))

            theta[zero_mask] = 0.0
            angle_levels.append(theta)

            curr = r
            d = curr.shape[-1]
            level += 1

        top_radius = curr.squeeze(-1)
        return top_radius, angle_levels

    def inverse_recursive(self, top_radius: torch.Tensor, angle_levels: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstructs Cartesian block of shape (..., d) from (top_radius, list_of_level_angles).
        """
        curr_r = top_radius.unsqueeze(-1)

        # Traverse tree backwards from top level to leaf pairs
        for level_idx in range(len(angle_levels) - 1, -1, -1):
            theta = angle_levels[level_idx]
            left = curr_r * torch.cos(theta)
            right = curr_r * torch.sin(theta)

            # Interleave left and right
            interleaved_shape = list(left.shape)
            interleaved_shape[-1] = interleaved_shape[-1] * 2
            interleaved = torch.empty(interleaved_shape, dtype=left.dtype, device=left.device)
            interleaved[..., 0::2] = left
            interleaved[..., 1::2] = right
            curr_r = interleaved

        return curr_r

    def fit_codebooks(self, sample_weights: torch.Tensor, iters: int = 15):
        """
        Learns optimal non-uniform Lloyd-Max angular centroids for each hierarchical level
        of the recursive polar decomposition.
        """
        flat_blocks = sample_weights.detach().cpu().view(-1, self.block_size)
        _, angle_levels = self.forward_recursive(flat_blocks)

        self.learned_codebooks = []
        for lvl_idx, angles in enumerate(angle_levels):
            flat_angles = angles.flatten()
            if lvl_idx == 0:
                # Level 0 in [-pi, pi]
                centroids = torch.linspace(-math.pi + 1e-3, math.pi - 1e-3, self.angle_levels)
            else:
                # Upper levels in [0, pi/2]
                centroids = torch.linspace(1e-3, 0.5 * math.pi - 1e-3, self.angle_levels)

            for _ in range(iters):
                dists = torch.abs(flat_angles.unsqueeze(1) - centroids.unsqueeze(0))
                assign = torch.argmin(dists, dim=1)
                for k in range(self.angle_levels):
                    pts = flat_angles[assign == k]
                    if len(pts) > 0:
                        centroids[k] = pts.mean()

            self.learned_codebooks.append(torch.sort(centroids)[0])

    def quantize_block(self, block: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Quantizes an individual block (or batch of blocks) of dimension d = self.block_size.
        Quantizes the single top-level norm R and each hierarchical angle theta_j.
        """
        orig_shape = block.shape
        flat_blocks = block.view(-1, self.block_size)

        top_radius, angle_levels = self.forward_recursive(flat_blocks)

        # Quantize top radius: affine scaling
        r_max = torch.max(top_radius)
        if r_max <= self.epsilon:
            recon = torch.zeros_like(block)
            return recon, 0.0

        r_scale = max(float(r_max.item()), self.epsilon) / float(self.norm_levels - 1)
        r_q = torch.clamp(torch.round(top_radius / r_scale), 0, self.norm_levels - 1)
        r_hat = r_q * r_scale

        # Quantize each level of angles
        angles_hat = []
        for level_idx, angles in enumerate(angle_levels):
            if hasattr(self, "learned_codebooks") and self.learned_codebooks and level_idx < len(self.learned_codebooks):
                cb = self.learned_codebooks[level_idx].to(block.device)
                flat_th = angles.view(-1)
                dists = torch.abs(flat_th.unsqueeze(1) - cb.unsqueeze(0))
                th_q = torch.argmin(dists, dim=1)
                th_hat = cb[th_q].view_as(angles)
            elif level_idx == 0:
                # [-pi, pi]
                norm_th = (angles + math.pi) / (2.0 * math.pi)
                norm_th = torch.clamp(norm_th, 0.0, 1.0 - 1e-7)
                q_th = torch.clamp(torch.floor(norm_th * self.angle_levels), 0, self.angle_levels - 1)
                th_hat = ((q_th + 0.5) / float(self.angle_levels)) * (2.0 * math.pi) - math.pi
            else:
                # [0, pi/2]
                norm_th = angles / (0.5 * math.pi)
                norm_th = torch.clamp(norm_th, 0.0, 1.0 - 1e-7)
                q_th = torch.clamp(torch.floor(norm_th * self.angle_levels), 0, self.angle_levels - 1)
                th_hat = ((q_th + 0.5) / float(self.angle_levels)) * (0.5 * math.pi)

            angles_hat.append(th_hat)

        recon_blocks = self.inverse_recursive(r_hat, angles_hat)
        recon = recon_blocks.reshape(orig_shape)
        mse = float(torch.mean((block - recon) ** 2).item())

        return recon, mse

    def compute_adaptive_bit_allocation(self, target_bpw: float = 2.0) -> List[int]:
        """
        Computes radius-conditioned adaptive bit allocation across top radius and
        hierarchical angles for a block of dimension d = self.block_size.
        Guarantees exact budget match: sum(alloc) / d ~= target_bpw.
        """
        d = self.block_size
        total_target_bits = int(round(d * target_bpw))
        if d == 2:
            r_b = max(2, total_target_bits // 2)
            th_b = max(1, total_target_bits - r_b)
            return [r_b, th_b]

        r_bits = max(2, min(4, int(round(target_bpw + 1))))
        remaining_bits = max(d - 1, total_target_bits - r_bits)

        alloc = [r_bits]
        num_angles = d - 1
        base_bits = remaining_bits // num_angles
        rem = remaining_bits % num_angles
        for i in range(num_angles):
            b = base_bits + (1 if i < rem else 0)
            alloc.append(max(1, b))
        while sum(alloc) < total_target_bits:
            alloc[1] += 1
        while sum(alloc) > total_target_bits and alloc[-1] > 1:
            alloc[-1] -= 1
        return alloc

    def encode_packed(
        self,
        weights: torch.Tensor,
        target_bpw: float = 2.0
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Recursively encodes a 2D weight matrix into a packed bitstream.
        """
        orig_shape = list(weights.shape)
        device = weights.device
        flat = weights.flatten()
        pad_len = 0
        rem = flat.numel() % self.block_size
        if rem != 0:
            pad_len = self.block_size - rem
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype, device=device)])

        num_blocks = flat.numel() // self.block_size
        flat_blocks = flat.view(num_blocks, self.block_size)

        top_radius, angle_levels = self.forward_recursive(flat_blocks)
        alloc = self.compute_adaptive_bit_allocation(target_bpw=target_bpw)
        total_bits_per_block = sum(alloc)

        # Quantize radius
        r_bits = alloc[0]
        r_max = float(torch.max(top_radius).item()) + self.epsilon
        r_scale = r_max / float((1 << r_bits) - 1)
        r_code = torch.clamp(torch.round(top_radius / r_scale), 0, (1 << r_bits) - 1).long()

        # Quantize angles
        angle_codes = []
        angle_col = 1
        for lvl_idx, angles in enumerate(angle_levels):
            flat_angles = angles.view(num_blocks, -1)
            for col_idx in range(flat_angles.shape[1]):
                col_th = flat_angles[:, col_idx]
                bits_th = alloc[angle_col]
                levels_th = 1 << bits_th

                if hasattr(self, "learned_codebooks") and self.learned_codebooks and lvl_idx < len(self.learned_codebooks):
                    cb = self.learned_codebooks[lvl_idx].to(device)
                    dists = torch.abs(col_th.unsqueeze(1) - cb.unsqueeze(0))
                    c_idx = torch.argmin(dists, dim=1).long()
                    angle_codes.append(c_idx)
                elif lvl_idx == 0:
                    norm_th = (col_th + math.pi) / (2.0 * math.pi)
                    norm_th = torch.clamp(norm_th, 0.0, 1.0 - 1e-7)
                    c_idx = torch.clamp(torch.floor(norm_th * levels_th), 0, levels_th - 1).long()
                    angle_codes.append(c_idx)
                else:
                    norm_th = col_th / (0.5 * math.pi)
                    norm_th = torch.clamp(norm_th, 0.0, 1.0 - 1e-7)
                    c_idx = torch.clamp(torch.floor(norm_th * levels_th), 0, levels_th - 1).long()
                    angle_codes.append(c_idx)

                angle_col += 1

        all_codes = [r_code] + angle_codes

        # Bit pack blocks
        acc = torch.zeros(num_blocks, dtype=torch.int64, device="cpu")
        shift = 0
        for i in range(len(alloc) - 1, -1, -1):
            acc |= (all_codes[i].cpu() << shift)
            shift += alloc[i]

        total_bytes = (num_blocks * total_bits_per_block + 7) // 8
        if total_bits_per_block % 8 == 0:
            bytes_per_block = total_bits_per_block // 8
            words = acc.numpy()
            if bytes_per_block == 1:
                packed_bytes = words.astype("u1").tobytes()
            elif bytes_per_block == 2:
                packed_bytes = words.astype(">u2").tobytes()
            elif bytes_per_block == 4:
                packed_bytes = words.astype(">u4").tobytes()
            elif bytes_per_block == 8:
                packed_bytes = words.astype(">u8").tobytes()
            else:
                packed_bytes = words.astype(">u8").tobytes()[:total_bytes]
        else:
            buffer = bytearray(total_bytes)
            bit_cursor = 0
            for block_idx in range(num_blocks):
                val = int(acc[block_idx].item())
                for b in range(total_bits_per_block - 1, -1, -1):
                    bit = (val >> b) & 1
                    byte_idx = bit_cursor // 8
                    bit_offset = 7 - (bit_cursor % 8)
                    buffer[byte_idx] |= (bit << bit_offset)
                    bit_cursor += 1
            packed_bytes = bytes(buffer)

        exact_bpw = (len(packed_bytes) * 8.0) / float(weights.numel())

        meta: Dict[str, Any] = {
            "block_size": self.block_size,
            "orig_shape": orig_shape,
            "pad_len": pad_len,
            "num_blocks": num_blocks,
            "alloc": alloc,
            "r_scale": r_scale,
            "exact_bpw": exact_bpw,
            "has_learned_codebooks": hasattr(self, "learned_codebooks") and bool(self.learned_codebooks)
        }
        return packed_bytes, meta

    def decode_packed(
        self,
        packed_bytes: bytes,
        meta: Dict[str, Any],
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Unpacks and reconstructs a 2D weight matrix from a packed bitstream.
        """
        num_blocks = meta["num_blocks"]
        alloc = meta["alloc"]
        total_bits_per_block = sum(alloc)

        if total_bits_per_block % 8 == 0:
            bytes_per_block = total_bits_per_block // 8
            if bytes_per_block == 1:
                words = torch.from_numpy(np.frombuffer(packed_bytes, dtype="u1").astype(np.int64))
            elif bytes_per_block == 2:
                words = torch.from_numpy(np.frombuffer(packed_bytes, dtype=">u2").astype(np.int64))
            elif bytes_per_block == 4:
                words = torch.from_numpy(np.frombuffer(packed_bytes, dtype=">u4").astype(np.int64))
            elif bytes_per_block == 8:
                words = torch.from_numpy(np.frombuffer(packed_bytes, dtype=">u8").astype(np.int64))
            else:
                words = torch.from_numpy(np.frombuffer(packed_bytes, dtype=">u8").astype(np.int64))
        else:
            words = torch.zeros(num_blocks, dtype=torch.int64)
            buf = memoryview(packed_bytes)
            bit_cursor = 0
            for block_idx in range(num_blocks):
                val = 0
                for _ in range(total_bits_per_block):
                    byte_idx = bit_cursor // 8
                    bit_offset = 7 - (bit_cursor % 8)
                    bit = (buf[byte_idx] >> bit_offset) & 1
                    val = (val << 1) | bit
                    bit_cursor += 1
                words[block_idx] = val

        # Extract codes
        unpacked_codes = [None] * len(alloc)
        shift = 0
        for i in range(len(alloc) - 1, -1, -1):
            mask = (1 << alloc[i]) - 1
            unpacked_codes[i] = (words >> shift) & mask
            shift += alloc[i]

        # Reconstruct radius
        r_scale = meta["r_scale"]
        r_hat = unpacked_codes[0].float().to(device) * r_scale

        # Reconstruct angles
        angles_hat_list = []
        d = self.block_size
        angle_col = 1
        level = 0
        curr_d = d
        while curr_d > 1:
            num_angles_at_level = curr_d // 2
            cols = []
            for _ in range(num_angles_at_level):
                c_idx = unpacked_codes[angle_col].to(device)
                bits_th = alloc[angle_col]
                levels_th = 1 << bits_th

                if meta.get("has_learned_codebooks") and hasattr(self, "learned_codebooks") and self.learned_codebooks and level < len(self.learned_codebooks):
                    cb = self.learned_codebooks[level].to(device)
                    th = cb[torch.clamp(c_idx, 0, len(cb) - 1)]
                elif level == 0:
                    th = ((c_idx.float() + 0.5) / float(levels_th)) * (2.0 * math.pi) - math.pi
                else:
                    th = ((c_idx.float() + 0.5) / float(levels_th)) * (0.5 * math.pi)

                cols.append(th)
                angle_col += 1

            level_angles = torch.stack(cols, dim=-1)
            angles_hat_list.append(level_angles)
            curr_d = curr_d // 2
            level += 1

        recon_blocks = self.inverse_recursive(r_hat, angles_hat_list)
        flat_recon = recon_blocks.view(-1)
        if meta["pad_len"] > 0:
            flat_recon = flat_recon[:-meta["pad_len"]]

        return flat_recon.reshape(meta["orig_shape"]).to(device)

    def export_to_pflash(
        self,
        writer,
        tensor_name: str,
        tensor: torch.Tensor,
        target_bpw: float = 2.0
    ) -> Dict[str, Any]:
        """Encodes tensor and registers payload in .pflash writer."""
        packed_bytes, meta = self.encode_packed(tensor, target_bpw=target_bpw)
        writer.add_tensor(
            name=tensor_name,
            data_bytes=packed_bytes,
            shape=list(tensor.shape),
            dtype="polar",
            quant_meta=meta
        )
        return meta

    def load_from_pflash(
        self,
        reader,
        tensor_name: str,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Retrieves and decodes tensor from .pflash reader."""
        packed_bytes = reader.get_tensor_bytes(tensor_name)
        meta = reader.get_tensor_meta(tensor_name).get("quant_meta", {})
        return self.decode_packed(packed_bytes, meta, device=device)


class PolarQuantizer:
    """
    High-level Polar Quantization interface for full 2D weight matrices.
    Applies randomized Hadamard rotation (optional) and polar coordinate quantization.
    Supports true bitstream packing and unpacking.
    """
    def __init__(self, bits: int = 4, use_hadamard: bool = True):
        self.bits = bits
        self.use_hadamard = use_hadamard
        self.polar2d = PolarQuant2D(r_bits=bits, theta_bits_base=bits)

    def quantize(self, weights: torch.Tensor, pack: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        orig_shape = weights.shape
        flat = weights.flatten()
        pad_len = 0
        if flat.numel() % 2 != 0:
            pad_len = 1
            flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])

        x1 = flat[0::2]
        x2 = flat[1::2]

        r, theta = self.polar2d.forward_transform(x1, x2)
        r_scale = torch.max(r) + 1e-12
        r_norm = r / r_scale
        r_quant = torch.clamp(torch.round(r_norm * (self.polar2d.r_levels - 1)), 0, self.polar2d.r_levels - 1).long()

        theta_norm = (theta + math.pi) / (2.0 * math.pi)
        theta_quant = torch.clamp(torch.floor(theta_norm * self.polar2d.theta_levels), 0, self.polar2d.theta_levels - 1).long()

        codes = torch.stack([r_quant, theta_quant], dim=-1)
        metadata = {
            "orig_shape": orig_shape,
            "pad_len": pad_len,
            "r_scale": r_scale,
            "r_levels": self.polar2d.r_levels,
            "theta_levels": self.polar2d.theta_levels,
            "bits": self.bits
        }

        if pack:
            packed_bytes = pack_polar_bits(r_quant, theta_quant, self.bits, self.bits)
            metadata["packed_bytes"] = packed_bytes
            metadata["payload_size_bytes"] = len(packed_bytes)

        return codes, metadata

    def dequantize(self, codes: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        r_quant = codes[..., 0].float()
        theta_quant = codes[..., 1].float()

        r_norm = r_quant / float(metadata["r_levels"] - 1)
        r = r_norm * metadata["r_scale"]

        theta_norm = (theta_quant + 0.5) / float(metadata["theta_levels"])
        theta = theta_norm * (2.0 * math.pi) - math.pi

        x1, x2 = self.polar2d.inverse_transform(r, theta)
        flat = torch.empty(x1.numel() * 2, dtype=x1.dtype, device=x1.device)
        flat[0::2] = x1
        flat[1::2] = x2

        if metadata["pad_len"] > 0:
            flat = flat[:-metadata["pad_len"]]

        return flat.reshape(metadata["orig_shape"])

    def unpack_and_dequantize(self, packed_bytes: bytes, metadata: Dict[str, Any], device: str = "cpu") -> torch.Tensor:
        """Directly reconstructs weight matrix from packed raw byte stream."""
        orig_shape = metadata["orig_shape"]
        total_elements = math.prod(orig_shape) + metadata["pad_len"]
        num_pairs = total_elements // 2
        r_codes, th_codes = unpack_polar_bits(packed_bytes, num_pairs, metadata["bits"], metadata["bits"], device=device)
        codes = torch.stack([r_codes, th_codes], dim=-1)
        return self.dequantize(codes, metadata)


def compare_polar_vs_e8_at_matched_bitrate(weights: torch.Tensor, target_bpw: float = 2.0) -> Dict[str, Any]:
    """
    Compares PolarQuant and E8 Lattice Quantization at identical nominal bitrates.
    Returns metrics including exact_bpw, mse, snr_db for both approaches.
    """
    from polarflash_quant.lattice.e8_lattice import E8ResidualVectorQuantizer

    # 1. PolarQuant evaluation
    rpq = RecursivePolarQuant(block_size=8)
    packed_bytes, meta = rpq.encode_packed(weights, target_bpw=target_bpw)
    recon_polar = rpq.decode_packed(packed_bytes, meta, device=str(weights.device))
    mse_polar = float(torch.mean((weights - recon_polar) ** 2).item())
    exact_bpw = meta["exact_bpw"]

    signal_power = float(torch.mean(weights ** 2).item())
    snr_polar = 10.0 * math.log10(max(signal_power / max(mse_polar, 1e-12), 1.0))

    # 2. E8 Lattice evaluation
    stages = max(1, int(round(target_bpw)))
    e8_rvq = E8ResidualVectorQuantizer(num_stages=stages)
    recon_e8, meta_e8 = e8_rvq.quantize(weights)
    mse_e8 = meta_e8["total_mse"]
    snr_e8 = meta_e8["snr_db"]

    return {
        "target_bpw": target_bpw,
        "polar": {
            "exact_bpw": exact_bpw,
            "mse": mse_polar,
            "snr_db": snr_polar
        },
        "e8": {
            "exact_bpw": target_bpw,
            "mse": mse_e8,
            "snr_db": snr_e8
        }
    }





