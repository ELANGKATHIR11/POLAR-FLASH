"""
POLAR-FLASH: Diagonal Fisher Information & Sparse Ternary Residual Engine
Implements Part II Section 6 of the POLAR-FLASH Mathematical Specification:
1. Diagonal Fisher Information estimation: F_i = E_{(x,y)} [(dL / dw_i)^2].
2. Layer sensitivity ranking: alpha_l = mean(F_{l, i}) for rate-distortion allocation and DARE merging.
3. Sparse ternary residual decomposition: r_i in {-s, 0, +s} with configurable sparsity (1%, 3%, 5%, 10%).
4. Optimal ternary residual scale derivation: s* = sum_{i in S} |e_i| / |S|.
5. Comparative sparsity analysis (1%, 3%, 5%, 10%) with exact index, sign, and scale overhead accounting.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional


class FisherSensitivityEstimator:
    """
    Estimates the diagonal Fisher Information Matrix (FIM) across model weights
    using calibration samples and empirical squared gradients.
    """
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}

    def compute_empirical_fisher(
        self,
        model: nn.Module,
        calibration_dataloader: Any,
        num_batches: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Computes diagonal Fisher: F_i = (1 / M) * sum_{m=1}^M (g_{i, m})^2
        """
        model.eval()
        fisher_accum: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros_like(param.data, device=device)

        batch_count = 0
        for batch in calibration_dataloader:
            if batch_count >= num_batches:
                break
            model.zero_grad()

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                labels = batch.get("labels", input_ids).to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)

            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += (param.grad.detach() ** 2)

            batch_count += 1

        for name in fisher_accum:
            fisher_accum[name] /= max(batch_count, 1)

        self.fisher_diagonal = fisher_accum
        return fisher_accum

    def compute_layer_sensitivities(self) -> Dict[str, float]:
        """
        Computes scalar sensitivity alpha_l = mean(FIM_l) for each layer.
        Used directly in Lagrangian rate allocation: min sum alpha_l * D_l(R_l).
        """
        sensitivities: Dict[str, float] = {}
        for name, f_tensor in self.fisher_diagonal.items():
            sensitivities[name] = float(torch.mean(f_tensor).item())
        return sensitivities


class SparseTernaryResidual:
    """
    Constructs optimal sparse ternary residuals:
    r_i in {-s, 0, +s}
    to correct low-bit quantization error e = w - w_hat.
    Sparsity budget: k / N coordinates are non-zero (typically 1%, 3%, 5%, 10%).
    """
    def __init__(self, sparsity_fraction: float = 0.05):
        self.sparsity_fraction = max(0.001, min(0.5, sparsity_fraction))

    def decompose(
        self,
        error_residual: torch.Tensor,
        fisher_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Selects top-k coordinates to receive ternary correction {-s, +s}.
        Coordinates are ranked by importance metric:
        - If fisher_weights provided: score = |e_i| * sqrt(F_i)
        - Else: score = |e_i|
        Optimal scale: s* = (sum_{i in TopK} |e_i|) / k.
        """
        orig_shape = error_residual.shape
        flat_e = error_residual.detach().cpu().float().view(-1)
        n = flat_e.numel()
        k = max(1, int(n * self.sparsity_fraction))

        if fisher_weights is not None:
            flat_f = fisher_weights.detach().cpu().float().view(-1)
            score = torch.abs(flat_e) * torch.sqrt(torch.clamp(flat_f, min=1e-12))
        else:
            score = torch.abs(flat_e)

        topk_scores, topk_indices = torch.topk(score, k=k, sorted=False)
        selected_errors = flat_e[topk_indices]

        # Optimal ternary magnitude: mean of absolute errors in active set
        scale = float(torch.mean(torch.abs(selected_errors)).item())

        # Ternary assignments in {-s, +s}
        ternary_signs = torch.sign(selected_errors)
        ternary_signs[ternary_signs == 0] = 1.0
        ternary_values = ternary_signs * scale

        # Construct sparse tensor
        r_sparse = torch.zeros(n, dtype=torch.float32)
        r_sparse[topk_indices] = ternary_values
        r_sparse = r_sparse.reshape(orig_shape)

        remaining_e = error_residual - r_sparse
        mse_before = float(torch.mean(error_residual ** 2).item())
        mse_after = float(torch.mean(remaining_e ** 2).item())
        error_reduction_pct = (1.0 - (mse_after / max(mse_before, 1e-15))) * 100.0

        # Exact storage overhead accounting:
        # index bits: ceil(log2(n)) per non-zero + 1 bit sign per non-zero + 32-bit float scale
        index_bits_per_nonzero = math.ceil(math.log2(max(n, 2)))
        sign_bits_per_nonzero = 1
        scale_bits = 32
        total_residual_bits = k * (index_bits_per_nonzero + sign_bits_per_nonzero) + scale_bits
        effective_residual_bpw = total_residual_bits / max(n, 1)

        metadata = {
            "sparsity_fraction": self.sparsity_fraction,
            "total_coords": n,
            "nonzero_coords": k,
            "optimal_scale": scale,
            "mse_before_residual": mse_before,
            "mse_after_residual": mse_after,
            "error_reduction_percentage": error_reduction_pct,
            "effective_residual_bpw": effective_residual_bpw,
            "topk_indices": topk_indices,
            "ternary_signs": ternary_signs
        }

        return r_sparse, metadata


def compare_sparsity_levels(
    error_residual: torch.Tensor,
    fisher_weights: Optional[torch.Tensor] = None,
    sparsity_levels: List[float] = [0.01, 0.03, 0.05, 0.10]
) -> Dict[str, Any]:
    """
    Evaluates and compares ternary residual correction across 1%, 3%, 5%, and 10% sparsity.
    Computes exact trade-off: error reduction percentage vs. bits-per-weight storage overhead.
    """
    results = {}
    for sp in sparsity_levels:
        decomposer = SparseTernaryResidual(sparsity_fraction=sp)
        _, meta = decomposer.decompose(error_residual, fisher_weights=fisher_weights)
        label = f"{int(sp * 100)}%"
        results[label] = {
            "sparsity_fraction": sp,
            "nonzero_count": meta["nonzero_coords"],
            "optimal_scale": meta["optimal_scale"],
            "mse_before": meta["mse_before_residual"],
            "mse_after": meta["mse_after_residual"],
            "error_reduction_percentage": meta["error_reduction_percentage"],
            "effective_residual_bpw": meta["effective_residual_bpw"]
        }
    return results
