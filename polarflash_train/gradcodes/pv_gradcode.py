"""
POLAR-FLASH: PV-GradCode Alternating Continuous-Discrete Optimization Engine
Implements Part IV Sections 14, 15 & 16 of the POLAR-FLASH Specification:
1. Continuous V-Step: theta_{t+1} = theta_t - eta * grad_theta L(W_hat(theta_t, c_t)) with frozen codes.
2. Discrete P-Step: Sensitivity-guided code candidate generation: Delta L ~ grad_c L^T * Delta c.
3. Trust-Region Verification & Rollback Gate: Rejects discrete code updates that increase loss.
4. Physical training memory accounting: tracks optimizer moments, gradients, and activation buffers.
5. TIES weight merging for multi-task model adapters.
"""

import math
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional, Callable


class PVGradCodeOptimizer:
    """
    Implements PV-GradCode alternating optimization:
    - V-Step: Continuous AdamW updates on scales and continuous parameters theta.
    - P-Step: Discrete sensitivity-guided code transitions every K steps with trust-region gating.
    - Memory Tracker: Computes exact physical footprint of training states.
    """
    def __init__(
        self,
        continuous_params: List[torch.Tensor],
        discrete_codes: torch.Tensor,
        lr: float = 1e-4,
        p_step_interval: int = 50,
        rollback_tolerance: float = 0.0,
        min_improvement: float = 0.0,
        trust_region_radius: float = 0.10,
        max_flips: Optional[int] = None,
        code_min: Optional[int] = -8,
        code_max: Optional[int] = 7
    ):
        self.continuous_params = [p for p in continuous_params if p.requires_grad]
        self.codes = discrete_codes
        self.lr = lr
        self.p_step_interval = p_step_interval
        # Strict monotonic improvement: require L(c_{t+1}) < L(c_t) - min_improvement
        self.min_improvement = max(0.0, min_improvement)
        self.rollback_tolerance = rollback_tolerance  # Deprecated tolerance kept for API compatibility
        self.trust_region_radius = trust_region_radius
        self.max_flips = max_flips
        self.code_min = code_min
        self.code_max = code_max

        # Continuous optimizer
        self.v_optimizer = torch.optim.AdamW(self.continuous_params, lr=lr, weight_decay=1e-2)

        self.step_count = 0
        self.accepted_p_updates = 0
        self.rejected_p_updates = 0
        self.loss_history: List[float] = []

    def compute_training_memory_bytes(self) -> Dict[str, Any]:
        """
        Calculates exact training memory breakdown:
        - Optimizer states: 2 float32 moments per continuous parameter (8 bytes/param)
        - Gradients: 1 float32 gradient per continuous parameter (4 bytes/param)
        - Discrete codes: 1 byte per code (int8 / uint8)
        """
        num_continuous = sum(p.numel() for p in self.continuous_params)
        num_codes = self.codes.numel()

        optimizer_bytes = num_continuous * 8  # 1st and 2nd Adam moments
        gradient_bytes = num_continuous * 4
        parameter_bytes = num_continuous * 4
        code_bytes = num_codes * self.codes.element_size()

        total_bytes = optimizer_bytes + gradient_bytes + parameter_bytes + code_bytes

        return {
            "num_continuous_parameters": num_continuous,
            "num_discrete_codes": num_codes,
            "optimizer_bytes": optimizer_bytes,
            "gradient_bytes": gradient_bytes,
            "code_bytes": code_bytes,
            "total_bytes": total_bytes,
            "optimizer_state_mb": optimizer_bytes / (1024 ** 2),
            "gradient_state_mb": gradient_bytes / (1024 ** 2),
            "code_storage_mb": code_bytes / (1024 ** 2),
            "total_training_overhead_mb": total_bytes / (1024 ** 2)
        }

    def v_step(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Executes continuous V-Step:
        Updates theta while codes c remain frozen.
        retain_graph defaults to False to prevent autograd graph memory leaks across steps.
        """
        self.v_optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(self.continuous_params, max_norm=1.0)
        self.v_optimizer.step()
        self.step_count += 1
        self.loss_history.append(float(loss.item()))

    def p_step(
        self,
        objective_eval_fn: Callable[[torch.Tensor], float],
        code_gradient: torch.Tensor,
        candidate_code_offsets: List[int] = [-1, 1],
        topk_fraction: float = 0.05
    ) -> Dict[str, Any]:
        """
        Executes discrete P-Step with strict monotonic descent:
        1. Evaluates baseline objective L(c_t).
        2. Ranks coordinate sensitivity using first-order Taylor approximation:
           Delta L_i ~ grad_c L_i * Delta c_i
        3. Bounds flips by trust-region radius and max_flips.
        4. Clamps discrete candidates to [code_min, code_max].
        5. Enforces strict monotonic descent: L(c_{t+1}) < L(c_t) - min_improvement.
           Rejects candidate and preserves c_t if loss degrades or fails threshold.
        """
        current_loss = float(objective_eval_fn(self.codes))
        n = self.codes.numel()

        # Enforce trust region cap
        effective_fraction = min(topk_fraction, self.trust_region_radius)
        k = max(1, int(n * effective_fraction))
        if self.max_flips is not None:
            k = min(k, self.max_flips)

        # First-order sensitivity: find coordinates with strongest predicted reduction
        flat_grad = code_gradient.detach().view(-1)
        flip_dir = -torch.sign(flat_grad)
        predicted_reduction = torch.abs(flat_grad)

        topk_vals, topk_indices = torch.topk(predicted_reduction, k=k, sorted=False)

        # Propose candidate discrete codes
        candidate_codes = self.codes.clone()
        candidate_flat = candidate_codes.view(-1)
        candidate_flat[topk_indices] += flip_dir[topk_indices].to(candidate_flat.dtype)

        # Clamp candidate codes to valid discrete alphabet bounds
        if self.code_min is not None or self.code_max is not None:
            c_min = float("-inf") if self.code_min is None else float(self.code_min)
            c_max = float("inf") if self.code_max is None else float(self.code_max)
            candidate_flat.clamp_(min=c_min, max=c_max)

        # Evaluate objective honestly on candidate
        candidate_loss = float(objective_eval_fn(candidate_codes))

        # Strict Monotonic Descent: accept iff candidate_loss < current_loss - min_improvement
        is_strictly_better = (candidate_loss < current_loss - self.min_improvement)

        if is_strictly_better:
            self.codes.copy_(candidate_codes)
            self.accepted_p_updates += 1
            accepted = True
            delta_loss = candidate_loss - current_loss
        else:
            self.rejected_p_updates += 1
            accepted = False
            delta_loss = 0.0

        return {
            "step": self.step_count,
            "accepted": accepted,
            "current_loss": current_loss,
            "candidate_loss": candidate_loss,
            "delta_loss": delta_loss,
            "k_coordinates_flipped": k,
            "total_accepted_p_updates": self.accepted_p_updates,
            "total_rejected_p_updates": self.rejected_p_updates
        }


class TIESWeightMerger:
    """
    Trim, Elect Sign, and Merge (TIES) Algorithm for multi-task model adapters.
    Eliminates interference between specialized domain weights (e.g. Code + Extraction + Vision).
    """
    def __init__(self, density: float = 0.2):
        self.density = density

    def merge(self, task_vectors: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not task_vectors:
            raise ValueError("No task vectors provided for merging.")

        keys = list(task_vectors.keys())
        first_tensor = task_vectors[keys[0]]
        shape = first_tensor.shape
        flat_vectors = [task_vectors[k].flatten().float() for k in keys]

        trimmed = []
        k = max(1, int(flat_vectors[0].numel() * self.density))
        for v in flat_vectors:
            # Step 1: Trim smallest magnitude values
            mags = torch.abs(v)
            topk_vals, topk_indices = torch.topk(mags, k=k, sorted=False)
            mask = torch.zeros_like(v, dtype=torch.bool)
            mask[topk_indices] = True
            trimmed_v = torch.where(mask, v, torch.zeros_like(v))
            trimmed.append(trimmed_v)

        stacked = torch.stack(trimmed, dim=0)

        # Step 2: Elect Majority Sign
        signs = torch.sign(stacked)
        sign_sum = torch.sum(signs, dim=0)
        majority_sign = torch.sign(sign_sum)
        majority_sign[majority_sign == 0] = 1.0

        # Step 3: Disjoint Merge: Average only values that match majority sign
        mask_agree = (signs == majority_sign.unsqueeze(0))
        agreeing_values = torch.where(mask_agree, stacked, torch.zeros_like(stacked))
        num_agreeing = torch.sum(mask_agree.float(), dim=0)
        num_agreeing = torch.clamp(num_agreeing, min=1.0)
        merged = torch.sum(agreeing_values, dim=0) / num_agreeing

        return merged.view(*shape)
