"""
POLAR-FLASH: PV-GradCode Alternating Continuous-Discrete Optimization Engine
Implements Part IV Sections 14, 15 & 16 of the POLAR-FLASH Specification:
1. Continuous V-Step: theta_{t+1} = theta_t - eta * grad_theta L(W_hat(theta_t, c_t)) with frozen codes.
2. Discrete P-Step: Sensitivity-guided code candidate generation: Delta L ~ grad_c L^T * Delta c.
3. Verification & Rollback Gate: Rejects discrete code updates that increase validation/calibration loss.
4. Convergence tracking: ||grad_theta L||_2, accepted discrete updates, and rejected updates.
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
    - P-Step: Discrete sensitivity-guided code transitions every K steps.
    """
    def __init__(
        self,
        continuous_params: List[torch.Tensor],
        discrete_codes: torch.Tensor,
        lr: float = 1e-4,
        p_step_interval: int = 50,
        rollback_tolerance: float = 0.01
    ):
        self.continuous_params = [p for p in continuous_params if p.requires_grad]
        self.codes = discrete_codes
        self.lr = lr
        self.p_step_interval = p_step_interval
        self.rollback_tolerance = rollback_tolerance

        # Continuous optimizer
        self.v_optimizer = torch.optim.AdamW(self.continuous_params, lr=lr, weight_decay=1e-2)

        self.step_count = 0
        self.accepted_p_updates = 0
        self.rejected_p_updates = 0
        self.loss_history: List[float] = []

    def v_step(self, loss: torch.Tensor):
        """
        Executes continuous V-Step:
        Updates theta while codes c remain frozen.
        """
        self.v_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # Gradient clipping for numerical stability
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
        Executes discrete P-Step:
        1. Ranks coordinate sensitivity using first-order Taylor approximation:
           Delta L_i ~ grad_c L_i * Delta c_i
        2. Proposes discrete candidate flips for top-k coordinates.
        3. Evaluates actual objective and rolls back if loss degrades beyond tolerance.
        """
        current_loss = objective_eval_fn(self.codes)
        n = self.codes.numel()
        k = max(1, int(n * topk_fraction))

        # First-order sensitivity: find coordinates with strongest negative product (greatest loss reduction)
        flat_grad = code_gradient.detach().view(-1)
        # Choosing direction opposing gradient: flip_dir = -sign(grad)
        flip_dir = -torch.sign(flat_grad)
        predicted_reduction = torch.abs(flat_grad)

        topk_vals, topk_indices = torch.topk(predicted_reduction, k=k, sorted=False)

        # Create candidate codes
        candidate_codes = self.codes.clone()
        candidate_flat = candidate_codes.view(-1)
        candidate_flat[topk_indices] += flip_dir[topk_indices].to(candidate_flat.dtype)

        # Objective evaluation
        candidate_loss = objective_eval_fn(candidate_codes)

        # Rollback gate: Accept if candidate_loss <= current_loss * (1.0 + rollback_tolerance)
        if candidate_loss <= current_loss:
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

        # Step 3: Disjoint Merge (filter elements that disagree with elected sign)
        disjoint_mask = (signs == majority_sign.unsqueeze(0))
        disjoint_vals = torch.where(disjoint_mask, stacked, torch.zeros_like(stacked))
        counts = torch.sum(disjoint_mask.float(), dim=0)
        counts[counts == 0] = 1.0

        merged_flat = torch.sum(disjoint_vals, dim=0) / counts
        return merged_flat.reshape(shape).to(first_tensor.dtype)

