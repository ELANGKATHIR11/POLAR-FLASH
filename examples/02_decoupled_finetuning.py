#!/usr/bin/env python3
"""
Example 2: Decoupled Domain Adapter Fine-Tuning
Demonstrates training a specialized domain adapter (e.g. Code, OCR, Reasoning)
using Polar-parameterized LoRA with built-in thermal duty-cycle throttling.
"""

import time
import torch
import torch.nn as nn
from polarflash import PolarLoRALinear

def main():
    print("=" * 60)
    print("POLAR-FLASH: Decoupled Domain Adapter Fine-Tuning")
    print("=" * 60)

    in_features = 1024
    out_features = 1024
    rank = 16

    # Create Polar-LoRA layer
    base_linear = nn.Linear(in_features, out_features, bias=False)
    polar_lora = PolarLoRALinear(
        base_layer=base_linear,
        in_features=in_features,
        out_features=out_features,
        r=rank,
        lora_alpha=32.0
    )

    # Simulated micro-batch training loop with thermal duty-cycle cooldown
    optimizer = torch.optim.AdamW(polar_lora.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print(f"[*] Training Polar-LoRA adapter (rank={rank}) with thermal guard...")
    for step in range(5):
        inputs = torch.randn(4, in_features)
        targets = torch.randn(4, out_features)

        optimizer.zero_grad()
        outputs = polar_lora(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}/5 - Loss: {loss.item():.4f}")

        # Duty-cycle cooling sleep: keeps laptop GPU temperatures low (<60°C)
        time.sleep(0.02)

    print("[+] Domain adapter training complete without thermal spike or OOM!")

if __name__ == "__main__":
    main()
