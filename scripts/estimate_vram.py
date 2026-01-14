#!/usr/bin/env python
# coding: utf-8

"""Estimate VRAM/RAM breakdown for OMGSR LoRA training.

This script is designed to run on CPU and *estimate* GPU VRAM usage for the
parameter/optimizer/gradient parts. The activation/temp memory is highly
model- and implementation-dependent; we print a conservative explanation and
the terms that dominate it.

Usage:
  uv run python scripts/estimate_vram.py --config configs/omgsr_sana_1024_rain.yml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    mp = str(mixed_precision or "no").lower()
    if mp in {"bf16", "bf16-mixed"}:
        return torch.bfloat16
    if mp in {"fp16", "16", "16-mixed"}:
        return torch.float16
    return torch.float32


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{size:.0f}{unit}"
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}TB"


def _count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters(recurse=True):
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


@dataclass
class AdamWStateAssumptions:
    # Commonly used mental models for AdamW memory:
    # - exp_avg (m) and exp_avg_sq (v) states, typically fp32
    # - sometimes an fp32 master copy of weights is also kept (depends on impl)
    state_dtype: torch.dtype = torch.float32
    has_master_weights_fp32: bool = True


def _estimate_adamw_bytes(trainable_params: int, assumptions: AdamWStateAssumptions) -> int:
    state_bytes = _bytes_per_element(assumptions.state_dtype)
    # m + v
    total = trainable_params * 2 * state_bytes
    if assumptions.has_master_weights_fp32:
        total += trainable_params * _bytes_per_element(torch.float32)
    return int(total)


def _safe_import_training_model():
    # Lazy import so this script can be used even if some heavy deps are missing.
    from train.train_omgsr_sana_rain_lightning import OMGSR_SanaRain_Lightning

    return OMGSR_SanaRain_Lightning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config, e.g. configs/omgsr_sana_1024_rain.yml",
    )
    parser.add_argument(
        "--adam_state",
        type=str,
        default="fp32_master",
        choices=["fp32_master", "fp32_nomaster", "bf16_master", "bf16_nomaster"],
        help=(
            "AdamW state dtype assumption. 'fp32_master' is conservative (largest). "
            "If you're using bitsandbytes 8-bit Adam, optimizer state can be much smaller."
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    weight_dtype = _dtype_from_mixed_precision(getattr(cfg, "mixed_precision", "no"))
    w_bytes = _bytes_per_element(weight_dtype)

    # Build the LightningModule on CPU (safe for CPU-only analysis).
    ModelCls = _safe_import_training_model()
    model = ModelCls(cfg)
    model.to(torch.device("cpu"))
    model.setup(stage="fit")

    parts: Dict[str, torch.nn.Module] = {
        "fixed_vae": model.fixed_vae,
        "lora_vae": model.lora_vae,
        "sana_transformer": model.sana_transformer,
        "net_dv3d": model.net_dv3d,
        "net_disc": model.net_disc,
    }

    total_params = 0
    trainable_params = 0
    per_part: Dict[str, Tuple[int, int]] = {}
    for name, m in parts.items():
        tp, tr = _count_params(m)
        per_part[name] = (tp, tr)
        total_params += tp
        trainable_params += tr

    # Parameter memory: weights live on device regardless of trainable/frozen.
    weights_bytes = total_params * w_bytes

    # Gradient memory (rough): for trainable params only. Usually same dtype as param.
    grads_bytes = trainable_params * w_bytes

    if args.adam_state == "fp32_master":
        adam_assump = AdamWStateAssumptions(state_dtype=torch.float32, has_master_weights_fp32=True)
    elif args.adam_state == "fp32_nomaster":
        adam_assump = AdamWStateAssumptions(state_dtype=torch.float32, has_master_weights_fp32=False)
    elif args.adam_state == "bf16_master":
        adam_assump = AdamWStateAssumptions(state_dtype=torch.bfloat16, has_master_weights_fp32=True)
    else:
        adam_assump = AdamWStateAssumptions(state_dtype=torch.bfloat16, has_master_weights_fp32=False)

    opt_bytes = _estimate_adamw_bytes(trainable_params, adam_assump)

    print("\n=== OMGSR VRAM/RAM estimator (CPU-side) ===")
    print(f"Config: {args.config}")
    print(f"Assumed weight dtype: {weight_dtype} ({w_bytes} bytes/elem)")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params:     {total_params:,}")

    print("\n--- Per-component parameters ---")
    for name, (tp, tr) in per_part.items():
        print(f"{name:16s} total={tp:>12,}  trainable={tr:>10,}")

    print("\n--- Memory terms that you can estimate accurately ---")
    print(f"Weights (all params):        {_format_bytes(weights_bytes)}")
    print(f"Gradients (trainable only):  {_format_bytes(grads_bytes)}")
    print(
        "AdamW states (trainable):   "
        f"{_format_bytes(opt_bytes)}  "
        f"(state_dtype={adam_assump.state_dtype}, master_fp32={adam_assump.has_master_weights_fp32})"
    )
    print(f"Subtotal (W+G+Opt):          {_format_bytes(weights_bytes + grads_bytes + opt_bytes)}")

    print("\n--- Why peak VRAM is much larger in practice ---")
    print(
        "Peak VRAM during training is usually dominated by activations + temporary buffers, "
        "not by LoRA weights/optimizer state. This is especially true for 1024px training "
        "with ConvNeXt/DINO perceptual loss and a discriminator: you backprop through large "
        "feature maps of shape ~[B,C,H,W] at multiple stages.\n"
        "Activation memory depends on: (1) resolution, (2) batch, (3) whether you backprop "
        "through DINO/disc, (4) attention implementation (xformers/flash-attn), (5) gradient "
        "checkpointing. It's hard to compute exactly without running a step on the target device."
    )

    print("\n--- Practical takeaways ---")
    print(
        "1) If your observed peak VRAM is ~45-50GB, and this script reports only a few GB for W+G+Opt, "
        "   then activations/temp buffers are the remainder.\n"
        "2) To reduce peak VRAM: enable xformers/flash-attn (if supported), keep gradient_checkpointing=True, "
        "   lower resolution, or remove/disable DINO/disc backward paths.\n"
        "3) For an exact peak measurement, run 1 training step on GPU and print torch.cuda.max_memory_allocated()."
    )


if __name__ == "__main__":
    main()
