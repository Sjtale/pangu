#!/usr/bin/env python3
"""Inventory real DCU fast-attention backends without rerunning SDPA fallback."""

import argparse
import importlib.util
import json
import shutil

import torch


BACKEND_MODULES = (
    "flash_attn",
    "aiter",
    "xformers",
    "triton",
)


def capability_report():
    installed = {
        name: importlib.util.find_spec(name) is not None for name in BACKEND_MODULES
    }
    cuda_backend = getattr(torch.backends, "cuda", None)
    flash_available = None
    if cuda_backend is not None and hasattr(cuda_backend, "is_flash_attention_available"):
        try:
            flash_available = bool(cuda_backend.is_flash_attention_available())
        except Exception:
            flash_available = None
    profilers = {
        name: shutil.which(name)
        for name in ("rocprofv2", "rocprof", "hipprof")
        if shutil.which(name)
    }
    # EarthAttention3D requires both an arbitrary learned additive bias and a
    # shifted-window additive mask. Presence of a package alone is insufficient;
    # a backend adapter must explicitly prove support for both before benchmarking.
    return {
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "torch_flash_attention_available": flash_available,
        "installed_modules": installed,
        "profilers": profilers,
        "required_features": [
            "fp16 q/k/v",
            "arbitrary learned additive earth-position bias",
            "shifted-window additive mask",
            "non-causal inference",
        ],
        "compatible_backend_verified": False,
        "decision": (
            "STOP unless a discovered backend adapter proves both additive-bias "
            "and shifted-mask support and profiler output confirms a fused kernel."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = capability_report()
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
