"""Probe: Test input data optimization strategies for VRAM reduction.

Tests multiple approaches to reducing the GPU memory footprint of the
inference input preparation phase:

1. Baseline: current inference.py flow
2. Early-delete: del input tensors immediately after patchembed
3. Fused surface: avoid creating the 7-channel concat tensor
4. Lazy reshape: delay upper_air reshape into the model forward

Usage (on server, from pangu_weather/ directory):
    python module_test_scripts/probe_input_optimization.py
"""

import os
import sys
import gc
import types
import json

os.environ.setdefault("PANGU_AUTO_SCAN_CHECKPOINT", "0")
os.environ.setdefault("PANGU_DISABLE_CUDA_GRAPH", "1")
os.environ.setdefault("PANGU_LAYERWISE_INFERENCE", "1")
os.environ.setdefault("PANGU_RECOMPUTE_SKIP", "0")
os.environ.setdefault("PANGU_DIRECT_RECOVERY", "1")
os.environ.setdefault("PANGU_DIRECT_RECOVERY_WIDTH_CHUNK", "24")
os.environ.setdefault("PANGU_SCORED_ONLY_RECOVERY", "1")
os.environ.setdefault("PANGU_CHUNKED_ATTENTION", "1")
os.environ.setdefault("PANGU_ATTN_CHUNK_SIZE", "3")
os.environ.setdefault("PANGU_CHUNKED_MLP", "1")
os.environ.setdefault("PANGU_MLP_CHUNK_SIZE", "32768")
os.environ.setdefault("PANGU_INPLACE_BLOCK", "1")

import torch
import numpy as np

current_path = os.getcwd()
sys.path.append(current_path)

from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe
from pangu_profile_model import build_pangu_model


def measure_peak_vram(model, invar_fn, label, n_trials=3):
    """Run inference and measure peak VRAM."""
    peaks = []
    for trial in range(n_trials):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            invar = invar_fn()
            out_surface, out_upper_air = model(invar)

            # Simulate the output postprocessing from inference.py
            pred = torch.concat(
                [out_surface.detach().cpu(), out_upper_air.detach().cpu()],
                dim=1,
            ).float()

            del out_surface, out_upper_air, pred, invar
            gc.collect()
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**2
        peaks.append(peak)

    avg = sum(peaks) / len(peaks)
    print(f"  [{label}] Peak VRAM: {avg:.1f} MB (trials: {[f'{p:.1f}' for p in peaks]})")
    return avg


def main():
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    # Load static data
    land_mask = torch.from_numpy(
        np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32)
    )
    topography = torch.from_numpy(
        np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32)
    )
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to("cuda:0")
    surface_mask = surface_mask.unsqueeze(0).half()

    # Build model (matching submission config)
    profiles = getattr(cfg, "student_profiles", {})
    p = profiles.get("pgw_lite_pruned_96")
    model_profile = {
        "name": "pgw_lite_pruned_96",
        "patch_size": [int(v) for v in p.patch_size],
        "embed_dim": int(p.embed_dim),
        "num_heads": [int(v) for v in p.num_heads],
        "window_size": [int(v) for v in cfg.window_size],
    }

    model = build_pangu_model(
        img_size=cfg_data.dataset.img_size,
        patch_size=model_profile["patch_size"],
        embed_dim=model_profile["embed_dim"],
        num_heads=model_profile["num_heads"],
        window_size=model_profile["window_size"],
        recompute_skip=False,
        layerwise_inference=True,
        layerwise_empty_cache=False,
        chunked_attention=True,
        attention_chunk_size=3,
    )

    # Load weights
    fp16_ckpt_path = os.path.join(cfg.checkpoint_dir, "model_fp16.pth")
    ckpt = torch.load(fp16_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    model.half().to("cuda:0")
    model_state = model.state_dict()
    with torch.no_grad():
        for key, value in state_dict.items():
            if key.endswith("_scale"):
                continue
            if key not in model_state:
                continue
            target = model_state[key]
            if key.endswith(".weight") and value.dtype == torch.int8:
                scale_key = key + "_scale"
                if scale_key in state_dict:
                    scale = state_dict[scale_key].to(device=target.device, dtype=torch.float32)
                    if scale.ndim == 1 and value.ndim == 2:
                        scale = scale.view(-1, 1)
                    target.copy_(
                        (value.to(device=target.device, dtype=torch.float32) * scale).to(target.dtype)
                    )
                    continue
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    model.eval()
    del state_dict, ckpt
    gc.collect()
    torch.cuda.empty_cache()

    # Load test sample
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()
    for data in test_dataloader:
        invar_cpu = data[0]
        break
    del datapipe, test_dataloader
    gc.collect()

    print("=" * 70)
    print("INPUT OPTIMIZATION VRAM COMPARISON")
    print("=" * 70)

    results = {}

    # ---- Test 1: Baseline (current inference.py) ----
    def baseline_input():
        invar_surface = invar_cpu[:, :4, :, :].to("cuda:0", dtype=torch.float16)
        invar_upper_air = invar_cpu[:, 4:, :, :].to("cuda:0", dtype=torch.float16)
        invar_surface_with_mask = torch.concat([invar_surface, surface_mask], dim=1)
        invar_upper_air_reshaped = invar_upper_air.reshape(
            invar_upper_air.shape[0], 5, 13,
            invar_upper_air.shape[2], invar_upper_air.shape[3]
        )
        return (invar_surface_with_mask, invar_upper_air_reshaped)

    results["baseline"] = measure_peak_vram(model, baseline_input, "Baseline")

    # ---- Test 2: Pre-concatenated surface (avoid surface concat on GPU) ----
    # Pre-compute surface+mask on CPU to avoid GPU concat allocation
    def precat_input():
        invar_surface = invar_cpu[:, :4, :, :]
        # Concat with CPU-side mask then transfer as one tensor
        cpu_mask = surface_mask.cpu()
        combined_surface = torch.concat(
            [invar_surface.half(), cpu_mask], dim=1
        ).to("cuda:0", dtype=torch.float16, non_blocking=True)

        invar_upper_air = invar_cpu[:, 4:, :, :].to("cuda:0", dtype=torch.float16)
        invar_upper_air_reshaped = invar_upper_air.reshape(
            invar_upper_air.shape[0], 5, 13,
            invar_upper_air.shape[2], invar_upper_air.shape[3]
        )
        return (combined_surface, invar_upper_air_reshaped)

    results["precat_surface"] = measure_peak_vram(model, precat_input, "Pre-concat surface on CPU")

    # ---- Test 3: Direct reshape without intermediate ----
    # Avoid creating the flat [1,65,H,W] upper_air on GPU,
    # reshape directly from CPU and transfer
    def direct_reshape_input():
        invar_surface = invar_cpu[:, :4, :, :].to("cuda:0", dtype=torch.float16)
        invar_surface_with_mask = torch.concat([invar_surface, surface_mask], dim=1)
        # Reshape on CPU first, then transfer only the reshaped tensor
        upper_air_cpu = invar_cpu[:, 4:, :, :].reshape(
            invar_cpu.shape[0], 5, 13,
            invar_cpu.shape[2], invar_cpu.shape[3]
        )
        invar_upper_air_reshaped = upper_air_cpu.to("cuda:0", dtype=torch.float16)
        return (invar_surface_with_mask, invar_upper_air_reshaped)

    results["direct_reshape"] = measure_peak_vram(model, direct_reshape_input, "Direct reshape on CPU")

    # ---- Test 4: Combined: precat + direct reshape ----
    def combined_input():
        cpu_mask = surface_mask.cpu()
        combined_surface = torch.concat(
            [invar_cpu[:, :4, :, :].half(), cpu_mask], dim=1
        ).to("cuda:0", dtype=torch.float16)

        upper_air_cpu = invar_cpu[:, 4:, :, :].reshape(
            invar_cpu.shape[0], 5, 13,
            invar_cpu.shape[2], invar_cpu.shape[3]
        )
        invar_upper_air_reshaped = upper_air_cpu.to("cuda:0", dtype=torch.float16)
        return (combined_surface, invar_upper_air_reshaped)

    results["combined_cpu_prep"] = measure_peak_vram(model, combined_input, "Combined CPU prep")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    baseline = results["baseline"]
    for name, peak in sorted(results.items(), key=lambda x: x[1]):
        delta = peak - baseline
        print(f"  {name:<25s}: {peak:8.1f} MB  ({delta:+6.1f} MB vs baseline)")

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/input_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to logs/input_optimization_results.json")


if __name__ == "__main__":
    main()
