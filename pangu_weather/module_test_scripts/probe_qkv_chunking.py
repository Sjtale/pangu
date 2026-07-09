"""Probe: Test QKV projection chunking and attention chunk size sweeps.

Tests memory impact of:
1. Baseline chunked attention (chunk_size=3)
2. QKV projection splitting (Q, K, V computed separately)
3. Attention chunk size sweep (1, 2, 3, 4, 5)
4. Combined QKV split + smaller chunk size

Usage (on server, from pangu_weather/ directory):
    python module_test_scripts/probe_qkv_chunking.py
"""

import os
import sys
import gc
import types
import json
import time

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
from onescience.modules.attention.earthattention3d import EarthAttention3D
from pangu_profile_model import (
    build_pangu_model,
    enable_chunked_attention,
    _forward_chunked_earth_attention_3d,
)


def _forward_split_qkv_chunked_attention(self, x, mask=None):
    """Chunked attention with split Q/K/V projections to reduce peak QKV alloc.
    
    Instead of computing qkv = self.qkv(x) which creates a [B, N, W, 3*C] tensor,
    this computes Q, K, V separately using weight slices, so only one [B, N, W, C]
    intermediate exists at a time.
    """
    BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels = x.shape
    head_dim = Channels // self.num_heads

    # Split QKV weight [3*C, C] into three [C, C] chunks
    qkv_weight = self.qkv.weight  # [3*Channels, Channels]
    qkv_bias = self.qkv.bias if self.qkv.bias is not None else None

    q_weight = qkv_weight[:Channels]
    k_weight = qkv_weight[Channels:2*Channels]
    v_weight = qkv_weight[2*Channels:]

    q_bias = qkv_bias[:Channels] if qkv_bias is not None else None
    k_bias = qkv_bias[Channels:2*Channels] if qkv_bias is not None else None
    v_bias = qkv_bias[2*Channels:] if qkv_bias is not None else None

    # Compute Q, K, V separately
    x_flat = x.reshape(-1, Channels)
    q = torch.nn.functional.linear(x_flat, q_weight, q_bias)
    q = q.reshape(BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens,
                  self.num_heads, head_dim).permute(0, 3, 1, 2, 4)

    k = torch.nn.functional.linear(x_flat, k_weight, k_bias)
    k = k.reshape(BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens,
                  self.num_heads, head_dim).permute(0, 3, 1, 2, 4)

    v = torch.nn.functional.linear(x_flat, v_weight, v_bias)
    v = v.reshape(BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens,
                  self.num_heads, head_dim).permute(0, 3, 1, 2, 4)
    del x_flat

    q = q * self.scale

    # Earth position bias (same as original)
    cache_bias = bool(getattr(self, "_pangu_cache_earth_bias", False))
    cached = getattr(self, "_cached_earth_position_bias", None)
    if cache_bias and cached is not None and cached.device == q.device:
        earth_position_bias = cached
    else:
        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.num_pressure_height_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        earth_position_bias = earth_position_bias.unsqueeze(0)
        if cache_bias:
            self._cached_earth_position_bias = earth_position_bias

    # Chunked attention computation
    chunk_size = max(1, int(getattr(self, "_pangu_attention_chunk_size", 3)))
    out = x.new_empty(
        BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels
    )
    for start in range(0, BatchTimesWidthWindows, chunk_size):
        end = min(start + chunk_size, BatchTimesWidthWindows)
        q_chunk = q[start:end]
        k_chunk = k[start:end]
        v_chunk = v[start:end]

        attn_chunk = q_chunk @ k_chunk.transpose(-2, -1)
        attn_chunk = attn_chunk + earth_position_bias

        if mask is not None:
            NumWidthWindows = mask.shape[0]
            mask_indices = (
                torch.arange(start, end, device=mask.device) % NumWidthWindows
            )
            mask_chunk = mask.index_select(0, mask_indices)
            attn_chunk = attn_chunk + mask_chunk.unsqueeze(1)

        attn_chunk = self.softmax(attn_chunk)
        attn_chunk = self.attn_drop(attn_chunk)

        out_chunk = (
            (attn_chunk @ v_chunk)
            .permute(0, 2, 3, 1, 4)
            .reshape(q_chunk.shape[0], NumPressureHeightWindows, WindowTokens, Channels)
        )
        out[start:end].copy_(out_chunk)

    x = self.proj(out)
    x = self.proj_drop(x)
    return x


def enable_split_qkv_attention(model, chunk_size=3, cache_earth_bias=False):
    """Patch EarthAttention3D with split QKV variant."""
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "EarthAttention3D":
            module._pangu_attention_chunk_size = chunk_size
            module._pangu_cache_earth_bias = bool(cache_earth_bias)
            module.forward = types.MethodType(
                _forward_split_qkv_chunked_attention, module
            )
            patched += 1
    return patched


def run_test(model, model_input, label, n_trials=3):
    """Run inference and measure VRAM + timing."""
    peaks = []
    times = []
    for trial in range(n_trials):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            out_surface, out_upper_air = model(model_input)
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            # Cleanup
            del out_surface, out_upper_air

        peak = torch.cuda.max_memory_allocated() / 1024**2
        elapsed = (t_end - t_start) * 1000
        peaks.append(peak)
        times.append(elapsed)

    avg_peak = sum(peaks) / len(peaks)
    avg_time = sum(times) / len(times)
    print(
        f"  [{label:35s}] VRAM: {avg_peak:8.1f} MB, "
        f"Time: {avg_time:6.1f} ms "
        f"(peaks={[f'{p:.0f}' for p in peaks]})"
    )
    return {"label": label, "avg_peak_mb": avg_peak, "avg_time_ms": avg_time,
            "peaks": peaks, "times": times}


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

    # Load test sample
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()
    for data in test_dataloader:
        invar_cpu = data[0]
        break
    del datapipe, test_dataloader
    gc.collect()

    # Prepare input
    invar_surface = invar_cpu[:, :4, :, :].to("cuda:0", dtype=torch.float16)
    invar_upper_air = invar_cpu[:, 4:, :, :].to("cuda:0", dtype=torch.float16)
    invar_surface_with_mask = torch.concat([invar_surface, surface_mask], dim=1)
    invar_upper_air_reshaped = invar_upper_air.reshape(
        invar_upper_air.shape[0], 5, 13,
        invar_upper_air.shape[2], invar_upper_air.shape[3]
    )
    model_input = (invar_surface_with_mask, invar_upper_air_reshaped)

    profiles = getattr(cfg, "student_profiles", {})
    p = profiles.get("pgw_lite_pruned_96")
    model_profile = {
        "patch_size": [int(v) for v in p.patch_size],
        "embed_dim": int(p.embed_dim),
        "num_heads": [int(v) for v in p.num_heads],
        "window_size": [int(v) for v in cfg.window_size],
    }

    results = []

    def build_and_load(chunk_size=3, split_qkv=False):
        """Build model with specific attention config and load weights."""
        model = build_pangu_model(
            img_size=cfg_data.dataset.img_size,
            patch_size=model_profile["patch_size"],
            embed_dim=model_profile["embed_dim"],
            num_heads=model_profile["num_heads"],
            window_size=model_profile["window_size"],
            recompute_skip=False,
            layerwise_inference=True,
            layerwise_empty_cache=False,
            chunked_attention=not split_qkv,  # disable if we'll apply split variant
            attention_chunk_size=chunk_size,
        )
        if split_qkv:
            enable_split_qkv_attention(model, chunk_size=chunk_size)

        model.half().to("cuda:0")
        fp16_ckpt_path = os.path.join(cfg.checkpoint_dir, "model_fp16.pth")
        ckpt = torch.load(fp16_ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
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
        return model

    print("=" * 80)
    print("QKV CHUNKING & ATTENTION CHUNK SIZE SWEEP")
    print("=" * 80)

    # ---- Sweep attention chunk sizes with original QKV ----
    for chunk_size in [1, 2, 3, 4, 5]:
        model = build_and_load(chunk_size=chunk_size, split_qkv=False)
        r = run_test(model, model_input, f"original_qkv_chunk{chunk_size}")
        results.append(r)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Sweep with split QKV ----
    for chunk_size in [1, 2, 3]:
        model = build_and_load(chunk_size=chunk_size, split_qkv=True)
        r = run_test(model, model_input, f"split_qkv_chunk{chunk_size}")
        results.append(r)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<35s} {'Peak VRAM':>10} {'Time':>10} {'ΔVRAM':>10}")
    print("-" * 80)
    baseline_peak = results[2]["avg_peak_mb"]  # chunk_size=3 is the baseline
    for r in results:
        delta = r["avg_peak_mb"] - baseline_peak
        print(
            f"{r['label']:<35s} "
            f"{r['avg_peak_mb']:>9.1f}M "
            f"{r['avg_time_ms']:>9.1f}ms "
            f"{delta:>+9.1f}M"
        )

    # Save
    os.makedirs("logs", exist_ok=True)
    with open("logs/qkv_chunking_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to logs/qkv_chunking_results.json")


if __name__ == "__main__":
    main()
