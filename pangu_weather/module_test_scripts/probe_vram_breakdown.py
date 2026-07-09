"""VRAM breakdown probe for Pangu-Weather inference pipeline.

This script instruments every stage of the inference forward pass to record
the delta in torch.cuda.max_memory_allocated(). It runs on the DCU server
and produces a breakdown showing exactly where the 657.9 MB peak is spent.

Usage (on server, from pangu_weather/ directory):
    PANGU_PROFILE_MEMORY=1 python module_test_scripts/probe_vram_breakdown.py

The script uses the exact same model loading + inference flow as inference.py
but intercepts each stage to record memory deltas.
"""

import os
import sys
import json
import gc

# Set defaults BEFORE any torch import to match submission
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
os.environ.setdefault("PANGU_DISABLE_AFFINE_CALIBRATION", "1")
os.environ.setdefault("PANGU_GLOBAL_MEAN_CORRECTION", "0")
os.environ.setdefault("PANGU_STREAM_WEIGHTS", "0")
os.environ.setdefault("PANGU_SPLIT_RECOVERY", "0")
os.environ.setdefault("PANGU_CACHE_EARTH_BIAS", "0")
os.environ.setdefault("PANGU_INPLACE_BLOCK", "1")

import torch
import numpy as np

current_path = os.getcwd()
sys.path.append(current_path)

from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe
from pangu_profile_model import build_pangu_model


# ---- Memory tracking helpers ----

class VRAMTracker:
    """Track VRAM at each checkpoint and compute deltas."""

    def __init__(self):
        self.records = []
        self._last_peak = 0
        self._last_allocated = 0

    def checkpoint(self, tag):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()

        delta_peak = peak - self._last_peak
        delta_alloc = allocated - self._last_allocated

        record = {
            "tag": tag,
            "allocated_mb": allocated / 1024**2,
            "reserved_mb": reserved / 1024**2,
            "peak_mb": peak / 1024**2,
            "delta_peak_mb": delta_peak / 1024**2,
            "delta_alloc_mb": delta_alloc / 1024**2,
        }
        self.records.append(record)
        self._last_peak = peak
        self._last_allocated = allocated

        print(
            f"[VRAM] {tag:45s} | "
            f"alloc={allocated/1024**2:8.1f} MB | "
            f"peak={peak/1024**2:8.1f} MB | "
            f"Δpeak={delta_peak/1024**2:+8.1f} MB | "
            f"Δalloc={delta_alloc/1024**2:+8.1f} MB"
        )

    def reset_peak(self, tag="reset_peak"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._last_peak = 0
            print(f"[VRAM] {tag}: reset peak stats")

    def summary(self):
        print("\n" + "=" * 100)
        print("VRAM BREAKDOWN SUMMARY")
        print("=" * 100)
        print(f"{'Stage':<45} {'Allocated':>10} {'Peak':>10} {'ΔPeak':>10} {'ΔAlloc':>10}")
        print("-" * 100)
        for r in self.records:
            print(
                f"{r['tag']:<45} "
                f"{r['allocated_mb']:>9.1f}M "
                f"{r['peak_mb']:>9.1f}M "
                f"{r['delta_peak_mb']:>+9.1f}M "
                f"{r['delta_alloc_mb']:>+9.1f}M"
            )
        print("=" * 100)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
        print(f"VRAM breakdown saved to: {path}")


def main():
    tracker = VRAMTracker()

    # ---- Load config ----
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    # ---- Load static data ----
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
    tracker.checkpoint("after_static_mask_load")

    # ---- Load checkpoint ----
    fp16_ckpt_path = os.path.join(cfg.checkpoint_dir, "model_fp16.pth")
    if not os.path.exists(fp16_ckpt_path):
        print(f"ERROR: checkpoint not found at {fp16_ckpt_path}")
        return

    ckpt = torch.load(fp16_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    tracker.checkpoint("after_checkpoint_load_cpu")

    # ---- Infer model profile ----
    model_profile = ckpt.get("model_profile", {})
    if not model_profile:
        # Fallback to pgw_lite_pruned_96
        profiles = getattr(cfg, "student_profiles", {})
        p = profiles.get("pgw_lite_pruned_96")
        model_profile = {
            "name": "pgw_lite_pruned_96",
            "patch_size": [int(v) for v in p.patch_size],
            "embed_dim": int(p.embed_dim),
            "num_heads": [int(v) for v in p.num_heads],
            "window_size": [int(v) for v in cfg.window_size],
        }
    else:
        model_profile["patch_size"] = [int(v) for v in model_profile["patch_size"]]
        model_profile["num_heads"] = [int(v) for v in model_profile["num_heads"]]
        model_profile["window_size"] = [int(v) for v in model_profile.get("window_size", cfg.window_size)]
        if "depth_blocks" in model_profile and model_profile["depth_blocks"] is not None:
            model_profile["depth_blocks"] = [int(v) for v in model_profile["depth_blocks"]]

    print(f"Profile: {model_profile['name']}, embed={model_profile['embed_dim']}, patch={model_profile['patch_size']}")

    # ---- Build model ----
    model = build_pangu_model(
        img_size=cfg_data.dataset.img_size,
        patch_size=model_profile["patch_size"],
        embed_dim=model_profile["embed_dim"],
        num_heads=model_profile["num_heads"],
        window_size=model_profile["window_size"],
        depth_blocks=model_profile.get("depth_blocks", None),
        recompute_skip=False,
        layerwise_inference=True,
        layerwise_empty_cache=False,
        chunked_attention=True,
        attention_chunk_size=3,
    )
    model.half()
    model = model.to("cuda:0")
    tracker.checkpoint("after_model_build_to_cuda")

    # ---- Load weights (incremental dequantize) ----
    model_state = model.state_dict()
    loaded = 0
    with torch.no_grad():
        for key, value in state_dict.items():
            if key.endswith("_scale"):
                continue
            if key not in model_state:
                continue
            target = model_state[key]
            # Dequantize INT8
            if key.endswith(".weight") and value.dtype == torch.int8:
                scale_key = key + "_scale"
                if scale_key in state_dict:
                    scale = state_dict[scale_key].to(device=target.device, dtype=torch.float32)
                    if scale.ndim == 1 and value.ndim == 2:
                        scale = scale.view(-1, 1)
                    loaded_val = (
                        value.to(device=target.device, dtype=torch.float32) * scale
                    ).to(target.dtype)
                    target.copy_(loaded_val)
                    del loaded_val
                    loaded += 1
                    continue
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded += 1
    print(f"Loaded {loaded} tensors incrementally")
    model.eval()
    tracker.checkpoint("after_weight_load")

    # ---- Cleanup checkpoint ----
    del state_dict, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    tracker.checkpoint("after_checkpoint_cleanup")

    # ---- Count model parameters ----
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"Model params: {total_params:,} ({total_bytes/1024**2:.1f} MB)")
    print(f"Model buffers: {total_buffers/1024**2:.1f} MB")
    print(f"Total model resident: {(total_bytes+total_buffers)/1024**2:.1f} MB")

    # ---- Load one test sample ----
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()

    for data in test_dataloader:
        invar = data[0]
        break
    del datapipe, test_dataloader
    gc.collect()

    # ---- Prepare input (matches inference.py exactly) ----
    tracker.reset_peak("before_inference")
    tracker.checkpoint("baseline_before_inference")

    invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float16)
    tracker.checkpoint("invar_surface_to_gpu")

    invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float16)
    tracker.checkpoint("invar_upper_air_to_gpu")

    invar_surface_with_mask = torch.concat([invar_surface, surface_mask], dim=1)
    tracker.checkpoint("surface_concat_with_mask")

    invar_upper_air_reshaped = invar_upper_air.reshape(
        invar_upper_air.shape[0], 5, 13,
        invar_upper_air.shape[2], invar_upper_air.shape[3]
    )
    tracker.checkpoint("upper_air_reshape")

    model_input = [invar_surface_with_mask, invar_upper_air_reshaped]

    # ---- Print input tensor sizes ----
    print(f"\nInput tensor sizes:")
    for name, t in [
        ("surface_mask", surface_mask),
        ("invar_surface", invar_surface),
        ("invar_upper_air", invar_upper_air),
        ("invar_surface_with_mask", invar_surface_with_mask),
        ("invar_upper_air_reshaped", invar_upper_air_reshaped),
    ]:
        print(f"  {name}: {list(t.shape)} = {t.numel()*t.element_size()/1024**2:.1f} MB")
        
    del invar_surface, invar_upper_air, invar_surface_with_mask, invar_upper_air_reshaped
    if 't' in locals():
        del t, name
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Instrumented forward pass ----
    # Reset peak before the actual timed forward
    tracker.reset_peak("before_forward")
    tracker.checkpoint("before_model_forward")

    with torch.inference_mode():
        # Now run the instrumented layerwise forward
        # We manually replicate _forward_layerwise to get per-stage VRAM
        from pangu_profile_model import (
            _embed_sequence,
            _run_fuser_layerwise,
            _run_sample_layerwise,
            _recover_outputs,
        )

        sequence, Batch, PressureLevels, Height, Width = _embed_sequence(model, model_input)
        tracker.checkpoint("embed_sequence")
        print(f"  sequence shape: {list(sequence.shape)} = {sequence.numel()*sequence.element_size()/1024**2:.1f} MB")

        # We can now delete the input tensors to see if it helps
        # (but don't delete yet - measure first)

        sequence = _run_fuser_layerwise(model, model.layer1, sequence, False, None)
        tracker.checkpoint("layer1_forward")

        skip_sequence = sequence
        tracker.checkpoint("skip_sequence_saved")

        sequence = _run_sample_layerwise(model, model.downsample, sequence, False, None)
        tracker.checkpoint("downsample")

        sequence = _run_fuser_layerwise(model, model.layer2, sequence, False, None)
        tracker.checkpoint("layer2_forward")

        sequence = _run_fuser_layerwise(model, model.layer3, sequence, False, None)
        tracker.checkpoint("layer3_forward")

        sequence = _run_sample_layerwise(model, model.upsample, sequence, False, None)
        tracker.checkpoint("upsample")

        sequence = _run_fuser_layerwise(model, model.layer4, sequence, False, None)
        tracker.checkpoint("layer4_forward")

        # Skip concat
        sequence = torch.concat([sequence, skip_sequence], dim=-1)
        del skip_sequence
        tracker.checkpoint("skip_concat")

        # Recovery
        output_surface, output_upper_air = _recover_outputs(
            model, sequence, Batch, PressureLevels, Height, Width
        )
        tracker.checkpoint("recovery")

        # Output reshape
        del sequence
        tracker.checkpoint("after_del_sequence")

    # ---- Post-processing (matches inference.py) ----
    out_upper_air = output_upper_air.reshape(invar.shape[0], 65, invar.shape[2], invar.shape[3])
    pred_tensor = torch.concat(
        [output_surface.detach().cpu(), out_upper_air.detach().cpu()],
        dim=1,
    ).float()
    tracker.checkpoint("output_postprocess")

    # ---- Cleanup ----
    del output_surface, output_upper_air, out_upper_air, pred_tensor
    del model_input
    gc.collect()
    torch.cuda.empty_cache()
    tracker.checkpoint("final_cleanup")

    # ---- Final report ----
    print(f"\n{'='*60}")
    print(f"FINAL: Max VRAM = {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    print(f"{'='*60}")

    tracker.summary()

    # Save results
    os.makedirs("logs", exist_ok=True)
    tracker.save("logs/vram_breakdown.json")

    # ---- Optimization opportunity analysis ----
    print("\n" + "=" * 60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 60)

    print("""
Key areas to investigate based on theoretical analysis:

1. INPUT DATA: invar_upper_air [1,65,721,1440] FP16 = ~134 MB
   - This is transferred to GPU before model(invar) and stays alive
   - After patchembed3d, the raw input is no longer needed
   - Opportunity: ~130 MB if we can del input inside timing block

2. SKIP CONCAT: torch.concat([sequence, skip_sequence], dim=-1)
   - Creates a new [1,131040,192] FP16 tensor = ~48 MB
   - The split recovery path avoids this but was rejected for negligible gain
   - Re-evaluate with the current tighter baseline

3. ATTENTION PEAKS: QKV projection creates 3x output at once
   - Each EarthAttention3D block computes qkv = self.qkv(x) → 3*dim output
   - Chunked attention already helps but QKV projection is still full-width

4. RECOVERY OUTPUT: scored-only recovery creates [1,5,13,H,W]
   - Even scored-only still allocates the full output tensor
   - Could potentially compute and write scored channels incrementally
""")


if __name__ == "__main__":
    main()
