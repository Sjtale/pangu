import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

import argparse

# Setup path
current_path = os.getcwd()
sys.path.append(current_path)

from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe
from pangu_profile_model import build_pangu_model

def get_quantization_profile(cfg):
    profile_name = os.environ.get(
        "PANGU_QUANTIZE_PROFILE", getattr(cfg, "default_student_profile", "pgw_lite_pruned_96")
    )
    profiles = getattr(cfg, "student_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown model profile: {profile_name}")
    profile = profiles[profile_name]
    res = {
        "name": profile_name,
        "patch_size": [int(v) for v in profile.patch_size],
        "embed_dim": int(profile.embed_dim),
        "num_heads": [int(v) for v in profile.num_heads],
        "window_size": [int(v) for v in cfg.window_size],
    }
    if hasattr(profile, "depth_blocks"):
        res["depth_blocks"] = [int(v) for v in profile.depth_blocks]
    return res

def simulate_quantization(weight, bits=4):
    """Simulate low-bit quantization on weight to measure sensitivity."""
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    # Per-channel quantization
    max_val = torch.amax(torch.abs(weight), dim=1, keepdim=True)
    scale = torch.where(max_val > 0, max_val / float(qmax), torch.ones_like(max_val))

    q_weight = torch.clamp(torch.round(weight / scale), qmin, qmax)
    dequant_w = (q_weight * scale).to(weight.dtype)
    return dequant_w

def main():
    parser = argparse.ArgumentParser(description="Quantization Sensitivity Profiler")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the FP16 checkpoint to scan")
    args = parser.parse_args()

    config_path = os.path.join(os.getcwd(), "conf/config.yaml")
    cfg = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    profile = get_quantization_profile(cfg)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = [
            "./data/checkpoints/model_pgw_lite_pruned_96_fp16.pth",
            "./data/checkpoints/model_pgw_lite_pruned_96_latest.pth",
            "./data/checkpoints/model_dequantized_fp16.pth"
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break
        if checkpoint_path is None:
            print("Error: Could not find any default FP16 checkpoints in ./data/checkpoints/. Please specify --checkpoint.")
            return

    print(f"Loading FP16 checkpoint for sensitivity scan: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Build model structure
    if "model_profile" in ckpt and isinstance(ckpt["model_profile"], dict):
        profile = ckpt["model_profile"]
        profile["patch_size"] = [int(v) for v in profile["patch_size"]]
        profile["num_heads"] = [int(v) for v in profile["num_heads"]]
        profile["window_size"] = [int(v) for v in profile.get("window_size", cfg.window_size)]
        if "depth_blocks" in profile and profile["depth_blocks"] is not None:
            profile["depth_blocks"] = [int(v) for v in profile["depth_blocks"]]
        print(f"ℹ️ Loaded architecture profile '{profile['name']}' directly from checkpoint metadata.")

    use_gqa = any("q_proj" in k for k in state_dict.keys())
    use_swiglu = any("mlp.w1" in k for k in state_dict.keys())
    use_rmsnorm = any("norm1.weight" in k for k in state_dict.keys()) and not any("norm1.bias" in k for k in state_dict.keys())
    share_deep_blocks = profile.get("share_deep_blocks")

    model = build_pangu_model(
        img_size=cfg_data.dataset.img_size,
        patch_size=profile["patch_size"],
        embed_dim=profile["embed_dim"],
        num_heads=profile["num_heads"],
        window_size=profile["window_size"],
        depth_blocks=profile.get("depth_blocks", None),
        use_swiglu=use_swiglu,
        use_rmsnorm=use_rmsnorm,
        use_gqa=use_gqa,
        share_deep_blocks=share_deep_blocks,
    )

    # Load FP16 weights into model
    # We must average layer2/layer3 if block sharing is active
    if share_deep_blocks == "layer2_to_layer3":
        from distill_train import average_layer2_layer3_for_sharing
        state_dict = average_layer2_layer3_for_sharing(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model.half().to("cuda:0").eval()

    # Load a single sample from test dataloader
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)

    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to('cuda:0')
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1).half()

    print("Fetching test sample...")
    for data in test_dataloader:
        invar = data[0]
        break

    invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float16)
    invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float16)
    invar_surface_with_mask = torch.concat([invar_surface, surface_mask], dim=1)
    invar_upper_air_reshaped = invar_upper_air.reshape(
        invar_upper_air.shape[0], 5, 13, invar_upper_air.shape[2], invar_upper_air.shape[3]
    )
    invar_sample = (invar_surface_with_mask, invar_upper_air_reshaped)

    # Get reference predictions (unquantized)
    with torch.no_grad():
        ref_surf, ref_upper = model(invar_sample)

    # Find all linear layers to evaluate
    linear_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules[name] = module

    print(f"Found {len(linear_modules)} linear layers. Running INT4 sensitivity scan...")

    results = []
    # Use bits=4 to ensure non-zero quantization deviation on pre-quantized checkpoint
    bits = 4

    for idx, (name, module) in enumerate(linear_modules.items(), start=1):
        orig_w = module.weight.data.clone()

        # Apply simulated quantization
        sim_w = simulate_quantization(orig_w, bits=bits)
        module.weight.data.copy_(sim_w)

        # Run forward pass
        with torch.no_grad():
            surf, upper = model(invar_sample)

        # Compute MSE deviation
        surf_mse = torch.mean((surf - ref_surf) ** 2).item()
        upper_mse = torch.mean((upper - ref_upper) ** 2).item()
        total_mse = surf_mse + upper_mse

        results.append({
            "name": name,
            "mse_deviation": total_mse,
            "param_count": orig_w.numel()
        })

        # Restore original weight
        module.weight.data.copy_(orig_w)

        if idx % 10 == 0 or idx == len(linear_modules):
            print(f"Scanned {idx}/{len(linear_modules)} layers...")

    # Sort results by sensitivity descending
    results.sort(key=lambda x: x["mse_deviation"], reverse=True)

    # Save results to json
    out_dir = "./data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quant_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSensitivity ranking saved to: {out_path}")
    print("\n--- Top 15 Most Quantization-Sensitive Layers ---")
    for r_idx, r in enumerate(results[:15], start=1):
        print(f"{r_idx:2d}. {r['name']}: MSE={r['mse_deviation']:.4e} (Params: {r['param_count']})")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
