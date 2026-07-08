import os
import sys
import json
import argparse
import torch
import torch.nn as nn

# Setup path
current_path = os.getcwd()
sys.path.append(current_path)

from onescience.utils.YParams import YParams
from pangu_profile_model import build_pangu_model

def cfg_list(value):
    return [int(v) for v in value]

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
        "patch_size": cfg_list(profile.patch_size),
        "embed_dim": int(profile.embed_dim),
        "num_heads": cfg_list(profile.num_heads),
        "window_size": cfg_list(cfg.window_size),
    }
    if hasattr(profile, "depth_blocks"):
        res["depth_blocks"] = cfg_list(profile.depth_blocks)
    return res

def quantize_per_output_channel(value):
    max_val = torch.amax(torch.abs(value), dim=1, keepdim=True)
    scale = torch.where(max_val > 0, max_val / 127.0, torch.ones_like(max_val))
    q_weight = torch.clamp(torch.round(value / scale), -128, 127).to(torch.int8)
    return q_weight, scale.to(torch.float16)

def main():
    parser = argparse.ArgumentParser(description="Sensitivity-Guided Mixed Precision PTQ")
    parser.add_argument("--keep-count", type=int, default=5, help="Number of top sensitive layers to keep in FP16")
    parser.add_argument("--keep-ratio", type=float, default=None, help="Ratio of top sensitive layers to keep in FP16 (overrides keep-count)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the FP16 checkpoint to quantize")
    args = parser.parse_args()

    config_path = os.path.join(os.getcwd(), "conf/config.yaml")
    cfg = YParams(config_path, "model")
    profile = get_quantization_profile(cfg)

    sensitivity_path = "./data/quant_sensitivity.json"
    if not os.path.exists(sensitivity_path):
        raise FileNotFoundError(f"Sensitivity ranked file not found at: {sensitivity_path}. Please run analyze_quant_sensitivity.py first.")

    with open(sensitivity_path, "r") as f:
        sensitivity_results = json.load(f)

    # Determine keep list
    total_layers = len(sensitivity_results)
    keep_count = args.keep_count
    if args.keep_ratio is not None:
        keep_count = int(total_layers * args.keep_ratio)
    keep_count = max(0, min(keep_count, total_layers))

    keep_names = set(r["name"] for r in sensitivity_results[:keep_count])
    print(f"Keeping the top {keep_count} / {total_layers} most sensitive layers in FP16:")
    for i, name in enumerate(sensitivity_results[:keep_count], start=1):
        print(f"  {i}. {name['name']} (MSE: {name['mse_deviation']:.4e})")

    dequant_ckpt_path = args.checkpoint
    if dequant_ckpt_path is None:
        candidates = [
            "./data/checkpoints/model_pgw_lite_pruned_96_fp16.pth",
            "./data/checkpoints/model_pgw_lite_pruned_96_latest.pth",
            "./data/checkpoints/model_dequantized_fp16.pth"
        ]
        for c in candidates:
            if os.path.exists(c):
                dequant_ckpt_path = c
                break
        if dequant_ckpt_path is None:
            raise FileNotFoundError("Could not find any default FP16 checkpoints in ./data/checkpoints/. Please specify --checkpoint.")

    print(f"\nLoading FP16 weights from: {dequant_ckpt_path}")
    ckpt = torch.load(dequant_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Instantiate model structure to find all Linear modules
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
        img_size=[721, 1440],
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

    # Record all weight keys belonging to Linear layers
    linear_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_keys.add(name + ".weight")

    # Perform mixed quantization
    mixed_state_dict = {}
    quantized_count = 0
    fp16_count = 0
    total_keys = len(state_dict)

    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")

        # Handle block sharing
        if share_deep_blocks == "layer2_to_layer3" and clean_key.startswith("layer3."):
            continue

        # Drop static buffers
        if clean_key.endswith("attn_mask") or clean_key.endswith("relative_position_index"):
            continue

        if clean_key in linear_keys:
            module_name = clean_key[:-len(".weight")]
            if module_name in keep_names:
                # Keep in FP16
                mixed_state_dict[clean_key] = value.to(torch.float16)
                fp16_count += 1
                continue
            else:
                # Quantize to INT8
                if torch.is_floating_point(value):
                    q_weight, scale = quantize_per_output_channel(value)
                    mixed_state_dict[clean_key] = q_weight
                    mixed_state_dict[clean_key + "_scale"] = scale
                    quantized_count += 1
                    continue

        # Force all remaining weights (LN, Conv, bias) to FP16
        if torch.is_floating_point(value):
            mixed_state_dict[clean_key] = value.to(torch.float16)
        else:
            mixed_state_dict[clean_key] = value

    print(f"\nMixed Quantization finished:")
    print(f"  - FP16 Linear layers: {fp16_count}")
    print(f"  - INT8 Linear layers: {quantized_count}")

    # Save checkpoint
    out_name = f"model_{profile['name']}_quantized.pth"
    if profile["name"] == "pgw_lite_pruned_96":
        out_name = "model_fp16.pth"  # Overwrite model_fp16.pth directly for verification

    out_path = os.path.join(cfg.checkpoint_dir, out_name)

    mixed_ckpt = {
        "model_state_dict": mixed_state_dict,
        "quantization": {
            "method": "Sensitivity-Guided-Mixed-Precision-PTQ",
            "scheme": "mixed_channel_int8_fp16",
            "fp16_keep_count": fp16_count,
            "quantized_keys_count": quantized_count,
            "target_profile": profile["name"],
        },
        "model_profile": ckpt.get("model_profile", profile),
    }

    # Backup the original quantized pth if overwriting model_fp16.pth
    if out_name == "model_fp16.pth":
        orig_backup_path = os.path.join(cfg.checkpoint_dir, "model_fp16.pth.orig_quant")
        if not os.path.exists(orig_backup_path):
            import shutil
            shutil.copy2(out_path, orig_backup_path)
            print(f"Backed up original quantized model to: {orig_backup_path}")

    torch.save(mixed_ckpt, out_path)
    print(f"Successfully saved mixed-precision model to: {out_path}")

    # Print final size
    orig_size = os.path.getsize(dequant_ckpt_path) / (1024**2)
    new_size = os.path.getsize(out_path) / (1024**2)
    print(f"Mathematical FP16 size: {orig_size:.2f} MB")
    print(f"Mixed-Precision size: {new_size:.2f} MB")
    print(f"Size reduction compared to FP16: {(1.0 - new_size/orig_size)*100:.2f}%")

if __name__ == "__main__":
    main()
