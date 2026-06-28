"""Post-Training Quantization (PTQ) to compress Pangu distilled model.

Quantizes only Linear layers' weights to INT8, leaving bias, LayerNorm, and Conv
layers in FP16 to maintain maximum prediction accuracy.
"""

import os
import sys
import torch

current_path = os.getcwd()
sys.path.append(current_path)

from onescience.utils.YParams import YParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pangu_profile_model import build_pangu_model


def cfg_list(value):
    return [int(v) for v in value]


def get_model_profile(cfg, profile_name):
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


def get_quantization_profile(cfg):
    profile_name = os.environ.get(
        "PANGU_QUANTIZE_PROFILE", getattr(cfg, "default_student_profile", "student_160")
    )
    return get_model_profile(cfg, profile_name)


def candidate_checkpoints(cfg, profile):
    name = profile["name"]
    if name == "student_160":
        return [
            os.path.join(cfg.checkpoint_dir, "model_distilled_latest.pth"),
            os.path.join(cfg.checkpoint_dir, cfg.distilled_train_checkpoint),
            os.path.join(cfg.checkpoint_dir, cfg.distilled_checkpoint),
        ]
    if name == "pgw_lite_patch8":
        return [
            os.path.join(cfg.checkpoint_dir, cfg.pgw_lite_distilled_latest_checkpoint),
            os.path.join(cfg.checkpoint_dir, cfg.pgw_lite_distilled_train_checkpoint),
            os.path.join(cfg.checkpoint_dir, cfg.pgw_lite_distilled_checkpoint),
        ]
    return [
        os.path.join(cfg.checkpoint_dir, f"model_{name}_latest.pth"),
        os.path.join(cfg.checkpoint_dir, f"model_{name}_train.pth"),
        os.path.join(cfg.checkpoint_dir, f"model_{name}_fp16.pth"),
    ]


def quantized_checkpoint_name(cfg, profile):
    name = profile["name"]
    if name == "student_160":
        return getattr(cfg, "quantized_checkpoint", "model_distilled_quantized.pth")
    if name == "pgw_lite_patch8":
        return cfg.pgw_lite_quantized_checkpoint
    return f"model_{name}_quantized.pth"


def quantize_per_output_channel(value):
    max_val = torch.amax(torch.abs(value), dim=1, keepdim=True)
    scale = torch.where(max_val > 0, max_val / 127.0, torch.ones_like(max_val))
    q_weight = torch.clamp(torch.round(value / scale), -128, 127).to(torch.int8)
    return q_weight, scale.to(torch.float16)


def main():
    config_path = os.path.join(os.getcwd(), "conf/config.yaml")
    cfg = YParams(config_path, "model")
    profile = get_quantization_profile(cfg)

    # Instantiate the target student structure so Linear keys match its profile.
    model = build_pangu_model(
        img_size=[721, 1440],  # AI4S benchmark resolution
        patch_size=profile["patch_size"],
        embed_dim=profile["embed_dim"],
        num_heads=profile["num_heads"],
        window_size=profile["window_size"],
        depth_blocks=profile.get("depth_blocks", None),
    )

    # Record all weight keys belonging to Linear layers
    linear_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_keys.add(name + ".weight")

    # Prioritize loading the latest training checkpoint, fallback to distilled FP16/train state.
    possible_paths = candidate_checkpoints(cfg, profile)
    distilled_path = None
    for p in possible_paths:
        if os.path.exists(p):
            distilled_path = p
            break

    if distilled_path is None:
        raise FileNotFoundError(f"No distilled checkpoint found in: {cfg.checkpoint_dir}")

    print(f"Loading distilled weights from: {distilled_path}")
    ckpt = torch.load(distilled_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Perform quantization
    quantized_state_dict = {}
    quantized_count = 0
    total_keys = len(state_dict)

    # Track largest non-quantized floating point weights
    unquantized_sizes = []

    for key, value in state_dict.items():
        # Dynamically strip DDP prefix 'module.' for both matching and final loading compatibility
        clean_key = key.replace("module.", "")
        
        # Drop static buffers that are re-created at __init__ to save massive space (~800MB)
        if clean_key.endswith("attn_mask") or clean_key.endswith("relative_position_index"):
            continue

        if clean_key in linear_keys:
            if torch.is_floating_point(value):
                q_weight, scale = quantize_per_output_channel(value)

                quantized_state_dict[clean_key] = q_weight
                quantized_state_dict[clean_key + "_scale"] = scale
                quantized_count += 1
                continue

        # Force all remaining floating point weights to FP16 to save space (since latest.pth might be FP32)
        if torch.is_floating_point(value):
            fp16_val = value.to(torch.float16)
            quantized_state_dict[clean_key] = fp16_val
            unquantized_sizes.append((clean_key, fp16_val.numel(), fp16_val.shape))
        else:
            quantized_state_dict[clean_key] = value

    print(f"Quantized {quantized_count} / {total_keys} weight keys to INT8.")

    # Print the top 10 largest non-quantized weights for analysis
    unquantized_sizes.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Top 10 Largest Non-Quantized Weights ---")
    for k, n, s in unquantized_sizes[:10]:
        print(f"{k}: shape={list(s)} size={n * 2 / 1024 / 1024:.2f} MB (FP16)")
    print("--------------------------------------------\n")

    # Save the quantized state dict
    quantized_ckpt = {
        "model_state_dict": quantized_state_dict,
        "quantization": {
            "method": "Linear-Weight-Only-PTQ-INT8",
            "scheme": "per_channel_int8",
            "scale_axis": 0,
            "quantized_keys_count": quantized_count,
            "target_profile": profile["name"],
            "target_embed_dim": profile["embed_dim"],
            "target_num_heads": tuple(profile["num_heads"]),
        },
        "model_profile": profile,
    }

    out_path = os.path.join(cfg.checkpoint_dir, quantized_checkpoint_name(cfg, profile))
    torch.save(quantized_ckpt, out_path)
    print(f"Successfully saved quantized model to: {out_path}")

    # Compare file sizes
    old_size = os.path.getsize(distilled_path) / (1024**2)
    new_size = os.path.getsize(out_path) / (1024**2)
    print(f"Original size: {old_size:.2f} MB")
    print(f"Quantized size: {new_size:.2f} MB")
    print(f"Size reduction: {(1.0 - new_size/old_size)*100:.2f}%")


if __name__ == "__main__":
    main()
