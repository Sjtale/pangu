"""Post-Training Quantization (PTQ) to compress Pangu distilled model.

Quantizes only Linear layers' weights to INT8, leaving bias, LayerNorm, and Conv
layers in FP16 to maintain maximum prediction accuracy.
"""

import os
import sys
import torch

current_path = os.getcwd()
sys.path.append(current_path)

from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams


def main():
    config_path = os.path.join(os.getcwd(), "conf/config.yaml")
    cfg = YParams(config_path, "model")

    # Instantiate the pruned student structure (embed_dim=160)
    model = Pangu(
        img_size=[721, 1440],  # AI4S benchmark resolution
        patch_size=cfg.patch_size,
        embed_dim=cfg.pruned_embed_dim,
        num_heads=cfg.pruned_num_heads,
        window_size=cfg.window_size,
    )

    # Record all weight keys belonging to Linear layers
    linear_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_keys.add(name + ".weight")

    # Load distilled checkpoint
    distilled_path = os.path.join(cfg.checkpoint_dir, cfg.distilled_checkpoint)
    # If the compact inference checkpoint doesn't exist yet, fallback to the training state
    if not os.path.exists(distilled_path):
        distilled_path = os.path.join(cfg.checkpoint_dir, cfg.distilled_train_checkpoint)

    if not os.path.exists(distilled_path):
        # Fallback to the latest checkpoints or print error
        latest_path = os.path.join(cfg.checkpoint_dir, "model_distilled_latest.pth")
        if os.path.exists(latest_path):
            distilled_path = latest_path
        else:
            raise FileNotFoundError(f"Distilled checkpoint not found at: {distilled_path}")

    print(f"Loading distilled weights from: {distilled_path}")
    ckpt = torch.load(distilled_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Perform quantization
    quantized_state_dict = {}
    quantized_count = 0
    total_keys = len(state_dict)

    for key, value in state_dict.items():
        if key in linear_keys:
            if torch.is_floating_point(value):
                # Symmetric quantization to [-128, 127]
                max_val = torch.max(torch.abs(value))
                scale = max_val / 127.0 if max_val > 0 else torch.tensor(1.0)
                scale = scale.to(torch.float16)  # Save scale as float16 to keep size minimal

                q_weight = torch.clamp(torch.round(value / scale.to(value.device)), -128, 127).to(torch.int8)

                quantized_state_dict[key] = q_weight
                quantized_state_dict[key + "_scale"] = scale
                quantized_count += 1
                continue

        quantized_state_dict[key] = value

    print(f"Quantized {quantized_count} / {total_keys} weight keys to INT8.")

    # Save the quantized state dict
    quantized_ckpt = {
        "model_state_dict": quantized_state_dict,
        "quantization": {
            "method": "Linear-Weight-Only-PTQ-INT8",
            "quantized_keys_count": quantized_count,
            "target_embed_dim": cfg.pruned_embed_dim,
            "target_num_heads": tuple(cfg.pruned_num_heads),
        }
    }

    out_path = os.path.join(cfg.checkpoint_dir, "model_distilled_quantized.pth")
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
