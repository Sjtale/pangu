"""Rank the fixed pruned-96 Linear layers with one deterministic INT4 scan."""

import argparse
import json
import os

import numpy as np
import torch
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams

from pangu_profile_model import build_pangu_model


PROFILE = {
    "name": "pgw_lite_pruned_96",
    "patch_size": [2, 8, 8],
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "depth_blocks": [2, 6, 6, 2],
    "window_size": [2, 6, 12],
}


def simulated_int4(weight):
    maximum = weight.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 7.0, torch.ones_like(maximum))
    return (weight.div(scale).round().clamp(-8, 7) * scale).to(weight.dtype)


def surface_mask(directory, device):
    arrays = [
        np.load(os.path.join(directory, name)).astype(np.float32)
        for name in ("land_mask.npy", "soil_type.npy", "topography.npy")
    ]
    arrays[2] = (arrays[2] - arrays[2].mean()) / (arrays[2].std() + 1e-6)
    return torch.from_numpy(np.stack(arrays)).unsqueeze(0).to(device).half()


def main():
    parser = argparse.ArgumentParser(description="Fixed INT4 sensitivity scan")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    output = "./data/quant_sensitivity.json"
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite sensitivity report: {output}")

    torch.manual_seed(42)
    np.random.seed(42)
    cfg_data = YParams("./conf/config.yaml", "datapipe")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_profile") != PROFILE:
        raise ValueError("Sensitivity input must be the fixed pgw_lite_pruned_96 checkpoint")
    state = checkpoint["model_state_dict"]
    model = build_pangu_model(
        img_size=cfg_data.dataset.img_size,
        patch_size=PROFILE["patch_size"],
        embed_dim=PROFILE["embed_dim"],
        num_heads=PROFILE["num_heads"],
        window_size=PROFILE["window_size"],
        depth_blocks=PROFILE["depth_blocks"],
    )
    model.load_state_dict(state, strict=True)
    model.half().to("cuda:0").eval()

    data = next(iter(ERA5Datapipe(params=cfg_data, distributed=False).test_dataloader()))
    inputs = data[0].to("cuda:0", dtype=torch.float16)
    mask = surface_mask(cfg_data.dataset.static_dir, inputs.device).expand(
        inputs.shape[0], -1, -1, -1
    )
    sample = (
        torch.cat((inputs[:, :4], mask), dim=1),
        inputs[:, 4:].reshape(inputs.shape[0], 5, 13, inputs.shape[2], inputs.shape[3]),
    )
    with torch.inference_mode():
        reference_surface, reference_upper = model(sample)

    results = []
    linear_modules = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    for name, module in linear_modules:
        original = module.weight.detach().clone()
        try:
            with torch.no_grad():
                module.weight.copy_(simulated_int4(original))
            with torch.inference_mode():
                surface, upper = model(sample)
            deviation = float(
                (surface.float() - reference_surface.float()).square().mean()
                + (upper.float() - reference_upper.float()).square().mean()
            )
        finally:
            with torch.no_grad():
                module.weight.copy_(original)
        results.append(
            {"name": name, "mse_deviation": deviation, "param_count": original.numel()}
        )

    results.sort(key=lambda item: (-item["mse_deviation"], item["name"]))
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Ranked {len(results)} Linear layers: {output}")


if __name__ == "__main__":
    main()
