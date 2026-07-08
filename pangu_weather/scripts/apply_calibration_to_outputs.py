import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

current_path = os.getcwd()
sys.path.append(current_path)

from calibration_utils import (
    apply_affine_calibration,
    apply_global_mean_correction,
    load_affine_calibration,
    load_global_mean_correction,
)
from onescience.utils.YParams import YParams


def load_means(cfg_data):
    data_dir = cfg_data.dataset.data_dir
    stats_dir = cfg_data.dataset.stats_dir
    channels = cfg_data.dataset.channels
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        variables = json.load(f)["variables"]
    channel_indices = [variables.index(v) for v in channels]
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))
    return mu[:, channel_indices, :, :]


def main():
    parser = argparse.ArgumentParser(
        description="Apply saved calibration files to existing denormalized prediction npy files."
    )
    parser.add_argument("--input-dir", default="./result/output")
    parser.add_argument("--output-dir", default="./result/output_calibrated")
    parser.add_argument("--checkpoint-dir", default="./data/checkpoints")
    parser.add_argument(
        "--mode",
        choices=["auto", "affine", "slope"],
        default="auto",
        help="auto prefers calibration_affine.npz and falls back to calibration_coeffs.npy.",
    )
    parser.add_argument(
        "--global-mean-correction",
        action="store_true",
        help="Also apply optional physics_mean_targets.npz if present.",
    )
    args = parser.parse_args()

    config_file_path = os.path.join(os.getcwd(), "conf/config.yaml")
    cfg_data = YParams(config_file_path, "datapipe")
    means = load_means(cfg_data)
    num_channels = int(means.shape[1])

    affine = None
    slope = None
    affine_path = os.path.join(args.checkpoint_dir, "calibration_affine.npz")
    slope_path = os.path.join(args.checkpoint_dir, "calibration_coeffs.npy")
    if args.mode in {"auto", "affine"} and os.path.exists(affine_path):
        affine = load_affine_calibration(affine_path, num_channels)
        print(f"Loaded affine calibration: {affine_path}")
    elif args.mode in {"auto", "slope"} and os.path.exists(slope_path):
        slope = np.load(slope_path).reshape(1, num_channels, 1, 1)
        print(f"Loaded slope calibration: {slope_path}")
    else:
        raise FileNotFoundError(
            f"No requested calibration file found in {args.checkpoint_dir}"
        )

    correction = None
    physics_path = os.path.join(args.checkpoint_dir, "physics_mean_targets.npz")
    if args.global_mean_correction:
        correction = load_global_mean_correction(physics_path, num_channels)
        print(f"Loaded global mean correction: {physics_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No npy files found in {args.input_dir}")

    for filename in tqdm(files, unit="files"):
        pred = np.load(os.path.join(args.input_dir, filename))
        if affine is not None:
            pred = apply_affine_calibration(pred, means, affine)
        else:
            pred = means + slope * (pred - means)
        pred = apply_global_mean_correction(pred, correction)
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite calibrated output for {filename}")
        np.save(os.path.join(args.output_dir, filename), pred.astype(np.float32))

    print(f"Saved {len(files)} calibrated prediction files to {args.output_dir}")


if __name__ == "__main__":
    main()
