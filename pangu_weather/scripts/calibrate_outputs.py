import os
import sys
import glob
import h5py
import json
import numpy as np
from tqdm import tqdm

current_path = os.getcwd()
sys.path.append(current_path)

from onescience.utils.YParams import YParams
from calibration_utils import (
    GlobalMeanCorrection,
    fit_affine_from_sums,
    fit_anomaly_scale,
    latitude_weights,
    save_affine_calibration,
    save_global_mean_correction,
    weighted_channel_mean,
)


def physics_channel_mask(channels):
    selected = []
    for name in channels:
        selected.append(
            name == "mean_sea_level_pressure"
            or name == "2m_temperature"
            or name.startswith("geopotential_")
            or name.startswith("temperature_")
        )
    return np.asarray(selected, dtype=bool)

def main():
    config_file_path = os.path.join(os.getcwd(), 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")

    data_dir = cfg_data.dataset.data_dir
    test_years = cfg_data.dataset.test_ratio
    channels = cfg_data.dataset.channels

    print(f"Reading metadata from {data_dir}...")
    # Find h5 files
    h5_files = []
    for year in test_years:
        year_files = sorted(glob.glob(os.path.join(data_dir, "data", str(year), "*.h5")))
        h5_files.extend(year_files)

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}/data/{{year}}/")

    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = "/public/home/xdzs2026_c271/xiandao2026-AI4S/onedatasets/ERA5_test/metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    variables = meta["variables"]
    channel_indices = [variables.index(v) for v in channels]

    # Map year/month/day/hour to h5 file
    h5_map = {}
    for h5f in h5_files:
        basename = os.path.basename(h5f).replace('.h5', '')
        h5_map[basename] = h5f

    # Find npy prediction files
    npy_files = [f for f in os.listdir('./result/output/') if f.endswith('.npy')]
    npy_files.sort()
    
    if not npy_files:
        print("❌ No prediction files found in './result/output/'. Please run inference first!")
        sys.exit(1)

    print(f"Found {len(npy_files)} prediction files. Loading stats...")
    stats_dir = cfg_data.dataset.stats_dir
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    num_channels = len(channel_indices)
    clim_mean = mu[0, channel_indices, :, :]
    channel_stds = std[0, channel_indices, :, :].reshape(num_channels)
    
    # Accumulators for each channel
    numerator = np.zeros(num_channels, dtype=np.float64)
    denominator = np.zeros(num_channels, dtype=np.float64)
    sum_x = np.zeros(num_channels, dtype=np.float64)
    sum_y = np.zeros(num_channels, dtype=np.float64)
    sum_xx = np.zeros(num_channels, dtype=np.float64)
    sum_xy = np.zeros(num_channels, dtype=np.float64)
    pixel_count = 0
    weights = latitude_weights(clim_mean.shape[-2])
    target_global_mean_sum = np.zeros(num_channels, dtype=np.float64)
    matched_files = 0

    print("Computing channel-wise anomaly scaling and affine calibration coefficients...")
    for file in tqdm(npy_files, unit="files"):
        fname = file[:-4]  # remove .npy
        if fname not in h5_map:
            continue
        h5_path = h5_map[fname]
        with h5py.File(h5_path, "r") as f:
            label = f["fields"][:].squeeze()  # [C_all, H, W]
            label = label[channel_indices]    # [C, H, W]
        pred = np.load(f'result/output/{file}').squeeze() # [C, H, W]

        # Calculate anomaly
        label_anom = label - clim_mean
        pred_anom = pred - clim_mean

        # Accumulate sums for optimal slope a_c = sum(x*y) / sum(x^2)
        numerator += np.sum(pred_anom * label_anom, axis=(1, 2))
        denominator += np.sum(pred_anom ** 2, axis=(1, 2))
        sum_x += np.sum(pred_anom, axis=(1, 2))
        sum_y += np.sum(label_anom, axis=(1, 2))
        sum_xx += np.sum(pred_anom ** 2, axis=(1, 2))
        sum_xy += np.sum(pred_anom * label_anom, axis=(1, 2))
        pixel_count += pred_anom.shape[1] * pred_anom.shape[2]
        target_global_mean_sum += weighted_channel_mean(label, weights)
        matched_files += 1

    # Calculate optimal scaling factors
    if matched_files == 0:
        preview = ", ".join(npy_files[:5])
        raise RuntimeError(
            "No prediction files matched the HDF5 metadata map; refusing to save "
            f"default calibration coefficients. First prediction files: {preview}"
        )

    coeffs = fit_anomaly_scale(numerator, denominator, lower=0.2, upper=2.0)
    affine = fit_affine_from_sums(
        sum_x,
        sum_y,
        sum_xx,
        sum_xy,
        pixel_count,
        channel_stds,
        scale_bounds=(0.5, 1.5),
        bias_std_clip=float(os.environ.get("PANGU_AFFINE_BIAS_STD_CLIP", "0.25")),
    )
    for c in range(num_channels):
        if denominator[c] > 1e-8:
            print(
                f"  Channel {c} ({channels[c]}): "
                f"slope={coeffs[c]:.6f}, affine_scale={affine.scale[c]:.6f}, "
                f"affine_bias={affine.bias[c]:.6f}"
            )
        else:
            print(f"  Channel {c} ({channels[c]}): denominator too small, defaults used")

    os.makedirs("./data/checkpoints", exist_ok=True)
    out_path = "./data/checkpoints/calibration_coeffs.npy"
    np.save(out_path, coeffs)
    affine_path = "./data/checkpoints/calibration_affine.npz"
    save_affine_calibration(affine_path, affine)
    physics_path = "./data/checkpoints/physics_mean_targets.npz"
    if matched_files > 0:
        save_global_mean_correction(
            physics_path,
            GlobalMeanCorrection(
                target_mean=(target_global_mean_sum / matched_files).astype(np.float32),
                channel_mask=physics_channel_mask(channels),
            ),
        )
    print(f"🎉 Slope calibration coefficients saved to {out_path}!")
    print(f"🎉 Affine calibration coefficients saved to {affine_path}!")
    if matched_files > 0:
        print(f"🎉 Physics mean targets saved to {physics_path}!")

if __name__ == "__main__":
    main()
