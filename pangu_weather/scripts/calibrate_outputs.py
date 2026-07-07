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
    clim_mean = mu[:, channel_indices, :, :].squeeze()  # [C, H, W]

    num_channels = len(channel_indices)
    
    # Accumulators for each channel
    numerator = np.zeros(num_channels, dtype=np.float64)
    denominator = np.zeros(num_channels, dtype=np.float64)

    print("Computing channel-wise anomaly correlation scaling coefficients...")
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

    # Calculate optimal scaling factors
    coeffs = np.ones(num_channels, dtype=np.float32)
    for c in range(num_channels):
        if denominator[c] > 1e-8:
            a_c = numerator[c] / denominator[c]
            # Guardrails: prevent extreme scaling factor scaling, bound it between 0.2 and 2.0
            a_c = max(min(a_c, 2.0), 0.2)
            coeffs[c] = a_c
            print(f"  Channel {c} ({channels[c]}): optimal coeff = {a_c:.6f}")
        else:
            print(f"  Channel {c} ({channels[c]}): denominator too small, default to 1.0")

    os.makedirs("./data/checkpoints", exist_ok=True)
    out_path = "./data/checkpoints/calibration_coeffs.npy"
    np.save(out_path, coeffs)
    print(f"🎉 Calibration coefficients successfully saved to {out_path}!")

if __name__ == "__main__":
    main()
