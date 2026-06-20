import json
import numpy as np
from pathlib import Path


"""
生成/ERA5_test/stats文件夹下的global_means.npy 和 global_stds.npy
再手动从./global_means.npy
      ./ERA5_test/stats/global_stds.npy
      添加到./stats/目录下
      
注意目录./stats_details/下global_means.npy和global_stds.npy是(1, 99, 1 , 1)通道, 区别所需要的(1, 69, 1 , 1)，不可直接复制

用法:
python build_global_stats.py 

输入:
    surface.nc  — ERA5 single-levels: msl, u10, v10, t2m (means, stds)
    pressure.nc — ERA5 pressure-levels: z, q, t, u, v @ 13 levels (means, stds)

输出:
    {out}/
      metadata.json
      ./global_means.npy
      ./stats/global_stds.npy
"""


BASE_DIR = Path(__file__).resolve().parent
STATS_DIR = BASE_DIR / "stats_details"

# 1. Read variables from metadata.json
with open(BASE_DIR / "ERA5_test/metadata.json", "r") as f:
    metadata = json.load(f)

variables = metadata["variables"]
print(f"Found {len(variables)} variables in metadata.json")

# 2. Collect means and stds for each variable (in order)
means_list = []
stds_list = []
missing = []

for var in variables:
    mean_path = STATS_DIR / f"{var}_means.npy"
    std_path = STATS_DIR / f"{var}_stds.npy"
    if mean_path.exists() and std_path.exists():
        mean_val_suguang_gen = np.load(mean_path)       # shape (1, 1, 1, 1)
        std_val_suguang_gen = np.load(std_path)          # shape (1, 1, 1, 1)
        means_list.append(mean_val_suguang_gen)
        stds_list.append(std_val_suguang_gen)
    else:
        missing.append(var)
        print(f"WARNING: missing stats for '{var}'")

if missing:
    raise FileNotFoundError(f"Missing stats files for {len(missing)} variables: {missing}")

# 3. Stack along axis=1 -> shape (1, 69, 1, 1)
global_means = np.concatenate(means_list, axis=1)
global_stds = np.concatenate(stds_list, axis=1)

print(f"global_means shape: {global_means.shape}")
print(f"global_stds shape: {global_stds.shape}")

# 4. Save to current directory
np.save(BASE_DIR / "global_means.npy", global_means)
np.save(BASE_DIR / "global_stds.npy", global_stds)

print("Saved global_means.npy and global_stds.npy")
