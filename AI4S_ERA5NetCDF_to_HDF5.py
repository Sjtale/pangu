#!/usr/bin/env python3
"""
NetCDF → HDF5 转换脚本 — 将下载的 ERA5 NC 文件转为训练用的 HDF5 格式

将下载的 surface_china_region.nc + pressure_china_region.nc,
转换为 onescience ERA5Datapipe 需要的 per-timestep HDF5 格式，并生成 metadata.json。

用法:
python AI4S_ERA5NetCDF_to_HDF5.py 

输入:
    surface.nc  — ERA5 single-levels: msl, u10, v10, t2m
    pressure.nc — ERA5 pressure-levels: z, q, t, u, v @ 13 levels

输出:
    {out}/
      metadata.json
      stats/global_means.npy
      stats/global_stds.npy
      data/{year}/{year}{month}{day}{hour}.h5
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from pathlib import Path
from tqdm import tqdm

# ===================== 批量文件转化 =====================
# 在这里列出所有【surface, pressure】对
FILE_PAIRS = [
    ("1-surface_201604_0305_0107.nc", "1-pressure_201604_0305_0107.nc"),
    # ("2-surface_200201_full_00061218.nc", "2-pressure_200201_full_00061218.nc"),
    # ("3-surface_201006_09-15_05111723.nc", "3-pressure_201006_09-15_05111723.nc"),
    ("4-surface_197704_22-28_04101622.nc", "4-pressure_197704_22-28_04101622.nc"),
    # 可以无限加
    # ("/path/surface2.nc", "/path/pressure2.nc"),
    # ("/path/surface3.nc", "/path/pressure3.nc"),
]
INPUT_ROOT = Path("../onedatasets/")
OUTPUT_ROOT = Path("../onedatasets/ERA5_test/")  # 输出目录
TIME_STEP = 6                # 时间步

# ======================================

# ============================================================
# Pangu 需要的变量顺序 (与 config.yaml 中 channels 一致)
# ============================================================
SURFACE_ORDER = [
    ("msl", "mean_sea_level_pressure"),
    ("u10", "10m_u_component_of_wind"),
    ("v10", "10m_v_component_of_wind"),
    ("t2m", "2m_temperature"),
]

PRESSURE_ORDER = [
    ("z", "geopotential"),
    ("q", "specific_humidity"),
    ("t", "temperature"),
    ("u", "u_component_of_wind"),
    ("v", "v_component_of_wind"),
]

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

ALL_CHANNELS = (
    [name for _, name in SURFACE_ORDER]
    + [f"{name}_{lev}" for _, name in PRESSURE_ORDER for lev in LEVELS]
)


def get_time_name(ds):
    if "valid_time" in ds.coords:
        return "valid_time"
    if "time" in ds.coords:
        return "time"
    raise ValueError("Cannot find valid_time or time in dataset")


def normalize_level_value(x):
    return int(float(x))


def main():
    # 初始化全局统计量（只一次）
    sum_vals = np.zeros(len(ALL_CHANNELS), dtype=np.float64)
    sum_sq_vals = np.zeros(len(ALL_CHANNELS), dtype=np.float64)
    count_vals = np.zeros(len(ALL_CHANNELS), dtype=np.int64)
    all_years = set()

    for surface_path, pressure_path in FILE_PAIRS:
        surface_path = INPUT_ROOT / surface_path
        pressure_path = INPUT_ROOT / pressure_path
        for p in [surface_path, pressure_path]:
            if not p.exists():
                print(f"❌ 文件不存在: {p}")
                sys.exit(1)

        data_dir = OUTPUT_ROOT / "data"
        stats_dir = OUTPUT_ROOT / "stats"
        static_dir = OUTPUT_ROOT / "static"

        for d in [data_dir, stats_dir, static_dir]:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)

        # ========== 1. 打开 NetCDF ==========
        print("=" * 60)
        print("读取 NetCDF 文件...")
        print(f"  地表: {surface_path}")
        print(f"  高空: {pressure_path}")

        sfc = xr.open_dataset(surface_path)
        pl = xr.open_dataset(pressure_path)

        sfc_time_name = get_time_name(sfc)
        pl_time_name = get_time_name(pl)

        print(f"  地表维度: {dict(sfc.sizes)}")
        print(f"  高空维度: {dict(pl.sizes)}")

        # ========== 2. 验证变量 ==========
        for short_name, _ in SURFACE_ORDER:
            if short_name not in sfc.data_vars:
                raise ValueError(f"地表文件缺少变量: {short_name}")
        for short_name, _ in PRESSURE_ORDER:
            if short_name not in pl.data_vars:
                raise ValueError(f"高空文件缺少变量: {short_name}")

        available_levels = [normalize_level_value(x) for x in pl["pressure_level"].values]
        missing_levels = [x for x in LEVELS if x not in available_levels]
        if missing_levels:
            raise ValueError(f"缺少等压面: {missing_levels}")

        # ========== 3. 对齐时间和网格 ==========
        sfc_times = pd.to_datetime(sfc[sfc_time_name].values)
        pl_times = pd.to_datetime(pl[pl_time_name].values)
        common_times = sorted(set(sfc_times).intersection(set(pl_times)))

        print(f"\n  地表时间点数: {len(sfc_times)}")
        print(f"  高空时间点数: {len(pl_times)}")
        print(f"  共同时间点数: {len(common_times)}")
        if len(common_times) > 0:
            print(f"  时间范围: {common_times[0]} ~ {common_times[-1]}")

        if len(common_times) < 2:
            raise ValueError("共同时间点太少，地表和高空时间未对齐。")

        # ========== 3b. 按 TIME_STEP 过滤时间点 ========== # 
        base_hour = common_times[0].hour
        valid_hours = set((base_hour + i * TIME_STEP) % 24 for i in range(24 // TIME_STEP)) # 过滤：最小值+增加6的倍数的值
        filtered_times = [t for t in common_times if t.hour in valid_hours]
        skipped = len(common_times) - len(filtered_times)
        if skipped > 0:
            print(f"  ⚠️  按 TIME_STEP={TIME_STEP}h 过滤，跳过 {skipped} 个时间点")
            print(f"  基准小时: {base_hour:02d}:00，保留小时: {sorted(valid_hours)}")
        if len(filtered_times) < 2:
            raise ValueError(
                f"按 TIME_STEP={TIME_STEP}h 过滤后时间点不足 ({len(filtered_times)} 个)。\n"
                f"请检查 NC 文件中的时间步长是否与 --time-step 一致。"
            )
        common_times = filtered_times

        sfc_lat = sfc["latitude"].values
        pl_lat = pl["latitude"].values
        if len(sfc_lat) != len(pl_lat) or not np.allclose(sfc_lat, pl_lat):
            raise ValueError("地表和高空纬度不一致。")

        H, W = len(sfc_lat), len(sfc["longitude"])
        C = len(ALL_CHANNELS)
        print(f"  输出尺寸: [{C} ch, {H} H, {W} W]")

        # ========== 4. 统计年分布 ==========
        years_seen = sorted(set(t.year for t in common_times))
        print(f"  覆盖年份: {years_seen}")

        # ========== 5. 预加载所有变量到内存 ==========
        print(f"\n加载 {len(common_times)} 个时间步到内存...")

        # 建立时间到索引的映射
        time_to_idx = {np.datetime64(t): i for i, t in enumerate(common_times)}

        # 地表变量: [T, H, W]
        surface_data = {}
        for short_name, _ in SURFACE_ORDER:
            arr = sfc[short_name].values  # [T, H, W]         
            surface_data[short_name] = arr.astype(np.float32)
            print(f"  地表 {short_name}: {arr.shape}")

        # 高空变量: [T, P, H, W]
        pressure_data = {}
        for short_name, _ in PRESSURE_ORDER:
            arr = pl[short_name].values
            pressure_data[short_name] = arr.astype(np.float32)
            print(f"  高空 {short_name}: {arr.shape}")

        # 获取压力层索引映射
        level_to_idx = {normalize_level_value(lv): i for i, lv in enumerate(pl["pressure_level"].values)}

        # ========== 6. 逐时间步写入 HDF5 ==========
        print(f"\n写入 {len(common_times)} 个时间步的 HDF5 文件...")
        for idx, t in enumerate(tqdm(common_times, desc="Writing HDF5", unit="step")):
            year = t.year
            year_dir = data_dir / str(year)
            year_dir.mkdir(exist_ok=True)

            ts_str = t.strftime("%Y%m%d%H")
            out_path = year_dir / f"{ts_str}.h5"

            arr = np.empty((C, H, W), dtype=np.float32)
            ch = 0

            # 4 个地表变量
            for short_name, _ in SURFACE_ORDER:
                arr[ch] = surface_data[short_name][idx]
                ch += 1
  
            # 5 类 × 13 层高空变量
            for short_name, _ in PRESSURE_ORDER:
                for lev in LEVELS:
                    p_idx = level_to_idx[lev]
                    arr[ch] = pressure_data[short_name][idx, p_idx]
                    ch += 1

            with h5py.File(out_path, "w") as f:
                f.create_dataset("fields", data=arr)

            # 累积统计量
            flat = arr.reshape(C, -1)
            valid = ~np.isnan(flat)
            for i in range(C):
                xi = flat[i][valid[i]]
                if xi.size > 0:
                    sum_vals[i] += xi.sum()
                    sum_sq_vals[i] += (xi ** 2).sum()
                    count_vals[i] += xi.size

        # 添加年份
        all_years.update(years_seen)
        sfc.close()
        pl.close()

    # ========== 7. 生成静态文件 ==========，需要手动导入
    print(f"✅ 静态文件需添加 (手动)")

    # ========== 8. 生成 metadata.json ==========
    metadata = {
        "years": sorted([str(y) for y in all_years]),
        "variables": ALL_CHANNELS,
    }

    # meta_path = OUTPUT_ROOT / "metadata.json"
    # with open(meta_path, "w") as f:
    #     json.dump(metadata, f, indent=2, ensure_ascii=False)
    # print(f"✅ metadata.json 已生成: {len(ALL_CHANNELS)} 变量, {len(years_seen)} 年")

    # ========== 9. 汇总 ==========
    total_h5 = sum(1 for _ in data_dir.rglob("*.h5"))
    print()
    print("=" * 60)
    print("✅ 转换完成！")
    print(f"   HDF5 文件数: {total_h5}")
    print(f"   年份: {sorted(all_years)}")
    print(f"   输出目录: {OUTPUT_ROOT.absolute()}")
    print(f"   数据尺寸: [{C}, {H}, {W}]")
    print()
    print("下一步:")
    print(f"   需在 config.yaml 中设置:")
    print(f"     stats_dir: \"{OUTPUT_ROOT.absolute()}/stats/\"")
    print(f"     static_dir: \"{OUTPUT_ROOT.absolute()}/static/\"")
    print(f"     data_dir: \"{OUTPUT_ROOT.absolute()}/\"")
    print(f"     img_size: [{H}, {W}]")
    print(f"     train_years: {years_seen[:-2] if len(years_seen) > 2 else years_seen}")

if __name__ == "__main__":
    main()
