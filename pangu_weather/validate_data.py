#!/usr/bin/env python3
"""
数据完整性 & 语义一致性校验脚本

在训练或推理前运行，检查：
  1. 数据目录结构完整性（data / stats / static / metadata）
  2. HDF5 文件命名规范 & 6h 时间间隔
  3. 变量通道数是否与 config.yaml 一致
  4. 时间连续性（无缺失）

用法:
    python validate_data.py                        # 读取 conf/config.yaml
    python validate_data.py --config conf/config.yaml
    python validate_data.py --data-dir /path/to/h5 --img-size 240 240

退出码:
    0 — 全部通过
    1 — 发现警告（仍可运行，但需注意）
    2 — 发现错误（不可运行）
"""

import os
import sys
import json
import glob
import argparse
import re
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# 常量
# ============================================================
EXPECTED_TIME_STEP_H = 6
EXPECTED_CHANNELS = 69  # 4 surface + 5×13 upper-air
SURFACE_VARS = 4
UPPER_AIR_VARS = 5
PRESSURE_LEVELS = 13


# ============================================================
# 工具函数
# ============================================================

def load_config(config_path):
    """从 config.yaml 读取 data_dir 等配置。"""
    try:
        import yaml
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        dp = raw.get("datapipe", {}).get("dataset", {})
        return {
            "data_dir": dp.get("data_dir"),
            "stats_dir": dp.get("stats_dir"),
            "static_dir": dp.get("static_dir"),
            "channels": dp.get("channels", []),
            "img_size": dp.get("img_size"),
        }
    except Exception as e:
        print(f"  ⚠️  无法解析 config.yaml: {e}")
        return {}


def parse_yyyymmddhh(filename):
    """尝试解析 YYYYMMDDHH 格式文件名，返回 datetime 或 None。"""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})\.h5$", filename)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]))
        except ValueError:
            return None
    return None


def parse_step_nnnn(filename):
    """尝试解析 YYYY_stepNNNN 格式文件名，返回 (year, step) 或 None。"""
    m = re.match(r"^(\d{4})_step(\d{4})\.h5$", filename)
    if m:
        return int(m[1]), int(m[2])
    return None


# ============================================================
# 检查项
# ============================================================

def check_directory_structure(paths, errors, warnings):
    """检查 data/stats/static/metadata.json 是否存在。"""
    data_dir = paths["data_dir"]
    stats_dir = paths["stats_dir"]
    static_dir = paths["static_dir"]

    if not os.path.isdir(data_dir):
        errors.append(f"data_dir 不存在: {data_dir}")
        return

    # 检查 data 目录下是否有年份子目录
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    if not subdirs:
        # 可能 data/ 下直接放的年目录，也可能是 data/data/ 嵌套
        nested = os.path.join(data_dir, "data")
        if os.path.isdir(nested):
            data_dir = nested
            paths["data_dir"] = data_dir
            subdirs = [d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]

    if not subdirs:
        errors.append(f"data_dir 下无年份子目录 (YYYY/): {data_dir}")
        return

    paths["years"] = sorted(subdirs)
    print(f"  data_dir: {data_dir}")
    print(f"  年份: {paths['years']}")

    # stats
    if not os.path.isdir(stats_dir):
        errors.append(f"stats_dir 不存在: {stats_dir}")
    else:
        mu = os.path.join(stats_dir, "global_means.npy")
        sd = os.path.join(stats_dir, "global_stds.npy")
        if not os.path.exists(mu) or not os.path.exists(sd):
            errors.append(f"stats/ 下缺少 global_means.npy 或 global_stds.npy")
        else:
            mu_shape = np.load(mu).shape
            sd_shape = np.load(sd).shape
            print(f"  global_means: {mu_shape}, global_stds: {sd_shape}")

    # static
    if not os.path.isdir(static_dir):
        errors.append(f"static_dir 不存在: {static_dir}")
    else:
        for name in ["land_mask", "soil_type", "topography"]:
            if not os.path.exists(os.path.join(static_dir, f"{name}.npy")):
                warnings.append(f"static/ 下缺少 {name}.npy")

    # metadata.json（在 data_dir 根目录）
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        # 可能在上层
        meta_path = os.path.join(os.path.dirname(data_dir), "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"  metadata.json: {len(meta.get('variables', []))} variables, "
              f"years={meta.get('years', [])}")
        paths["meta_path"] = meta_path
        paths["meta"] = meta


def check_file_naming_and_intervals(paths, errors, warnings):
    """扫描所有 HDF5 文件，判断命名格式并检查 6h 间隔。"""
    data_dir = paths["data_dir"]
    years = paths.get("years", [])
    if not years:
        return

    all_files = []
    format_type = None  # "yyyymmddhh" or "step"

    for year in years:
        year_dir = os.path.join(data_dir, str(year))
        files = sorted(glob.glob(os.path.join(year_dir, "*.h5")))
        for fp in files:
            fname = os.path.basename(fp)
            dt = parse_yyyymmddhh(fname)
            step = parse_step_nnnn(fname)
            if dt:
                if format_type is None:
                    format_type = "yyyymmddhh"
                all_files.append((year, fp, fname, dt, None))
            elif step:
                if format_type is None:
                    format_type = "step"
                all_files.append((year, fp, fname, None, step))
            else:
                warnings.append(f"无法解析文件名: {fname}")

    if not all_files:
        errors.append("未找到任何 HDF5 文件")
        return

    paths["format_type"] = format_type
    paths["all_files"] = all_files
    paths["total_files"] = len(all_files)

    print(f"  文件总数: {len(all_files)}")
    print(f"  文件命名格式: {format_type}")

    per_year = defaultdict(int)
    for year, _, _, _, _ in all_files:
        per_year[year] += 1
    for y in sorted(per_year):
        print(f"    {y}: {per_year[y]} 个文件")

    # ── 检查间隔 (仅 yyyymmddhh 格式) ──
    if format_type == "yyyymmddhh":
        for year in years:
            year_files = [(dt, fp) for y, fp, fn, dt, _ in all_files
                          if y == year and dt is not None]
            year_files.sort()
            if len(year_files) < 2:
                continue
            gaps = []
            for i in range(1, len(year_files)):
                gap = (year_files[i][0] - year_files[i-1][0]).total_seconds() / 3600
                gaps.append(gap)
            min_gap, max_gap = min(gaps), max(gaps)
            if min_gap != EXPECTED_TIME_STEP_H or max_gap != EXPECTED_TIME_STEP_H:
                errors.append(
                    f"{year} 年时间间隔不一致: "
                    f"min={min_gap:.1f}h, max={max_gap:.1f}h, "
                    f"期望={EXPECTED_TIME_STEP_H}h"
                )
            else:
                print(f"    {year}: 间隔 {min_gap:.0f}h ✓")

    elif format_type == "step":
        warnings.append(
            "文件使用 _stepNNNN 命名，无法从文件名验证时间间隔。\n"
            "  请确认这些文件是由 process_era5.py --time-step 6 生成的。"
        )
        # 至少检查 step 序号连续
        for year in years:
            year_steps = [step for y, _, _, _, (sy, step) in all_files
                          if y == year]
            year_steps.sort()
            if not year_steps:
                continue
            if year_steps != list(range(year_steps[0], year_steps[-1] + 1)):
                warnings.append(f"{year} 年 step 序号不连续: {year_steps}")


def check_channel_consistency(paths, errors, warnings):
    """检查 HDF5 文件中的变量数量是否与预期一致。"""
    all_files = paths.get("all_files", [])
    if not all_files:
        return

    sample_path = all_files[0][1]
    with h5py.File(sample_path, "r") as f:
        shape = f["fields"].shape
        print(f"  样本 shape: {shape}  (C={shape[0]}, H={shape[1]}, W={shape[2]})")

    if shape[0] != EXPECTED_CHANNELS:
        errors.append(
            f"通道数不匹配: 文件={shape[0]}, 期望={EXPECTED_CHANNELS} "
            f"(4 surface + 5*13 upper-air)"
        )

    # 检查 img_size 是否与 config 一致
    cfg_img_size = paths.get("cfg_img_size")
    if cfg_img_size and len(cfg_img_size) == 2:
        h_cfg, w_cfg = cfg_img_size
        h_file, w_file = shape[1], shape[2]
        if h_cfg != h_file or w_cfg != w_file:
            warnings.append(
                f"img_size 不匹配: config.yaml={cfg_img_size}, 数据=[{h_file}, {w_file}]"
            )

    # 抽样检查后几个文件是否存在空值
    sample_count = min(5, len(all_files))
    for i in range(1, sample_count):
        fp = all_files[-(i)][1]
        with h5py.File(fp, "r") as f:
            arr = f["fields"][:]
        if np.any(np.isnan(arr)):
            errors.append(f"文件含 NaN: {os.path.basename(fp)}")
            break


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pangu-Weather 数据校验")
    parser.add_argument("--config", default="conf/config.yaml",
                        help="配置文件路径 (默认: conf/config.yaml)")
    parser.add_argument("--data-dir", default=None,
                        help="直接指定 HDF5 数据根目录（覆盖 config）")
    parser.add_argument("--img-size", nargs=2, type=int, default=None,
                        help="预期空间尺寸 H W (默认: 从 config 读取)")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"⚠️  未找到配置文件: {config_path}，将使用手动参数。")
        cfg = {}
    else:
        cfg = load_config(config_path)

    data_dir = args.data_dir or cfg.get("data_dir")
    if not data_dir:
        print("❌ 未指定 data_dir。请通过 --config 或 --data-dir 指定。")
        sys.exit(2)

    stats_dir = cfg.get("stats_dir", os.path.join(data_dir, "stats"))
    static_dir = cfg.get("static_dir", os.path.join(data_dir, "static"))
    img_size = args.img_size or cfg.get("img_size")

    paths = {
        "data_dir": data_dir,
        "stats_dir": stats_dir,
        "static_dir": static_dir,
        "cfg_img_size": img_size,
    }

    errors = []
    warnings = []

    print("=" * 60)
    print("Pangu-Weather 数据完整性校验")
    print("=" * 60)
    print()

    # ── 1. 目录结构 ──
    print("── 1. 目录结构 ──")
    check_directory_structure(paths, errors, warnings)

    # ── 2. 文件命名 & 间隔 ──
    print("\n── 2. 文件命名 & 时间间隔 ──")
    check_file_naming_and_intervals(paths, errors, warnings)

    # ── 3. 通道一致性 ──
    print("\n── 3. 通道一致性 ──")
    check_channel_consistency(paths, errors, warnings)

    # ── 汇总 ──
    print()
    print("=" * 60)

    if warnings:
        print(f"⚠️  警告 ({len(warnings)} 条):")
        for w in warnings:
            print(f"   - {w}")
        print()

    if errors:
        print(f"❌ 错误 ({len(errors)} 条):")
        for e in errors:
            print(f"   - {e}")
        print()
        print("请修复以上错误后再运行训练或推理。")
        sys.exit(2)

    if warnings:
        print("✅ 检查通过（有警告，仍可运行，但建议关注）")
        sys.exit(1)
    else:
        print("✅ 全部检查通过！数据就绪。")
        print()
        print("下一步:")
        print(f"  推理: python inference.py")
        print(f"  评估: python result.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
