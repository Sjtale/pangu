"""
FP16 权重转换脚本
将 FP32 的 model_bak.pth 转换为 FP16 权重，同时清除优化器/调度器状态，
只保留推理所需的最小 state_dict。

用法:
    python scripts/convert_fp16.py [--checkpoint_dir ./data/checkpoints]

输出:
    {checkpoint_dir}/model_fp16.pth  — FP16 权重文件（仅含 model_state_dict）
"""

import torch
import os
import sys
import argparse


def convert_to_fp16(checkpoint_dir: str, backup_dir: str):
    local_src_path = os.path.join(checkpoint_dir, "model_bak.pth")
    backup_src_path = os.path.join(backup_dir, "model_bak.pth")
    src_path = local_src_path if os.path.exists(local_src_path) else backup_src_path
    dst_path = os.path.join(checkpoint_dir, "model_fp16.pth")

    if not os.path.exists(src_path):
        print(f"❌ 源文件不存在: {src_path}")
        sys.exit(1)

    # 统计原文件大小
    src_size_mb = os.path.getsize(src_path) / (1024 * 1024)
    print(f"📂 源文件: {src_path} ({src_size_mb:.1f} MB)")

    # 加载 checkpoint
    print("⏳ 加载 checkpoint ...")
    ckpt = torch.load(src_path, map_location="cpu")

    # 提取 model_state_dict（丢弃 optimizer、scheduler 等）
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        extra_keys = [k for k in ckpt.keys() if k != "model_state_dict"]
        if extra_keys:
            print(f"🗑️  丢弃非推理字段: {extra_keys}")
    else:
        # 可能直接是 state_dict
        state_dict = ckpt
        print("ℹ️  checkpoint 中未找到 'model_state_dict' 键，假定整体为 state_dict")

    # 统计参数
    total_params = sum(v.numel() for v in state_dict.values())
    fp32_count = sum(1 for v in state_dict.values() if v.dtype == torch.float32)
    print(f"📊 总参数量: {total_params:,}")
    print(f"📊 FP32 张量数: {fp32_count} / {len(state_dict)}")

    # 转换为 FP16
    print("⚡ 转换为 FP16 ...")
    fp16_state_dict = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            fp16_state_dict[k] = v.half()
        else:
            # 保持整数类型不变（如有）
            fp16_state_dict[k] = v

    # 只保存 model_state_dict，不包装额外字段
    torch.save({"model_state_dict": fp16_state_dict}, dst_path)

    dst_size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    reduction = (1 - dst_size_mb / src_size_mb) * 100

    print(f"✅ FP16 权重已保存: {dst_path}")
    print(f"📦 原文件大小:  {src_size_mb:.1f} MB")
    print(f"📦 新文件大小:  {dst_size_mb:.1f} MB")
    print(f"📉 体积缩减:    {reduction:.1f}%")
    print()
    print("接下来在 inference.py 中使用 model_fp16.pth 加载模型即可。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Pangu-Weather 权重从 FP32 转换为 FP16")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./data/checkpoints",
        help="checkpoint 目录路径 (默认: ./data/checkpoints)",
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        default="./pangu_backups",
        help="官方 FP32 备份目录 (默认: ./pangu_backups)",
    )
    args = parser.parse_args()
    convert_to_fp16(args.checkpoint_dir, args.backup_dir)
