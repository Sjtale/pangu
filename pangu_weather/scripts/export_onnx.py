"""
ONNX 导出 + onnxsim 图优化脚本
将 FP16 PyTorch 模型导出为 ONNX 格式，并使用 onnxsim 简化计算图。

用法 (在 pangu_weather 目录下运行):
    cd pangu_weather
    python scripts/export_onnx.py [--checkpoint_dir ./data/checkpoints]

输出:
    {checkpoint_dir}/model_fp16.onnx     — 原始 ONNX 模型
    {checkpoint_dir}/model_fp16_sim.onnx — onnxsim 简化后的 ONNX 模型（如果简化成功）

依赖:
    onnx >= 1.21.0, onnxsim >= 0.6.3 (比赛环境已提供)
"""

import torch
import onnx
import os
import sys
import argparse
import gc


def export_to_onnx(checkpoint_dir: str):
    # ---- 环境配置 (与 inference.py 保持一致) ----
    current_path = os.getcwd()
    sys.path.append(current_path)

    from onescience.models.pangu import Pangu
    from onescience.utils.YParams import YParams

    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    # ---- 加载模型 ----
    fp16_path = os.path.join(checkpoint_dir, "model_fp16.pth")
    local_fp32_path = os.path.join(checkpoint_dir, "model_bak.pth")
    backup_fp32_path = os.path.join(cfg.official_checkpoint_dir, "model_bak.pth")
    fp32_path = local_fp32_path if os.path.exists(local_fp32_path) else backup_fp32_path

    if os.path.exists(fp16_path):
        print(f"⚡ 加载 FP16 权重: {fp16_path}")
        ckpt = torch.load(fp16_path, map_location="cuda:0")
    elif os.path.exists(fp32_path):
        print(f"ℹ️  加载 FP32 权重: {fp32_path}")
        ckpt = torch.load(fp32_path, map_location="cuda:0")
    else:
        print(f"❌ 未找到权重文件: {fp16_path} 或 {fp32_path}")
        sys.exit(1)

    model = Pangu(
        img_size=cfg_data.dataset.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
    ).to("cuda:0")
    model.load_state_dict(ckpt["model_state_dict"])
    model.half()
    model.eval()

    # 释放 checkpoint 内存
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    print(f"📊 模型加载完成，当前显存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # ---- 构造 dummy 输入 ----
    # 输入格式: [batch, 72, 721, 1440]
    #   channels 0-3: 4 个地面变量
    #   channels 4-6: 3 个 static mask
    #   channels 7-71: 65 个高空变量
    dummy_input = torch.randn(
        1, 72,
        cfg_data.dataset.img_size[0],
        cfg_data.dataset.img_size[1],
        dtype=torch.float16, device="cuda:0",
    )

    # ---- 导出 ONNX ----
    onnx_path = os.path.join(checkpoint_dir, "model_fp16.onnx")

    print("⏳ 正在导出 ONNX 模型 (opset=15)，这可能需要几分钟...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=15,
            input_names=["input"],
            output_names=["output_surface", "output_upper_air"],
            do_constant_folding=True,
        )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ ONNX 模型已导出: {onnx_path} ({onnx_size_mb:.1f} MB)")

    # 释放 GPU 内存，留给 onnxsim
    del model, dummy_input
    gc.collect()
    torch.cuda.empty_cache()

    # ---- onnxsim 图简化 ----
    print("⏳ 使用 onnxsim 简化计算图 (常量折叠、冗余算子消除)...")
    try:
        from onnxsim import simplify

        model_onnx = onnx.load(onnx_path)
        model_sim, check = simplify(model_onnx)

        if check:
            sim_path = os.path.join(checkpoint_dir, "model_fp16_sim.onnx")
            onnx.save(model_sim, sim_path)
            sim_size_mb = os.path.getsize(sim_path) / (1024 * 1024)
            reduction = (1 - sim_size_mb / onnx_size_mb) * 100
            print(f"✅ 简化后模型: {sim_path} ({sim_size_mb:.1f} MB, 减小 {reduction:.1f}%)")
        else:
            print("⚠️  onnxsim 简化验证失败，请使用原始 ONNX 模型")

        del model_onnx, model_sim
        gc.collect()
    except Exception as e:
        print(f"⚠️  onnxsim 简化失败 ({e})，请使用原始 ONNX 模型")

    # ---- 检查 onnxruntime 可用性 ----
    print()
    print("=" * 60)
    print("📋 ONNX 导出完成！后续步骤:")
    print("=" * 60)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ onnxruntime 已安装 (版本: {ort.__version__})")
        print(f"   可用 Provider: {providers}")
        gpu_eps = [p for p in providers if "ROCM" in p or "CUDA" in p or "MIGraphX" in p]
        if gpu_eps:
            print(f"   🎯 检测到 GPU Provider: {gpu_eps}，可以使用 ONNX Runtime 加速推理！")
        else:
            print("   ⚠️  未检测到 GPU Provider，ONNX Runtime 将使用 CPU（可能更慢）")
    except ImportError:
        print("⚠️  未安装 onnxruntime")
        print("   如需使用 ONNX Runtime 推理，请安装: pip install onnxruntime-rocm (DCU) 或 onnxruntime-gpu (CUDA)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Pangu-Weather FP16 模型导出为 ONNX 并简化")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./data/checkpoints",
        help="checkpoint 目录路径 (默认: ./data/checkpoints)",
    )
    args = parser.parse_args()
    export_to_onnx(args.checkpoint_dir)
