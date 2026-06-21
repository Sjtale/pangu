import torch
import os
import sys
import glob
import numpy as np
import h5py
from tqdm import tqdm
import time
import json
import gc
from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe

# ---- 方向4.2: cuDNN 自动调优（固定输入尺寸 721×1440, batch=1）----
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class CUDAGraphWrapper:
    """CUDA Graph 推理包装类，兼容 model(invar) 调用格式。

    将模型的完整前向传播捕获为 CUDA Graph，重放时跳过所有 CPU 端
    kernel 启动开销。若捕获失败（如 DCU/ROCm 兼容性问题），调用方
    应 fallback 到原始 PyTorch 推理。
    """

    def __init__(self, model, example_input, warmup_iters=3):
        self.model = model
        self.static_input = torch.empty_like(example_input)

        # 初始化发生在主推理的 inference_mode 之外，因此必须在这里
        # 显式禁用 autograd，否则 warmup 会保留整个 Pangu 前向图。
        with torch.inference_mode():
            # Warmup: 让 caching allocator 稳定，确保后续捕获时内存地址不变
            for _ in range(warmup_iters):
                warmup_output = model(self.static_input)
            del warmup_output
            torch.cuda.synchronize()

            # Capture: 将整个 forward 录制为 CUDA Graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out_surface, self.static_out_upper = model(self.static_input)

    def __call__(self, x):
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_out_surface, self.static_out_upper


class ONNXModel:
    """ONNX Runtime 推理包装类，使其兼容 model(invar) 调用格式。

    优先使用 IOBinding 进行零拷贝 GPU 推理，自动回退到 numpy API。
    """

    def __init__(self, onnx_path, img_size=(721, 1440)):
        import onnxruntime as ort

        # 创建优化后的 Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 按优先级选择 Provider
        preferred = ['ROCMExecutionProvider', 'MIGraphXExecutionProvider',
                     'CUDAExecutionProvider', 'CPUExecutionProvider']
        available = ort.get_available_providers()
        use_providers = [p for p in preferred if p in available]

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=use_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_provider = self.session.get_providers()[0]
        self.is_gpu = any(kw in self.active_provider
                         for kw in ('ROCM', 'CUDA', 'MIGraphX'))
        self._use_iobinding = self.is_gpu  # 首次尝试 IOBinding

        # 预分配输出缓冲区（零拷贝 IOBinding 用）
        if self.is_gpu:
            self._out_surface = torch.empty(
                1, 4, img_size[0], img_size[1],
                dtype=torch.float16, device='cuda:0')
            self._out_upper_air = torch.empty(
                1, 5, 13, img_size[0], img_size[1],
                dtype=torch.float16, device='cuda:0')

        print(f"🔧 ONNX Runtime Provider: {self.active_provider}")

    def __call__(self, x):
        if self._use_iobinding:
            try:
                return self._forward_iobinding(x)
            except Exception as e:
                print(f"⚠️ IOBinding 推理失败 ({e})，切换到 numpy 模式")
                self._use_iobinding = False
        return self._forward_numpy(x)

    def _forward_iobinding(self, x):
        """零拷贝 GPU 推理: 输入输出均在 GPU 上，无 CPU 数据搬运"""
        io_binding = self.session.io_binding()
        io_binding.bind_input(
            name=self.input_name, device_type='cuda', device_id=0,
            element_type=np.float16, shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        io_binding.bind_output(
            name='output_surface', device_type='cuda', device_id=0,
            element_type=np.float16, shape=tuple(self._out_surface.shape),
            buffer_ptr=self._out_surface.data_ptr(),
        )
        io_binding.bind_output(
            name='output_upper_air', device_type='cuda', device_id=0,
            element_type=np.float16, shape=tuple(self._out_upper_air.shape),
            buffer_ptr=self._out_upper_air.data_ptr(),
        )
        self.session.run_with_iobinding(io_binding)
        return self._out_surface, self._out_upper_air

    def _forward_numpy(self, x):
        """标准 numpy 推理 (兼容性回退)"""
        input_np = x.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: input_np})
        return (
            torch.from_numpy(outputs[0]).to(x.device),
            torch.from_numpy(outputs[1]).to(x.device),
        )


def get_stats(data_dir, channels):
    """从 metadata.json 读取变量列表，提取归一化参数"""
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    all_variables = metadata["variables"]

    channel_indices = [all_variables.index(v) for v in channels]
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))   # [1, C, 1, 1]
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to('cuda:0')
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)
    surface_mask = surface_mask.half()  # FP16: static mask 也转为半精度

    # ---- 模型加载: 优先使用 ONNX Runtime，回退到 PyTorch FP16 ----
    onnx_sim_path = f"{cfg.checkpoint_dir}/model_fp16_sim.onnx"
    onnx_raw_path = f"{cfg.checkpoint_dir}/model_fp16.onnx"
    use_onnx = False
    pruned_ckpt_path = f"{cfg.checkpoint_dir}/{cfg.pruned_checkpoint}"
    enable_pruned = (
        os.environ.get("PANGU_USE_PRUNED", "1").lower() not in {"0", "false", "no"}
        and os.path.exists(pruned_ckpt_path)
    )
    # DCU 实测 ONNX ROCm EP 比 PyTorch FP16 慢，因此默认使用 PyTorch。
    enable_onnx = (
        os.environ.get("PANGU_USE_ONNX", "0").lower() not in {"0", "false", "no"}
        and not enable_pruned
    )

    for onnx_candidate in [onnx_sim_path, onnx_raw_path] if enable_onnx else []:
        if os.path.exists(onnx_candidate):
            try:
                model = ONNXModel(onnx_candidate, img_size=cfg_data.dataset.img_size)
                use_onnx = True
                print(f"⚡ 使用 ONNX Runtime 推理: {onnx_candidate}")
                break
            except Exception as e:
                print(f"⚠️ ONNX 模型加载失败 ({e})")

    if not enable_onnx:
        print("ℹ️  PANGU_USE_ONNX=0，使用 PyTorch FP16 推理")

    if not use_onnx:
        # ---- PyTorch FP16 回退 ----
        fp16_ckpt_path = f"{cfg.checkpoint_dir}/model_fp16.pth"
        fp32_ckpt_path = f"{cfg.checkpoint_dir}/model_bak.pth"
        if enable_pruned:
            print(f"✂️  加载结构化剪枝权重: {pruned_ckpt_path}")
            ckpt = torch.load(pruned_ckpt_path, map_location="cuda:0")
            model_embed_dim = cfg.pruned_embed_dim
            model_num_heads = cfg.pruned_num_heads
        elif os.path.exists(fp16_ckpt_path):
            print(f"⚡ 加载 FP16 权重: {fp16_ckpt_path}")
            ckpt = torch.load(fp16_ckpt_path, map_location="cuda:0")
            model_embed_dim = cfg.embed_dim
            model_num_heads = cfg.num_heads
        else:
            print(f"ℹ️  未找到 FP16 权重，回退加载 FP32: {fp32_ckpt_path}")
            ckpt = torch.load(fp32_ckpt_path, map_location="cuda:0")
            model_embed_dim = cfg.embed_dim
            model_num_heads = cfg.num_heads

        model = Pangu(img_size=cfg_data.dataset.img_size,
                      patch_size=cfg.patch_size,
                      embed_dim=model_embed_dim,
                      num_heads=model_num_heads,
                      window_size=cfg.window_size,
                      ).to('cuda:0')
        model.load_state_dict(ckpt["model_state_dict"])
        model.half()   # FP16: 确保整个模型在半精度下运行
        model.eval()

        # ---- 方向4.3: 释放 checkpoint 变量，清理显存碎片 ----
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

        # ---- 方向4.5: CUDA Graph 捕获（可选，DCU 上可能不支持）----
        _example = None
        try:
            _example = torch.empty(1, 72, cfg_data.dataset.img_size[0],
                                   cfg_data.dataset.img_size[1],
                                   dtype=torch.float16, device='cuda:0')
            model = CUDAGraphWrapper(model, _example)
            del _example
            torch.cuda.empty_cache()
            print("✅ CUDA Graph 捕获成功，推理将使用 Graph Replay")
        except Exception as e:
            print(f"⚠️ CUDA Graph 捕获失败 ({e})，使用标准 PyTorch 推理")
            if _example is not None:
                del _example
            gc.collect()
            torch.cuda.empty_cache()

    os.makedirs('result/output/', exist_ok=True)                          # AI4S, 输出路径不可更改
    print(f"📂 samples will be generated to './result/output/'")

    time_list = []
    first = True
    with torch.inference_mode():  # 方向4.1: 比 no_grad 更快（禁用 view tracking + version counters）
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]
            outvar = data[1]
            filename = data[4][-1][0]
            if first:
                first = False
                print(f"  invar  shape: {list(invar.shape)}   ← [batch, channels, H, W]")
                print(f"  outvar shape: {list(outvar.shape)}  ← [batch, channels, H, W]")
                print(f"  the first filename: {filename}")

            # FP16: 输入数据直接转为半精度
            # 方向4.3: non_blocking 异步数据传输，重叠 CPU/GPU 工作
            invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float16, non_blocking=True)
            invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float16, non_blocking=True)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

            #----------------------AI4S(时间度量不可更改)---------------------------
            start_time = time.perf_counter()      # AI4S(时间度量，位置不可更改)
            out_surface, out_upper_air = model(invar)
            end_time = time.perf_counter()        # AI4S(时间度量，位置不可更改)
            time_list.append(end_time-start_time) # AI4S(时间度量，位置不可更改)
            #---------------------------------------------------------------------

            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
            # FP16: 输出转回 float32 再做反归一化，避免半精度下乘法精度损失
            pred_var = torch.concat([out_surface, out_upper_air], dim=1).float().cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)


        #----------------------AI4S(时间度量不可更改)---------------------------
        # 保存到 time_list.json 文件
        with open("result/time_record.json", "w", encoding="utf-8") as f:
            json.dump(time_list, f, ensure_ascii=False, indent=4)
        #---------------------------------------------------------------------

    if torch.cuda.is_available():
        print(f"Max VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"Current VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
