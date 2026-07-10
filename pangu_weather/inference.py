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
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe
from pangu_profile_model import build_pangu_model, enable_streamed_weight_residency
from calibration_utils import (
    apply_affine_calibration,
    apply_global_mean_correction,
    load_affine_calibration,
    load_global_mean_correction,
)

# Submission defaults: the platform only executes inference.py and does not
# pass environment variables. Keep these overrideable for server A/B tests.
os.environ.setdefault("PANGU_AUTO_SCAN_CHECKPOINT", "0")
os.environ.setdefault("PANGU_DISABLE_CUDA_GRAPH", "1")
os.environ.setdefault("PANGU_LAYERWISE_INFERENCE", "1")
os.environ.setdefault("PANGU_RECOMPUTE_SKIP", "0")
os.environ.setdefault("PANGU_DIRECT_RECOVERY", "1")
os.environ.setdefault("PANGU_DIRECT_RECOVERY_WIDTH_CHUNK", "16")
os.environ.setdefault("PANGU_SCORED_ONLY_RECOVERY", "1")
os.environ.setdefault("PANGU_CHUNKED_ATTENTION", "1")
os.environ.setdefault("PANGU_ATTN_CHUNK_SIZE", "3")
os.environ.setdefault("PANGU_CHUNKED_QKV", "1")
os.environ.setdefault("PANGU_CHUNKED_PROJ", "1")
os.environ.setdefault("PANGU_CHUNKED_MLP", "1")
os.environ.setdefault("PANGU_MLP_CHUNK_SIZE", "32768")
os.environ.setdefault("PANGU_DISABLE_AFFINE_CALIBRATION", "1")
os.environ.setdefault("PANGU_GLOBAL_MEAN_CORRECTION", "0")
os.environ.setdefault("PANGU_STREAM_WEIGHTS", "0")
os.environ.setdefault("PANGU_SPLIT_RECOVERY", "0")
os.environ.setdefault("PANGU_CACHE_EARTH_BIAS", "0")
os.environ.setdefault("PANGU_INPLACE_BLOCK", "1")
os.environ.setdefault("PANGU_CLEAR_INPUT_REFS", "1")
os.environ.setdefault("PANGU_COMPACT_ATTN_MASK", "1")



# ---- 方向4.2: cuDNN 自动调优（固定输入尺寸 721×1440, batch=1）----
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class RuntimeQuantLinear(torch.nn.Module):
    """Weight-only INT8 Linear with resident quantized weights.

    This is an experimental U_vram path. It keeps INT8 weights and per-output
    scales as module buffers, then materializes one temporary FP16 weight inside
    each forward call. It reduces resident parameter memory but may trade speed
    for transient dequantization work.
    """

    def __init__(self, source_linear):
        super().__init__()
        self.in_features = int(source_linear.in_features)
        self.out_features = int(source_linear.out_features)
        self.register_buffer(
            "qweight",
            torch.empty(self.out_features, self.in_features, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "scale",
            torch.empty(self.out_features, 1, dtype=torch.float16),
            persistent=False,
        )
        if source_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(torch.empty(self.out_features))

    def forward(self, x):
        compute_dtype = x.dtype if torch.is_floating_point(x) else torch.float16
        weight = self.qweight.to(dtype=compute_dtype) * self.scale.to(dtype=compute_dtype)
        bias = self.bias.to(dtype=compute_dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)


class CUDAGraphWrapper:
    """CUDA Graph 推理包装类，兼容 model(invar) 调用格式。

    将模型的完整前向传播捕获为 CUDA Graph，重放时跳过所有 CPU 端
    kernel 启动开销。若捕获失败（如 DCU/ROCm 兼容性问题），调用方
    应 fallback 到原始 PyTorch 推理。
    """

    def __init__(self, model, example_input, warmup_iters=1):
        self.model = model
        if isinstance(example_input, (tuple, list)):
            self.static_input = tuple(torch.empty_like(t) for t in example_input)
        else:
            self.static_input = torch.empty_like(example_input)

        # 初始化发生在主推理的 inference_mode 之外，因此必须在这里
        # 显式禁用 autograd，否则 warmup 会保留整个 Pangu 前向图。
        with torch.inference_mode():
            # Warmup: 让 caching allocator 稳定，确保后续捕获时内存地址不变
            warmup_output = None
            for _ in range(warmup_iters):
                warmup_output = model(self.static_input)
            if warmup_output is not None:
                del warmup_output
            torch.cuda.synchronize()

            # Capture: 将整个 forward 录制为 CUDA Graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out_surface, self.static_out_upper = model(self.static_input)

    def __call__(self, x):
        if isinstance(x, (tuple, list)):
            for src, dst in zip(x, self.static_input):
                dst.copy_(src)
        else:
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


def get_stats(data_dir, stats_dir, channels):
    """从 metadata.json 读取变量列表，提取归一化参数"""
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    all_variables = metadata["variables"]

    channel_indices = [all_variables.index(v) for v in channels]
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))   # [1, C, 1, 1]
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


def _load_output_calibration(checkpoint_dir, means, stds):
    num_channels = int(stds.shape[1])
    affine_path = os.path.join(checkpoint_dir, "calibration_affine.npz")
    slope_path = os.path.join(checkpoint_dir, "calibration_coeffs.npy")

    if os.path.exists(affine_path) and not _is_enabled("PANGU_DISABLE_AFFINE_CALIBRATION"):
        try:
            affine = load_affine_calibration(affine_path, num_channels)
            print(f"🎯  Loaded affine calibration from {affine_path}.")
            return stds, affine
        except Exception as e:
            print(f"⚠️  Failed to load affine calibration: {e}")

    if os.path.exists(slope_path):
        try:
            coeffs = np.load(slope_path).reshape(1, -1, 1, 1)
            if coeffs.shape[1] != num_channels:
                raise ValueError(
                    f"expected {num_channels} channels, got {coeffs.shape[1]}"
                )
            stds = stds * coeffs
            print(f"🎯  Loaded slope calibration from {slope_path} and adjusted stds.")
        except Exception as e:
            print(f"⚠️  Failed to load slope calibration: {e}")
    return stds, None


def _load_global_mean_correction(checkpoint_dir, num_channels):
    if not _is_enabled("PANGU_GLOBAL_MEAN_CORRECTION"):
        return None
    path = os.path.join(checkpoint_dir, "physics_mean_targets.npz")
    if not os.path.exists(path):
        print(f"⚠️  PANGU_GLOBAL_MEAN_CORRECTION=1 but {path} is missing.")
        return None
    try:
        correction = load_global_mean_correction(path, num_channels)
        active = int(np.count_nonzero(correction.channel_mask))
        print(f"🌐  Loaded global mean correction from {path} for {active} channels.")
        return correction
    except Exception as e:
        print(f"⚠️  Failed to load global mean correction: {e}")
        return None


def _cfg_list(value):
    return [int(v) for v in value]


def _is_enabled(name, default=False):
    default_value = "1" if default else "0"
    return os.environ.get(name, default_value).lower() not in {"0", "false", "no"}


def _env_int(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _stream_weights_mode():
    raw_value = os.environ.get("PANGU_STREAM_WEIGHTS", "0").strip().lower()
    if raw_value in {"", "0", "false", "no", "off"}:
        return None
    if raw_value not in {"stage", "block"}:
        raise ValueError(
            "PANGU_STREAM_WEIGHTS must be one of: 0, stage, block; "
            f"got {raw_value!r}"
        )
    return raw_value


def _profile_cuda_memory(tag):
    if not _is_enabled("PANGU_PROFILE_MEMORY"):
        return
    if not torch.cuda.is_available():
        print(f"[MEM] {tag}: CUDA unavailable")
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"[MEM] {tag}: allocated={allocated:.1f} MB, "
        f"reserved={reserved:.1f} MB, peak={peak:.1f} MB"
    )


def _reset_cuda_peak(tag):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print(f"[MEM] {tag}: reset CUDA peak memory stats")


def _profile_from_config(cfg, profile_name):
    profiles = getattr(cfg, "student_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown model profile: {profile_name}")
    profile = profiles[profile_name]
    res = {
        "name": profile_name,
        "patch_size": _cfg_list(profile.patch_size),
        "embed_dim": int(profile.embed_dim),
        "num_heads": _cfg_list(profile.num_heads),
        "window_size": _cfg_list(cfg.window_size),
    }
    if hasattr(profile, "depth_blocks"):
        res["depth_blocks"] = _cfg_list(profile.depth_blocks)
    return res


def _default_profile(cfg):
    return {
        "name": "full_192",
        "patch_size": _cfg_list(cfg.patch_size),
        "embed_dim": int(cfg.embed_dim),
        "num_heads": _cfg_list(cfg.num_heads),
        "window_size": _cfg_list(cfg.window_size),
    }


def _profile_from_metadata(cfg, metadata):
    if not isinstance(metadata, dict):
        return None
    profile_name = metadata.get("name")
    if profile_name and profile_name in getattr(cfg, "student_profiles", {}):
        profile = _profile_from_config(cfg, profile_name)
    else:
        profile = _default_profile(cfg)
        if profile_name:
            profile["name"] = profile_name
    if "patch_size" in metadata:
        profile["patch_size"] = _cfg_list(metadata["patch_size"])
    if "embed_dim" in metadata:
        profile["embed_dim"] = int(metadata["embed_dim"])
    if "num_heads" in metadata:
        profile["num_heads"] = _cfg_list(metadata["num_heads"])
    if "window_size" in metadata:
        profile["window_size"] = _cfg_list(metadata["window_size"])
    if "depth_blocks" in metadata:
        profile["depth_blocks"] = _cfg_list(metadata["depth_blocks"])
    if metadata.get("share_deep_blocks"):
        profile["share_deep_blocks"] = str(metadata["share_deep_blocks"])
    return profile


def _infer_profile_from_state(cfg, ckpt, state_dict):
    profile = _profile_from_metadata(cfg, ckpt.get("model_profile"))
    if profile is not None:
        return profile

    quant_meta = ckpt.get("quantization", {})
    quant_profile = quant_meta.get("target_profile") if isinstance(quant_meta, dict) else None
    if quant_profile in getattr(cfg, "student_profiles", {}):
        return _profile_from_config(cfg, quant_profile)

    distill_meta = ckpt.get("distillation", {})
    distill_profile = distill_meta.get("student_profile") if isinstance(distill_meta, dict) else None
    if distill_profile in getattr(cfg, "student_profiles", {}):
        return _profile_from_config(cfg, distill_profile)

    profile = _default_profile(cfg)
    embed_weight = next(
        (
            tensor
            for key, tensor in state_dict.items()
            if key.endswith("patchembed2d.embedder.proj.weight")
            or key.endswith("patchembed2d.proj.weight")
        ),
        None,
    )
    if isinstance(embed_weight, torch.Tensor):
        profile["embed_dim"] = int(embed_weight.shape[0])
        profile["patch_size"] = [int(v) for v in embed_weight.shape[-3:]]
    if profile["embed_dim"] == int(cfg.pruned_embed_dim):
        profile["name"] = "student_160"
        profile["num_heads"] = _cfg_list(cfg.pruned_num_heads)
    elif profile["patch_size"][-2:] == [8, 8]:
        profile["name"] = getattr(cfg, "pgw_lite_profile", "pgw_lite_patch8")
    return profile


def _state_dict_keys(state_dict):
    return list(state_dict.keys())


def _layer_index_from_key(key):
    for idx, layer_name in enumerate(("layer1.", "layer2.", "layer3.", "layer4.")):
        if layer_name in key:
            return idx
    return None


def _infer_gqa_group_size(state_dict, model_profile, default):
    votes = {}
    num_heads_by_layer = [int(v) for v in model_profile["num_heads"]]
    for q_key in _state_dict_keys(state_dict):
        if not q_key.endswith(".q_proj.weight"):
            continue
        layer_idx = _layer_index_from_key(q_key)
        if layer_idx is None:
            continue
        kv_key = q_key[:-len(".q_proj.weight")] + ".kv_proj.weight"
        if kv_key not in state_dict:
            continue
        num_heads = num_heads_by_layer[layer_idx]
        q_weight = state_dict[q_key]
        kv_weight = state_dict[kv_key]
        q_dim = int(q_weight.shape[0])
        if q_dim % num_heads != 0:
            continue
        head_dim = q_dim // num_heads
        if head_dim == 0:
            continue
        kv_heads = int(kv_weight.shape[0]) // 2 // head_dim
        if kv_heads <= 0:
            continue
        group_size = max(1, num_heads // kv_heads)
        votes[group_size] = votes.get(group_size, 0) + 1
    if not votes:
        return default
    return max(votes.items(), key=lambda item: (item[1], -item[0]))[0]


def _detect_architecture_from_state(state_dict, model_profile):
    keys = _state_dict_keys(state_dict)
    use_gqa = any(".q_proj." in key or ".kv_proj." in key for key in keys)
    use_swiglu = any(".mlp.w1." in key or ".mlp.w2." in key for key in keys)
    has_norm_weight = any(".norm1.weight" in key or ".norm2.weight" in key for key in keys)
    has_norm_bias = any(".norm1.bias" in key or ".norm2.bias" in key for key in keys)
    use_rmsnorm = has_norm_weight and not has_norm_bias

    gqa_group_size = _env_int("PANGU_GQA_GROUP_SIZE", 2)
    if use_gqa:
        gqa_group_size = _infer_gqa_group_size(state_dict, model_profile, gqa_group_size)

    return {
        "use_gqa": use_gqa,
        "use_swiglu": use_swiglu,
        "use_rmsnorm": use_rmsnorm,
        "gqa_group_size": gqa_group_size,
    }


def _checkpoint_score(path, cfg, ckpt, state_dict, preferred_profile):
    basename = os.path.basename(path)
    score = 0
    quant_meta = ckpt.get("quantization", {}) if isinstance(ckpt, dict) else {}
    model_meta = ckpt.get("model_profile", {}) if isinstance(ckpt, dict) else {}
    distill_meta = ckpt.get("distillation", {}) if isinstance(ckpt, dict) else {}

    quant_profile = quant_meta.get("target_profile") if isinstance(quant_meta, dict) else None
    model_profile = model_meta.get("name") if isinstance(model_meta, dict) else None
    distill_profile = distill_meta.get("student_profile") if isinstance(distill_meta, dict) else None
    profile_names = {p for p in (quant_profile, model_profile, distill_profile) if p}

    if isinstance(quant_meta, dict) and quant_meta:
        score += 100
    if preferred_profile in profile_names:
        score += 60
    if any(str(profile).startswith("pgw_lite") for profile in profile_names):
        score += 30
    if "quant" in basename:
        score += 20
    if preferred_profile and preferred_profile in basename:
        score += 20
    if basename in {"model_bak.pth", "model_fp16.pth"}:
        score -= 40

    keys = _state_dict_keys(state_dict)
    if any(".q_proj." in key or ".kv_proj." in key for key in keys):
        score += 20
    if any(".mlp.w1." in key or ".mlp.w2." in key for key in keys):
        score += 20
    if any(key.endswith("_scale") for key in keys):
        score += 20

    try:
        profile = _infer_profile_from_state(cfg, ckpt, state_dict)
        if profile["name"] == preferred_profile:
            score += 30
        if profile["patch_size"][-2:] == [8, 8] and profile["embed_dim"] == 96:
            score += 20
    except Exception:
        pass

    return score


def _scan_checkpoint_path(cfg, preferred_profile):
    pattern = os.path.join(cfg.checkpoint_dir, "*.pth")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None

    best = None
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            score = _checkpoint_score(path, cfg, ckpt, state_dict, preferred_profile)
        except Exception as exc:
            print(f"⚠️ 跳过无法读取的 checkpoint: {path} ({exc})")
            continue
        finally:
            try:
                del ckpt
                del state_dict
            except Exception:
                pass

        item = (score, os.path.getmtime(path), path)
        if best is None or item > best:
            best = item

    if best is None:
        return None
    score, _, path = best
    print(f"🔎 自动扫描 checkpoint: {path} (score={score})")
    return path


def _load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    return ckpt, state_dict


def _dequantize_tensor_for_model(key, value, state_dict, target_tensor):
    target_device = target_tensor.device
    target_dtype = target_tensor.dtype
    if key.endswith(".weight") and value.dtype == torch.int8:
        scale_key = key + "_scale"
        if scale_key in state_dict:
            scale = state_dict[scale_key].to(device=target_device, dtype=torch.float32)
            if scale.ndim == 1 and value.ndim == 2:
                scale = scale.view(-1, 1)
            return (
                value.to(device=target_device, dtype=torch.float32) * scale
            ).to(target_dtype)
    return value.to(device=target_device, dtype=target_dtype)


def _quantized_linear_module_names(state_dict):
    names = set()
    for key, value in state_dict.items():
        if (
            key.endswith(".weight")
            and isinstance(value, torch.Tensor)
            and value.dtype == torch.int8
            and key + "_scale" in state_dict
        ):
            names.add(key[: -len(".weight")])
    return names


def _set_module_by_name(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _replace_quantized_linear_modules(model, state_dict):
    quantized_names = _quantized_linear_module_names(state_dict)
    replaced = 0
    for name, module in list(model.named_modules()):
        if name in quantized_names and isinstance(module, torch.nn.Linear):
            _set_module_by_name(model, name, RuntimeQuantLinear(module))
            replaced += 1
    return replaced, len(quantized_names)


def _copy_runtime_quant_linear(module, key, value, state_dict):
    scale_key = key + "_scale"
    if scale_key not in state_dict:
        raise KeyError(f"Missing quantization scale for {key}")
    if tuple(value.shape) != tuple(module.qweight.shape):
        raise RuntimeError(
            f"Shape mismatch for {key}: checkpoint "
            f"{tuple(value.shape)} vs runtime quant {tuple(module.qweight.shape)}"
        )
    scale = state_dict[scale_key]
    if scale.ndim == 1:
        scale = scale.view(-1, 1)
    if tuple(scale.shape) != tuple(module.scale.shape):
        raise RuntimeError(
            f"Shape mismatch for {scale_key}: checkpoint "
            f"{tuple(scale.shape)} vs runtime quant {tuple(module.scale.shape)}"
        )
    module.qweight.copy_(value.to(device=module.qweight.device, dtype=torch.int8))
    module.scale.copy_(scale.to(device=module.scale.device, dtype=module.scale.dtype))


def _load_runtime_quant_state_dict(model, state_dict):
    modules = dict(model.named_modules())
    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    loaded_count = 0
    consumed_keys = set()

    with torch.no_grad():
        for key, value in state_dict.items():
            if key.endswith("_scale"):
                continue
            module_name = key[: -len(".weight")] if key.endswith(".weight") else None
            module = modules.get(module_name) if module_name else None
            if isinstance(module, RuntimeQuantLinear):
                _copy_runtime_quant_linear(module, key, value, state_dict)
                consumed_keys.add(key)
                consumed_keys.add(key + "_scale")
                loaded_count += 1
                continue

            if key not in model_state:
                unexpected_keys.append(key)
                continue
            target_tensor = model_state[key]
            loaded_tensor = _dequantize_tensor_for_model(
                key, value, state_dict, target_tensor
            )
            if tuple(loaded_tensor.shape) != tuple(target_tensor.shape):
                raise RuntimeError(
                    f"Shape mismatch for {key}: checkpoint "
                    f"{tuple(loaded_tensor.shape)} vs model {tuple(target_tensor.shape)}"
                )
            target_tensor.copy_(loaded_tensor)
            consumed_keys.add(key)
            loaded_count += 1
            del loaded_tensor

    source_keys = {key for key in state_dict if not key.endswith("_scale")}
    for key in model_state:
        if key not in source_keys:
            missing_keys.append(key)

    skipped_scales = {
        key for key in state_dict if key.endswith("_scale") and key not in consumed_keys
    }
    if skipped_scales:
        unexpected_keys.extend(sorted(skipped_scales))

    return missing_keys, unexpected_keys, loaded_count


def _load_dequantized_state_dict_incremental(model, state_dict):
    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    loaded_count = 0

    with torch.no_grad():
        for key, value in state_dict.items():
            if key.endswith("_scale"):
                continue
            if key not in model_state:
                unexpected_keys.append(key)
                continue
            target_tensor = model_state[key]
            loaded_tensor = _dequantize_tensor_for_model(
                key, value, state_dict, target_tensor
            )
            if tuple(loaded_tensor.shape) != tuple(target_tensor.shape):
                raise RuntimeError(
                    f"Shape mismatch for {key}: checkpoint "
                    f"{tuple(loaded_tensor.shape)} vs model {tuple(target_tensor.shape)}"
                )
            target_tensor.copy_(loaded_tensor)
            loaded_count += 1
            del loaded_tensor

    source_keys = {key for key in state_dict if not key.endswith("_scale")}
    for key in model_state:
        if key not in source_keys:
            missing_keys.append(key)

    return missing_keys, unexpected_keys, loaded_count


def _dequantize_state_dict(state_dict, target_dtype):
    dequantized_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".weight") and value.dtype == torch.int8:
            scale_key = key + "_scale"
            if scale_key in state_dict:
                scale = state_dict[scale_key].to(device="cuda:0", dtype=torch.float32)
                if scale.ndim == 1 and value.ndim == 2:
                    scale = scale.view(-1, 1)
                dequantized_state_dict[key] = (
                    value.to(device="cuda:0", dtype=torch.float32) * scale
                ).to(target_dtype)
            else:
                dequantized_state_dict[key] = value.to(
                    device="cuda:0",
                    dtype=target_dtype if torch.is_floating_point(value) else value.dtype,
                )
        elif key.endswith("_scale"):
            continue
        else:
            dequantized_state_dict[key] = value.to(
                device="cuda:0",
                dtype=target_dtype if torch.is_floating_point(value) else value.dtype,
            )
    return dequantized_state_dict


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.stats_dir, cfg_data.dataset.channels)

    stds, affine_calibration = _load_output_calibration(cfg.checkpoint_dir, means, stds)
    global_mean_correction = _load_global_mean_correction(
        cfg.checkpoint_dir, int(means.shape[1])
    )

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to('cuda:0')
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)
    surface_mask = surface_mask.half()  # FP16: static mask 也转为半精度
    _profile_cuda_memory("after static mask load")

    # ---- 模型加载: 优先使用 ONNX Runtime，回退到 PyTorch FP16 ----
    onnx_sim_path = f"{cfg.checkpoint_dir}/model_fp16_sim.onnx"
    onnx_raw_path = f"{cfg.checkpoint_dir}/model_fp16.onnx"
    use_onnx = False
    pruned_ckpt_path = f"{cfg.checkpoint_dir}/{cfg.pruned_checkpoint}"
    distilled_ckpt_path = f"{cfg.checkpoint_dir}/{cfg.distilled_checkpoint}"
    
    # Resolve profile-specific pgw_lite filenames dynamically
    profile_name = os.environ.get("PANGU_STUDENT_PROFILE", getattr(cfg, "default_student_profile", "student_160"))
    if "pgw_lite" in profile_name:
        pgw_lite_quantized_path = f"{cfg.checkpoint_dir}/model_{profile_name}_quantized.pth"
        pgw_lite_ckpt_path = f"{cfg.checkpoint_dir}/model_{profile_name}_fp16.pth"
        # Backward compatibility fallback for pgw_lite_patch8 configured names
        if profile_name == "pgw_lite_patch8":
            legacy_quant = f"{cfg.checkpoint_dir}/{cfg.pgw_lite_quantized_checkpoint}"
            legacy_fp16 = f"{cfg.checkpoint_dir}/{cfg.pgw_lite_distilled_checkpoint}"
            if not os.path.exists(pgw_lite_quantized_path) and os.path.exists(legacy_quant):
                pgw_lite_quantized_path = legacy_quant
            if not os.path.exists(pgw_lite_ckpt_path) and os.path.exists(legacy_fp16):
                pgw_lite_ckpt_path = legacy_fp16
    else:
        pgw_lite_quantized_path = f"{cfg.checkpoint_dir}/{cfg.pgw_lite_quantized_checkpoint}"
        pgw_lite_ckpt_path = f"{cfg.checkpoint_dir}/{cfg.pgw_lite_distilled_checkpoint}"

    distilled_quantized_path = f"{cfg.checkpoint_dir}/{getattr(cfg, 'quantized_checkpoint', 'model_distilled_quantized.pth')}"
    explicit_checkpoint = os.environ.get("PANGU_CHECKPOINT")
    explicit_ckpt_path = None
    if explicit_checkpoint:
        explicit_ckpt_path = (
            explicit_checkpoint
            if os.path.isabs(explicit_checkpoint)
            else f"{cfg.checkpoint_dir}/{explicit_checkpoint}"
        )
        if not os.path.exists(explicit_ckpt_path):
            raise FileNotFoundError(f"PANGU_CHECKPOINT not found: {explicit_ckpt_path}")
    auto_ckpt_path = None
    if explicit_ckpt_path is None and _is_enabled("PANGU_AUTO_SCAN_CHECKPOINT", default=True):
        auto_ckpt_path = _scan_checkpoint_path(cfg, profile_name)
    enable_pgw_lite = (
        _is_enabled("PANGU_USE_PGW_LITE")
        and (os.path.exists(pgw_lite_quantized_path) or os.path.exists(pgw_lite_ckpt_path))
    )
    enable_distilled = (
        _is_enabled("PANGU_USE_DISTILLED")
        and os.path.exists(distilled_ckpt_path)
        and not enable_pgw_lite
    )
    enable_pruned = (
        _is_enabled("PANGU_USE_PRUNED")
        and os.path.exists(pruned_ckpt_path)
        and not enable_distilled
        and not enable_pgw_lite
    )
    # DCU 实测 ONNX ROCm EP 比 PyTorch FP16 慢，因此默认使用 PyTorch。
    enable_onnx = (
        _is_enabled("PANGU_USE_ONNX")
        and explicit_ckpt_path is None
        and auto_ckpt_path is None
        and not enable_pruned
        and not enable_distilled
        and not enable_pgw_lite
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
        try:
            fp16_checkpoint = os.environ.get("PANGU_FP16_CHECKPOINT", "model_fp16.pth")
            fp16_ckpt_path = (
                fp16_checkpoint
                if os.path.isabs(fp16_checkpoint)
                else f"{cfg.checkpoint_dir}/{fp16_checkpoint}"
            )
            local_fp32_path = f"{cfg.checkpoint_dir}/model_bak.pth"
            backup_fp32_path = f"{cfg.official_checkpoint_dir}/model_bak.pth"
            fp32_ckpt_path = (
                local_fp32_path if os.path.exists(local_fp32_path) else backup_fp32_path
            )
            if explicit_ckpt_path is not None:
                selected_path = explicit_ckpt_path
                print(f"📌 加载显式指定权重: {selected_path}")
                ckpt, state_dict = _load_checkpoint(selected_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _infer_profile_from_state(cfg, ckpt, state_dict)
            elif enable_pgw_lite:
                selected_path = pgw_lite_quantized_path if os.path.exists(pgw_lite_quantized_path) else pgw_lite_ckpt_path
                print(f"加载 PGW-Lite 学生权重: {selected_path}")
                ckpt, state_dict = _load_checkpoint(selected_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _infer_profile_from_state(cfg, ckpt, state_dict)
            elif auto_ckpt_path is not None:
                selected_path = auto_ckpt_path
                print(f"🔎 加载自动扫描权重: {selected_path}")
                ckpt, state_dict = _load_checkpoint(selected_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _infer_profile_from_state(cfg, ckpt, state_dict)
            elif enable_distilled:
                print(f"加载知识蒸馏学生权重: {distilled_ckpt_path}")
                ckpt, state_dict = _load_checkpoint(distilled_ckpt_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _infer_profile_from_state(cfg, ckpt, state_dict)
            elif enable_pruned:
                print(f"✂️  加载结构化剪枝权重: {pruned_ckpt_path}")
                ckpt, state_dict = _load_checkpoint(pruned_ckpt_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _profile_from_config(cfg, "student_160")
            elif os.path.exists(fp16_ckpt_path):
                print(f"⚡ 加载 FP16 权重: {fp16_ckpt_path}")
                ckpt, state_dict = _load_checkpoint(fp16_ckpt_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _infer_profile_from_state(cfg, ckpt, state_dict)
            elif os.path.exists(distilled_quantized_path):
                print(f"⚡ 加载量化学生权重: {distilled_quantized_path}")
                ckpt, state_dict = _load_checkpoint(distilled_quantized_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _infer_profile_from_state(cfg, ckpt, state_dict)
            else:
                print(f"ℹ️  未找到 FP16 权重，回退加载 FP32: {fp32_ckpt_path}")
                ckpt, state_dict = _load_checkpoint(fp32_ckpt_path)
                _profile_cuda_memory("after checkpoint load")
                model_profile = _infer_profile_from_state(cfg, ckpt, state_dict)
            print(
                f"ℹ️  模型结构 profile={model_profile['name']} "
                f"patch={model_profile['patch_size']} embed={model_profile['embed_dim']}"
            )
            share_deep_blocks = model_profile.get("share_deep_blocks")
            if share_deep_blocks:
                print(f"ℹ️  深层共享 share_deep_blocks={share_deep_blocks}")

            use_fp16 = os.environ.get("PANGU_USE_FP16", "1") == "1"
            target_dtype = torch.float16 if use_fp16 else torch.float32

            layerwise_inference = _is_enabled("PANGU_LAYERWISE_INFERENCE")
            layerwise_empty_cache = _is_enabled("PANGU_LAYERWISE_EMPTY_CACHE")
            recompute_skip = _is_enabled("PANGU_RECOMPUTE_SKIP")
            chunked_attention = _is_enabled("PANGU_CHUNKED_ATTENTION")
            attention_chunk_size = _env_int("PANGU_ATTN_CHUNK_SIZE", 3)
            stream_weights = _stream_weights_mode()
            stream_weights_pin_memory = _is_enabled(
                "PANGU_STREAM_WEIGHTS_PIN_MEMORY", default=True
            )
            stream_weights_empty_cache = _is_enabled(
                "PANGU_STREAM_WEIGHTS_EMPTY_CACHE", default=True
            )
            if stream_weights:
                layerwise_inference = True

            # Detect architecture configurations from loaded checkpoint
            arch_flags = _detect_architecture_from_state(state_dict, model_profile)
            use_gqa = arch_flags["use_gqa"]
            use_swiglu = arch_flags["use_swiglu"]
            use_rmsnorm = arch_flags["use_rmsnorm"]
            gqa_group_size = arch_flags["gqa_group_size"]
            print(
                "ℹ️  架构开关 "
                f"SwiGLU={int(use_swiglu)} RMSNorm={int(use_rmsnorm)} "
                f"GQA={int(use_gqa)} GQA_GROUP_SIZE={gqa_group_size}"
            )


            model = build_pangu_model(
                img_size=cfg_data.dataset.img_size,
                patch_size=model_profile["patch_size"],
                embed_dim=model_profile["embed_dim"],
                num_heads=model_profile["num_heads"],
                window_size=model_profile["window_size"],
                depth_blocks=model_profile.get("depth_blocks", None),
                recompute_skip=recompute_skip,
                layerwise_inference=layerwise_inference,
                layerwise_empty_cache=layerwise_empty_cache,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                use_gqa=use_gqa,
                kv_group_size=gqa_group_size,
                share_deep_blocks=share_deep_blocks,
                chunked_attention=chunked_attention,
                attention_chunk_size=attention_chunk_size,
            )
            runtime_quant_linear = (
                _is_enabled("PANGU_RUNTIME_QUANT_LINEAR")
                and not use_gqa
                and use_fp16
            )
            runtime_quant_replaced = 0
            if runtime_quant_linear:
                runtime_quant_replaced, runtime_quant_available = (
                    _replace_quantized_linear_modules(model, state_dict)
                )
                print(
                    "ℹ️  PANGU_RUNTIME_QUANT_LINEAR=1，替换 Linear "
                    f"{runtime_quant_replaced}/{runtime_quant_available}"
                )
                if runtime_quant_replaced == 0:
                    print("⚠️  未找到可替换的量化 Linear，回退常规增量加载")
                    runtime_quant_linear = False
            if use_fp16:
                model.half()   # FP16: ensure model storage is half before moving to GPU
            model = model.to('cuda:0')
            _profile_cuda_memory("after model build/to cuda")
            if layerwise_inference:
                print("🧩  PANGU_LAYERWISE_INFERENCE=1，启用逐层推理 forward")
                if layerwise_empty_cache:
                    print("🧹  PANGU_LAYERWISE_EMPTY_CACHE=1，逐层推理时清理 CUDA cache")
            if stream_weights:
                print(
                    "🪝  PANGU_STREAM_WEIGHTS="
                    f"{stream_weights}，推理时流式搬运 backbone 权重"
                )
            if recompute_skip:
                print("♻️  PANGU_RECOMPUTE_SKIP=1，推理时重算 skip activation")
            if _is_enabled("PANGU_CHUNKED_RECOVERY"):
                print(
                    "🧱  PANGU_CHUNKED_RECOVERY=1，"
                    f"chunk_size={_env_int('PANGU_RECOVERY_CHUNK_SIZE', 1)}"
                )
            if _is_enabled("PANGU_DIRECT_RECOVERY"):
                print(
                    "🧩  PANGU_DIRECT_RECOVERY=1，使用 direct patch unembedding，"
                    f"width_chunk={_env_int('PANGU_DIRECT_RECOVERY_WIDTH_CHUNK', 16)}"
                )
            if chunked_attention:
                patched_attention = getattr(model, "_pangu_chunked_attention_count", 0)
                print(
                    "🧠  PANGU_CHUNKED_ATTENTION=1，"
                    f"chunk_size={attention_chunk_size}，patched={patched_attention}"
                )
            incremental_state_load = _is_enabled(
                "PANGU_INCREMENTAL_STATE_LOAD", default=True
            )
            if runtime_quant_linear:
                print("ℹ️  使用 runtime QuantLinear 常驻 INT8 权重加载")
                missing_keys, unexpected_keys, loaded_count = (
                    _load_runtime_quant_state_dict(model, state_dict)
                )
                print(
                    f"ℹ️  QuantLinear 加载完成: loaded={loaded_count}, "
                    f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
                )
                if unexpected_keys:
                    print(f"⚠️  忽略未使用权重数: {len(unexpected_keys)}")
                _profile_cuda_memory("after runtime quant state load")
            elif incremental_state_load and not use_gqa:
                print("ℹ️  PANGU_INCREMENTAL_STATE_LOAD=1，逐 tensor 反量化并加载")
                missing_keys, unexpected_keys, loaded_count = (
                    _load_dequantized_state_dict_incremental(model, state_dict)
                )
                print(
                    f"ℹ️  增量加载完成: loaded={loaded_count}, "
                    f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
                )
                if unexpected_keys:
                    print(f"⚠️  忽略未使用权重数: {len(unexpected_keys)}")
                _profile_cuda_memory("after incremental state load")
            else:
                if use_gqa and incremental_state_load:
                    print("ℹ️  GQA checkpoint 需要 qkv 适配，回退完整 state_dict 加载")
                # ---- 反量化重建逻辑（仅在模型加载时运行，不影响单样本推理计时）----
                ckpt["model_state_dict"] = _dequantize_state_dict(state_dict, target_dtype)
                _profile_cuda_memory("after state dict dequantize/cast")
                from pangu_profile_model import adapt_qkv_for_gqa
                ckpt["model_state_dict"] = adapt_qkv_for_gqa(ckpt["model_state_dict"], model)
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
            _profile_cuda_memory("after load_state_dict/model eval")
            if stream_weights:
                offloaded_count, offloaded_bytes = enable_streamed_weight_residency(
                    model,
                    mode=stream_weights,
                    pin_memory=stream_weights_pin_memory,
                    empty_cache=stream_weights_empty_cache,
                )
                print(
                    "🪝  流式权重驻留已准备: "
                    f"mode={stream_weights}, modules={offloaded_count}, "
                    f"offloaded={offloaded_bytes / 1024**2:.1f} MB, "
                    f"pin_memory={int(stream_weights_pin_memory)}, "
                    f"empty_cache={int(stream_weights_empty_cache)}"
                )
                _profile_cuda_memory("after streamed weight offload")

            # ---- 方向4.3: 释放 checkpoint 变量，清理显存碎片 ----
            del state_dict
            del ckpt
            gc.collect()
            torch.cuda.empty_cache()
            _profile_cuda_memory("after checkpoint cleanup")
        except Exception as e:
            print(f"❌ 加载模型或权重出错: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists("data"):
                print("data/ 目录内容:", os.listdir("data"))
                if os.path.exists("data/checkpoints"):
                    print("data/checkpoints/ 目录内容:", os.listdir("data/checkpoints"))
            print("当前执行根目录内容:", os.listdir("."))
            raise e

        target_dtype = torch.float16 if use_fp16 else torch.float32
        # ---- 方向4.5: CUDA Graph 捕获（可选，DCU 上可能不支持）----
        _example = None
        if _is_enabled("PANGU_DISABLE_CUDA_GRAPH"):
            print("ℹ️  PANGU_DISABLE_CUDA_GRAPH=1，跳过 CUDA Graph 捕获，使用标准 PyTorch 推理")
        elif stream_weights:
            print("ℹ️  PANGU_STREAM_WEIGHTS 启用时跳过 CUDA Graph 捕获")
        elif _is_enabled("PANGU_LAYERWISE_INFERENCE") and not _is_enabled("PANGU_LAYERWISE_CUDA_GRAPH"):
            print("ℹ️  Layerwise 推理默认跳过整模型 CUDA Graph；如需捕获请设置 PANGU_LAYERWISE_CUDA_GRAPH=1")
        else:
            try:
                graph_warmup_iters = _env_int("PANGU_CUDA_GRAPH_WARMUP_ITERS", 1)
                if graph_warmup_iters != 1:
                    print(f"ℹ️  PANGU_CUDA_GRAPH_WARMUP_ITERS={graph_warmup_iters}")
                _example_surface = torch.empty(1, 7, cfg_data.dataset.img_size[0],
                                               cfg_data.dataset.img_size[1],
                                               dtype=target_dtype, device='cuda:0')
                _example_upper = torch.empty(1, 5, 13, cfg_data.dataset.img_size[0],
                                             cfg_data.dataset.img_size[1],
                                             dtype=target_dtype, device='cuda:0')
                _example = (_example_surface, _example_upper)
                model = CUDAGraphWrapper(model, _example, warmup_iters=graph_warmup_iters)
                del _example_surface, _example_upper, _example
                torch.cuda.empty_cache()
                print("✅ CUDA Graph 捕获成功，推理将使用 Graph Replay")
                _profile_cuda_memory("after CUDA Graph capture")
            except Exception as e:
                print(f"⚠️ CUDA Graph 捕获失败 ({e})，使用标准 PyTorch 推理")
                if _example is not None:
                    del _example
                gc.collect()
                torch.cuda.empty_cache()
                _profile_cuda_memory("after CUDA Graph fallback cleanup")

    os.makedirs('result/output/', exist_ok=True)                          # AI4S, 输出路径不可更改
    print(f"📂 samples will be generated to './result/output/'")
    _profile_cuda_memory("before inference loop")
    if _is_enabled("PANGU_RESET_PEAK_AFTER_LOAD"):
        _reset_cuda_peak("before inference loop")

    time_list = []
    first = True
    max_inference_batches = _env_int("PANGU_MAX_INFERENCE_BATCHES", 0)
    with torch.inference_mode():  # 方向4.1: 比 no_grad 更快（禁用 view tracking + version counters）
        for batch_index, data in enumerate(tqdm(test_dataloader, desc="Inferring testset", unit="batch"), start=1):
            if max_inference_batches > 0 and batch_index > max_inference_batches:
                print(f"ℹ️  PANGU_MAX_INFERENCE_BATCHES={max_inference_batches}; stopping early.")
                break
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
            invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=target_dtype, non_blocking=True)
            invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=target_dtype, non_blocking=True)
            # Avoid GPU concatenation of invar (150 MB saving)
            invar_surface_with_mask = torch.concat([invar_surface, surface_mask], dim=1)
            invar_upper_air_reshaped = invar_upper_air.reshape(
                invar_upper_air.shape[0], 5, 13, invar_upper_air.shape[2], invar_upper_air.shape[3]
            )
            upper_air_shape = tuple(invar_upper_air.shape)
            if _is_enabled("PANGU_CLEAR_INPUT_REFS", default=True):
                invar = [invar_surface_with_mask, invar_upper_air_reshaped]
                del invar_surface, invar_upper_air, invar_surface_with_mask, invar_upper_air_reshaped
            else:
                invar = (invar_surface_with_mask, invar_upper_air_reshaped)

            #----------------------AI4S(时间度量不可更改)---------------------------
            start_time = time.perf_counter()      # AI4S(时间度量，位置不可更改)
            out_surface, out_upper_air = model(invar)
            torch.cuda.synchronize()              # AI4S(时间度量，位置不可更改，新增)
            end_time = time.perf_counter()        # AI4S(时间度量，位置不可更改)
            time_list.append(end_time-start_time) # AI4S(时间度量，位置不可更改)
            #---------------------------------------------------------------------
            if _is_enabled("PANGU_PROFILE_MEMORY") and len(time_list) == 1:
                _profile_cuda_memory("after first timed forward")

            out_upper_air = out_upper_air.reshape(upper_air_shape)
            # FP16: 输出转回 float32 再做反归一化，避免半精度下乘法精度损失
            if _is_enabled("PANGU_CPU_OUTPUT_POSTPROCESS", default=True):
                pred_tensor = torch.concat(
                    [out_surface.detach().cpu(), out_upper_air.detach().cpu()],
                    dim=1,
                ).float()
                pred_var = pred_tensor.numpy()
            else:
                pred_var = torch.concat([out_surface, out_upper_air], dim=1).float().cpu().numpy()
            pred_var = pred_var * stds + means
            pred_var = apply_affine_calibration(pred_var, means, affine_calibration)
            pred_var = apply_global_mean_correction(pred_var, global_mean_correction)
            np.save(f"result/output/{filename}.npy", pred_var)
            if _is_enabled("PANGU_PROFILE_MEMORY") and len(time_list) == 1:
                _profile_cuda_memory("after first output postprocess")

            # Explicitly clear loop-local GPU tensor references to prevent caching allocator double-buffering peak VRAM
            if not _is_enabled("PANGU_CLEAR_INPUT_REFS", default=True):
                del invar_surface, invar_upper_air, invar_surface_with_mask, invar_upper_air_reshaped
            del invar, upper_air_shape
            del out_surface, out_upper_air
            if _is_enabled("PANGU_CPU_OUTPUT_POSTPROCESS", default=True):
                del pred_tensor


        #----------------------AI4S(时间度量不可更改)---------------------------
        # 保存到 time_list.json 文件
        with open("result/time_record.json", "w", encoding="utf-8") as f:
            json.dump(time_list, f, ensure_ascii=False, indent=4)
        #---------------------------------------------------------------------

    if torch.cuda.is_available():
        print(f"Max VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"Current VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
