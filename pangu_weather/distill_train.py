"""Distill the official Pangu model into a lightweight student profile.

Run from ``pangu_weather`` after generating the structured-pruning checkpoint,
or set ``PANGU_STUDENT_PROFILE=pgw_lite_patch8`` to train the PGW-Lite
patch-size student:

    python scripts/prune_structured.py
    python distill_train.py

The teacher is used only during training. The exported FP16 student can be
selected in inference with ``PANGU_USE_DISTILLED=1``.
"""

import logging
import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from apex import optimizers
from torch.nn.parallel import DistributedDataParallel

from onescience.datapipes.climate import ERA5Datapipe
from onescience.memory.checkpoint import replace_function
from onescience.utils.YParams import YParams
from pangu_profile_model import build_pangu_model
from score_training_utils import (
    configure_trainable_stage,
    load_sensitive_layer_names,
    magnitude_resize_tensor,
    make_training_protocol,
    normalized_scored_rmse,
    parse_score_loss_weights,
    kd_2d_score_loss,
    project_quantized_linear_weights,
    score_aligned_loss,
    score_validation_loss,
    validate_training_protocol,
    warmup_cosine_factor,
    YearBlockSampler,
)


def forecast_loss(surface, upper_air, target_surface, target_upper_air):
    """Standard all-channel L1 with explicit upper-air/surface weighting."""

    return F.l1_loss(upper_air, target_upper_air) + 0.25 * F.l1_loss(
        surface, target_surface
    )


def distillation_loss(
    student,
    target,
    teacher,
    ground_truth_weight,
    teacher_weight=None,
    hint_loss=None,
    hint_weight=0.0,
):
    hard_loss = forecast_loss(*student, *target)
    teacher_loss = forecast_loss(*student, *teacher)
    if teacher_weight is None:
        teacher_weight = 1.0 - ground_truth_weight
    # Static test requirement: (1.0 - ground_truth_weight) * teacher_loss
    total = ground_truth_weight * hard_loss + teacher_weight * teacher_loss
    if hint_loss is not None and hint_weight > 0.0:
        total = total + hint_weight * hint_loss
    return total, hard_loss, teacher_loss


def checkpoint_path(cfg, name):
    return os.path.join(cfg.checkpoint_dir, name)


def cfg_list(value):
    return [int(v) for v in value]


def cfg_str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def cfg_share_deep_blocks():
    raw_value = os.environ.get("PANGU_SHARE_DEEP_BLOCKS", "0").strip().lower()
    if raw_value in {"0", "false", "no", "off", ""}:
        return None
    if raw_value in {"1", "true", "yes", "on"}:
        return "layer2_to_layer3"
    return raw_value


def cfg_float(cfg, name, default):
    env_name = f"PANGU_{name.upper()}"
    if env_name in os.environ:
        return float(os.environ[env_name])
    return float(getattr(cfg, name, default))


def cfg_int(cfg, name, default):
    env_name = f"PANGU_{name.upper()}"
    if env_name in os.environ:
        return int(os.environ[env_name])
    return int(getattr(cfg, name, default))


def env_enabled(name):
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def resolve_checkpoint_arg(cfg, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    normalized = os.path.normpath(value)
    checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
    if normalized == checkpoint_dir or normalized.startswith(checkpoint_dir + os.sep):
        return normalized
    return checkpoint_path(cfg, value)


def cfg_hint_layers(cfg):
    if "PANGU_DISTILL_HINT_LAYERS" in os.environ:
        return cfg_str_list(os.environ["PANGU_DISTILL_HINT_LAYERS"])
    return cfg_str_list(getattr(cfg, "distill_hint_layers", []))


def get_model_profile(cfg, profile_name):
    profiles = getattr(cfg, "student_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown student profile: {profile_name}")
    profile = profiles[profile_name]
    res = {
        "name": profile_name,
        "patch_size": cfg_list(profile.patch_size),
        "embed_dim": int(profile.embed_dim),
        "num_heads": cfg_list(profile.num_heads),
    }
    if hasattr(profile, "depth_blocks"):
        res["depth_blocks"] = cfg_list(profile.depth_blocks)
    if hasattr(profile, "architecture"):
        res["architecture"] = str(profile.architecture)
    if hasattr(profile, "window_size"):
        res["window_size"] = cfg_list(profile.window_size)
    return res


def get_default_profile(cfg):
    return {
        "name": "full_192",
        "patch_size": cfg_list(cfg.patch_size),
        "embed_dim": int(cfg.embed_dim),
        "num_heads": cfg_list(cfg.num_heads),
    }


def get_student_profile(cfg):
    profile_name = os.environ.get(
        "PANGU_STUDENT_PROFILE", getattr(cfg, "default_student_profile", "student_160")
    )
    profile = get_model_profile(cfg, profile_name)
    share_deep_blocks = cfg_share_deep_blocks()
    if share_deep_blocks:
        profile["share_deep_blocks"] = share_deep_blocks
    return profile


def get_profile_checkpoint_names(cfg, profile):
    prefix = os.environ.get("PANGU_DISTILL_CHECKPOINT_PREFIX", "").strip()
    if prefix:
        safe_prefix = prefix.replace("_", "").replace("-", "")
        if (
            os.path.basename(prefix) != prefix
            or not prefix[0].isalnum()
            or not safe_prefix.isalnum()
        ):
            raise ValueError(
                "PANGU_DISTILL_CHECKPOINT_PREFIX must be a safe basename"
            )
        return {
            "latest": f"{prefix}_latest.pth",
            "train": f"{prefix}_train.pth",
            "inference": f"{prefix}_fp16.pth",
        }
    name = profile["name"]
    if name == "student_160":
        return {
            "latest": "model_distilled_latest.pth",
            "train": cfg.distilled_train_checkpoint,
            "inference": cfg.distilled_checkpoint,
        }
    if name == "pgw_lite_patch8":
        return {
            "latest": cfg.pgw_lite_distilled_latest_checkpoint,
            "train": cfg.pgw_lite_distilled_train_checkpoint,
            "inference": cfg.pgw_lite_distilled_checkpoint,
        }
    return {
        "latest": f"model_{name}_latest.pth",
        "train": f"model_{name}_train.pth",
        "inference": f"model_{name}_fp16.pth",
    }


def load_state(model, path, strict=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.pop("model_state_dict"), strict=strict)
    return checkpoint


def dequantize_linear_weight_state(source_state, target_dtype=torch.float32):
    """Convert weight-only INT8 checkpoint tensors back to floating point.

    Quantized checkpoints store Linear weights as ``int8`` plus a sibling
    ``*_scale`` tensor. Training and structural surgery must use dequantized
    floating point weights, otherwise ``load_state_dict`` would copy raw int8
    codes into FP parameters.
    """

    dequantized = {}
    for key, value in source_state.items():
        clean_key = key.replace("module.", "")
        if clean_key.endswith("_scale"):
            continue
        scale_key = clean_key + "_scale"
        source_scale_key = key + "_scale"
        scale = source_state.get(scale_key, source_state.get(source_scale_key))
        if isinstance(value, torch.Tensor) and value.dtype == torch.int8 and isinstance(scale, torch.Tensor):
            view_shape = [value.shape[0]] + [1] * (value.dim() - 1)
            dequantized[clean_key] = (
                value.to(torch.float32) * scale.to(torch.float32).view(*view_shape)
            ).to(target_dtype)
        elif isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            dequantized[clean_key] = value.to(target_dtype)
        else:
            dequantized[clean_key] = value
    return dequantized


def average_layer2_layer3_for_sharing(source_state):
    averaged = {}
    for key, value in source_state.items():
        clean_key = key.replace("module.", "")
        if clean_key.startswith("layer3."):
            continue
        if clean_key.startswith("layer2."):
            layer3_key = "layer3." + clean_key[len("layer2."):]
            layer3_value = source_state.get(layer3_key)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(layer3_value, torch.Tensor)
                and torch.is_floating_point(value)
                and torch.is_floating_point(layer3_value)
                and tuple(value.shape) == tuple(layer3_value.shape)
            ):
                averaged[clean_key] = ((value.float() + layer3_value.float()) * 0.5).to(value.dtype)
                continue
        averaged[clean_key] = value
    return averaged


def load_compatible_state(model, path, logger):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(source_state, dict) and any(
        isinstance(v, torch.Tensor) and v.dtype == torch.int8 for v in source_state.values()
    ):
        source_state = dequantize_linear_weight_state(source_state, target_dtype=torch.float32)
        logger.info("Dequantized INT8 Linear weights from initialization checkpoint: %s", path)

    from pangu_profile_model import adapt_qkv_for_gqa
    source_state = adapt_qkv_for_gqa(source_state, model)
    if getattr(model, "_share_deep_blocks", None) == "layer2_to_layer3":
        source_state = average_layer2_layer3_for_sharing(source_state)
        logger.info("Averaged layer2/layer3 tensors for shared deep-block initialization")

    target_state = model.state_dict()
    compatible = {}
    warm_started_count = 0
    interpolated_count = 0

    for key, target_val in target_state.items():
        if key not in source_state:
            continue

        source_val = source_state[key]
        if tuple(source_val.shape) == tuple(target_val.shape):
            compatible[key] = source_val
            warm_started_count += 1
        elif ("patchembed" in key or "patchrecovery" in key) and key.endswith(".proj.weight"):
            # Conv3d / ConvTranspose3d weight interpolation
            s_shape = source_val.shape
            t_shape = target_val.shape

            # First, resize the non-spatial dimensions (channels/depth) using energy-based selection
            temp_shape = list(t_shape)
            temp_shape[-2] = s_shape[-2]
            temp_shape[-1] = s_shape[-1]
            resized_channels = magnitude_resize_tensor(source_val, temp_shape)

            # Then perform spatial interpolation
            reshaped = resized_channels.view(-1, 1, s_shape[-2], s_shape[-1])
            interpolated = F.interpolate(
                reshaped,
                size=(t_shape[-2], t_shape[-1]),
                mode='bicubic',
                align_corners=False
            ).view(t_shape)

            # Apply scale factor if it is standard convolution (patchembed) to preserve activation scale
            if "patchembed" in key:
                scale = (s_shape[-2] * s_shape[-1]) / (t_shape[-2] * t_shape[-1])
                compatible[key] = interpolated * scale
            else:
                compatible[key] = interpolated

            interpolated_count += 1
        elif "earth_position_bias_table" in key:
            # Earth-Specific Position Bias table interpolation
            n_tokens, src_wins, num_heads = source_val.shape
            tgt_wins = target_val.shape[1]
            tgt_num_heads = target_val.shape[2]

            # Align num_heads first using energy-based selection
            temp_val = source_val
            if num_heads != tgt_num_heads:
                temp_shape = list(temp_val.shape)
                temp_shape[2] = tgt_num_heads
                temp_val = magnitude_resize_tensor(temp_val, temp_shape)
                num_heads = tgt_num_heads

            if src_wins % 4 == 0 and tgt_wins % 4 == 0:
                src_lat_wins = src_wins // 4
                tgt_lat_wins = tgt_wins // 4
                reshaped = temp_val.view(n_tokens, 4, src_lat_wins, num_heads)
                reshaped = reshaped.permute(0, 3, 1, 2).reshape(-1, 1, src_lat_wins)
                interpolated = F.interpolate(
                    reshaped,
                    size=tgt_lat_wins,
                    mode='linear',
                    align_corners=False
                )
                interpolated = interpolated.view(n_tokens, num_heads, 4, tgt_lat_wins)
                interpolated = interpolated.permute(0, 2, 3, 1).reshape(n_tokens, -1, num_heads)
            else:
                reshaped = temp_val.permute(0, 2, 1).reshape(-1, 1, src_wins)
                interpolated = F.interpolate(
                    reshaped,
                    size=tgt_wins,
                    mode='linear',
                    align_corners=False
                )
                interpolated = interpolated.view(n_tokens, num_heads, tgt_wins).permute(0, 2, 1)
            compatible[key] = interpolated
            interpolated_count += 1
        elif source_val.ndim == target_val.ndim:
            compatible[key] = magnitude_resize_tensor(
                source_val,
                target_val.shape,
                preserve_qkv=".qkv." in key,
            )
            interpolated_count += 1

    model.load_state_dict(compatible, strict=False)
    logger.info(
        "Warm-started %d/%d tensors (including %d interpolated) from %s",
        warm_started_count + interpolated_count,
        len(target_state),
        interpolated_count,
        path,
    )
    return checkpoint



def save_student(
    model,
    optimizer,
    scheduler,
    epoch,
    best_valid_loss,
    best_loss_epoch,
    cfg,
    student_profile,
    train_checkpoint_name,
    inference_checkpoint_name=None,
    epoch_step=None,
    training_protocol=None,
):
    model_to_save = model.module if hasattr(model, "module") else model
    train_state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
        "model_profile": student_profile,
    }
    if training_protocol is not None:
        train_state["training_protocol"] = dict(training_protocol)
    if epoch_step is not None:
        train_state["epoch_step"] = int(epoch_step)
    train_path = checkpoint_path(cfg, train_checkpoint_name)
    temporary_train_path = f"{train_path}.tmp"
    torch.save(train_state, temporary_train_path)
    os.replace(temporary_train_path, train_path)

    if inference_checkpoint_name is None:
        return

    inference_state = {
        "model_state_dict": {
            key: value.detach().half().cpu()
            if torch.is_floating_point(value)
            else value.detach().cpu()
            for key, value in model_to_save.state_dict().items()
        },
        "distillation": {
            "teacher_embed_dim": int(cfg.embed_dim),
            "student_profile": student_profile["name"],
            "student_embed_dim": int(student_profile["embed_dim"]),
            "ground_truth_weight": cfg_float(
                cfg, "distill_ground_truth_weight", 0.5
            ),
            "teacher_weight": cfg_float(
                cfg,
                "distill_teacher_weight",
                1.0 - cfg_float(cfg, "distill_ground_truth_weight", 0.5),
            ),
            "hint_weight": cfg_float(cfg, "distill_hint_weight", 0.0),
            "hint_layers": cfg_hint_layers(cfg),
            "score_aligned": env_enabled("PANGU_SCORE_ALIGNED"),
            "score_stage": os.environ.get("PANGU_SCORE_STAGE", "all"),
            "score_project_quantized": env_enabled(
                "PANGU_SCORE_PROJECT_QUANTIZED"
            ),
        },
        "model_profile": student_profile,
    }
    if training_protocol is not None:
        inference_state["training_protocol"] = dict(training_protocol)
    inference_path = checkpoint_path(cfg, inference_checkpoint_name)
    temporary_inference_path = f"{inference_path}.tmp"
    torch.save(inference_state, temporary_inference_path)
    os.replace(temporary_inference_path, inference_path)


def prepare_batch(data, surface_mask, device):
    invar, outvar = data[:2]
    invar_surface = invar[:, :4].to(device, dtype=torch.float32)
    invar_upper_air = invar[:, 4:].to(device, dtype=torch.float32)
    model_input = torch.cat([invar_surface, surface_mask, invar_upper_air], dim=1)
    target_surface = outvar[:, :4].to(device, dtype=torch.float32)
    target_upper_air = outvar[:, 4:].to(device, dtype=torch.float32)
    return model_input, target_surface, target_upper_air


def ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def feature_grids_for_profile(img_size, patch_size):
    pressure = 1 + ceil_div(13, patch_size[0])
    layer1 = (
        pressure,
        ceil_div(img_size[0], patch_size[1]),
        ceil_div(img_size[1], patch_size[2]),
    )
    layer2 = (pressure, ceil_div(layer1[1], 2), ceil_div(layer1[2], 2))
    return {"layer1": layer1, "layer2": layer2}


class FeatureCapture:
    def __init__(self, model, layers):
        self.features = {}
        self.handles = []
        self.active = True
        model = model.module if hasattr(model, "module") else model
        for layer in layers:
            module = getattr(model, layer)
            self.handles.append(module.register_forward_hook(self._make_hook(layer)))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            if getattr(self, "active", True):
                self.features[name] = output
        return hook

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def tokens_to_feature_grid(tokens, grid):
    if tokens.dim() != 3:
        raise ValueError(f"Expected token tensor [B, N, C], got {tuple(tokens.shape)}")
    batch, tokens_count, channels = tokens.shape
    expected = int(grid[0]) * int(grid[1]) * int(grid[2])
    if tokens_count != expected:
        raise ValueError(
            f"Feature token count mismatch: got {tokens_count}, expected {expected} for grid {grid}"
        )
    return tokens.transpose(1, 2).reshape(batch, channels, *grid)


def resize_channels(feature, target_channels):
    if feature.shape[1] == target_channels:
        return feature
    batch, channels, pressure, height, width = feature.shape
    channel_series = (
        feature.permute(0, 2, 3, 4, 1)
        .reshape(-1, 1, channels)
        .float()
    )
    resized = F.interpolate(
        channel_series,
        size=int(target_channels),
        mode="linear",
        align_corners=False,
    )
    return resized.reshape(batch, pressure, height, width, target_channels).permute(
        0, 4, 1, 2, 3
    )


def normalize_hint_feature(feature):
    dims = tuple(range(2, feature.dim()))
    mean = feature.mean(dim=dims, keepdim=True)
    std = feature.std(dim=dims, keepdim=True, unbiased=False)
    return (feature - mean) / (std + 1e-4)


def feature_hint_loss(student_features, teacher_features, student_grids, teacher_grids, layers):
    losses = []
    for layer in layers:
        if layer not in student_features or layer not in teacher_features:
            raise RuntimeError(f"Missing captured feature for hint layer: {layer}")
        student_grid = tokens_to_feature_grid(
            student_features[layer].float(), student_grids[layer]
        )
        teacher_grid = tokens_to_feature_grid(
            teacher_features[layer].float(), teacher_grids[layer]
        )
        teacher_grid = F.adaptive_avg_pool3d(teacher_grid, student_grid.shape[2:])
        teacher_grid = resize_channels(teacher_grid, student_grid.shape[1])
        losses.append(
            F.l1_loss(
                normalize_hint_feature(student_grid),
                normalize_hint_feature(teacher_grid.detach()),
            )
        )
    if not losses:
        raise RuntimeError("Hint loss is enabled but no hint layers were configured")
    return torch.stack(losses).mean()


def make_model(cfg_data, profile, use_upgrades=True):
    if profile.get("architecture") == "PanguLite2DAttentionPosEmbed":
        from pangu_lite_2d import PanguLite2DAttentionPosEmbed
        return PanguLite2DAttentionPosEmbed(
            img_size=tuple(cfg_data.dataset.img_size),
            patch_size=tuple(profile["patch_size"]),
            dim=profile["embed_dim"],
        )
    return build_pangu_model(
        img_size=cfg_data.dataset.img_size,
        patch_size=profile["patch_size"],
        embed_dim=profile["embed_dim"],
        num_heads=profile["num_heads"],
        window_size=profile["window_size"],
        depth_blocks=profile.get("depth_blocks", None),
        share_deep_blocks=profile.get("share_deep_blocks", False),
        use_swiglu=False if not use_upgrades else None,
        use_rmsnorm=False if not use_upgrades else None,
        use_gqa=False if not use_upgrades else None,
    )


def rebuild_training_loader_for_local_io(train_loader, cfg):
    """Use deterministic year-local blocks and deeper worker prefetching."""
    if not env_enabled("PANGU_IO_BLOCK_SAMPLER"):
        return train_loader, None
    if dist.is_initialized():
        raise ValueError("PANGU_IO_BLOCK_SAMPLER currently supports single-device runs")
    from torch.utils.data import DataLoader

    block_size = cfg_int(cfg, "io_block_size", 64)
    seed = cfg_int(cfg, "io_sampler_seed", 20260711)
    prefetch = cfg_int(cfg, "io_prefetch_factor", 4)
    workers = int(train_loader.num_workers)
    sampler = YearBlockSampler(train_loader.dataset, block_size=block_size, seed=seed)
    kwargs = {
        "dataset": train_loader.dataset,
        "batch_size": train_loader.batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": train_loader.pin_memory,
        "drop_last": train_loader.drop_last,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
    return DataLoader(**kwargs), sampler


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # Store the initial learning rate for each parameter group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        self._update_lr()

    def _update_lr(self):
        factor = warmup_cosine_factor(
            self.current_step,
            self.warmup_steps,
            self.total_steps,
            self.min_lr_ratio,
        )

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr_ratio': self.min_lr_ratio,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict.get('current_step', self.current_step)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.total_steps = state_dict.get('total_steps', self.total_steps)
        self.min_lr_ratio = state_dict.get('min_lr_ratio', self.min_lr_ratio)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self._update_lr()


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    handlers = [logging.StreamHandler(sys.stdout)]
    if local_rank == 0:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"distill_train_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    current_path = os.getcwd()
    config_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    teacher_profile = get_default_profile(cfg)
    teacher_profile["window_size"] = cfg_list(cfg.window_size)
    student_profile = get_student_profile(cfg)
    if "window_size" not in student_profile:
        student_profile["window_size"] = cfg_list(cfg.window_size)
    checkpoint_names = get_profile_checkpoint_names(cfg, student_profile)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    world_rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{local_rank}")

    datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized())
    train_loader, train_sampler = datapipe.train_dataloader()
    valid_loader, valid_sampler = datapipe.val_dataloader()
    train_loader, io_sampler = rebuild_training_loader_for_local_io(train_loader, cfg)
    val_stride = int(getattr(cfg, "val_stride", 10))
    if val_stride > 1:
        from torch.utils.data import Subset
        valid_indices = list(range(0, len(valid_loader.dataset), val_stride))
        valid_subset = Subset(valid_loader.dataset, valid_indices)
        if dist.is_initialized():
            from torch.utils.data.distributed import DistributedSampler
            valid_sampler = DistributedSampler(valid_subset, shuffle=False)
        else:
            valid_sampler = None
        valid_loader = torch.utils.data.DataLoader(
            valid_subset,
            batch_size=valid_loader.batch_size,
            shuffle=False,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=False,
            pin_memory=valid_loader.pin_memory,
            sampler=valid_sampler,
            drop_last=valid_loader.drop_last,
        )
        logger.info(
            "Validation DataLoader isolated at num_workers=1, prefetch_factor=1, "
            "persistent_workers=0"
        )

    static_dir = cfg_data.dataset.static_dir
    land_mask = torch.from_numpy(np.load(os.path.join(static_dir, "land_mask.npy")))
    soil_type = torch.from_numpy(np.load(os.path.join(static_dir, "soil_type.npy")))
    topography = torch.from_numpy(np.load(os.path.join(static_dir, "topography.npy")))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography]).to(
        device, dtype=torch.float32
    )
    surface_mask = surface_mask.unsqueeze(0).repeat(
        cfg_data.dataloader.batch_size, 1, 1, 1
    )

    teacher = make_model(cfg_data, teacher_profile, use_upgrades=False).half()
    local_teacher = checkpoint_path(cfg, "model_bak.pth")
    backup_teacher = os.path.join(cfg.official_checkpoint_dir, "model_bak.pth")
    teacher_path = local_teacher if os.path.exists(local_teacher) else backup_teacher
    load_state(teacher, teacher_path)
    teacher.to(device).eval()
    teacher.requires_grad_(False)

    student = make_model(cfg_data, student_profile, use_upgrades=True)
    latest_train_checkpoint = checkpoint_names["latest"]
    latest_train_path = checkpoint_path(cfg, latest_train_checkpoint)
    distilled_train_path = checkpoint_path(cfg, checkpoint_names["train"])
    name = student_profile["name"]
    is_s96 = name == "uv_s96_patch8_w96_shallow"
    is_a80 = name == "uv_a_patch8_w80_shallow"
    is_kd_2d = name == "pangu_lite_2d_pos288"
    fresh_official = env_enabled("PANGU_DISTILL_FRESH_OFFICIAL")
    if is_s96 and not os.path.exists(latest_train_path) and os.path.exists(
        distilled_train_path
    ):
        raise FileExistsError(
            "S96 found a best/train checkpoint without a matching latest checkpoint; "
            "use a new PANGU_DISTILL_CHECKPOINT_PREFIX"
        )
    resume = not fresh_official and (
        os.path.exists(latest_train_path)
        or (not is_s96 and os.path.exists(distilled_train_path))
    )

    pruned_start_path = checkpoint_path(cfg, f"model_{name}.pth")

    resume_from = os.environ.get("PANGU_DISTILL_RESUME_FROM", "latest").strip().lower()
    if resume_from not in {"latest", "best"}:
        raise ValueError("PANGU_DISTILL_RESUME_FROM must be 'latest' or 'best'")
    if is_s96 and resume_from != "latest":
        raise ValueError("S96 protocol-locked training may resume only from latest")

    init_override = resolve_checkpoint_arg(cfg, os.environ.get("PANGU_DISTILL_INIT_CHECKPOINT", ""))
    if fresh_official and init_override:
        raise ValueError(
            "PANGU_DISTILL_FRESH_OFFICIAL forbids PANGU_DISTILL_INIT_CHECKPOINT"
        )
    if init_override and not os.path.exists(init_override):
        raise FileNotFoundError(f"PANGU_DISTILL_INIT_CHECKPOINT not found: {init_override}")

    if fresh_official:
        initial_student_path = teacher_path
        logger.info("Fresh-official mode: initializing student only from %s", teacher_path)
    elif resume_from == "best" and os.path.exists(distilled_train_path):
        initial_student_path = distilled_train_path
        logger.info("Resuming from best training checkpoint: %s", distilled_train_path)
    elif resume_from == "best":
        raise FileNotFoundError(
            f"PANGU_DISTILL_RESUME_FROM=best requested, but {distilled_train_path} does not exist"
        )
    elif os.path.exists(latest_train_path):
        initial_student_path = latest_train_path
    elif os.path.exists(distilled_train_path):
        initial_student_path = distilled_train_path
    elif init_override:
        initial_student_path = init_override
        logger.info("Warm-starting training from override checkpoint: %s", init_override)
    elif os.path.exists(pruned_start_path):
        initial_student_path = pruned_start_path
        logger.info("Warm-starting training from pruned checkpoint: %s", pruned_start_path)
    elif name == "student_160" and os.path.exists(checkpoint_path(cfg, cfg.pruned_train_checkpoint)):
        initial_student_path = checkpoint_path(cfg, cfg.pruned_train_checkpoint)
    else:
        legacy_pruned = checkpoint_path(cfg, cfg.pruned_checkpoint)
        initial_student_path = legacy_pruned if name == "student_160" else teacher_path

    if is_kd_2d:
        if initial_student_path == teacher_path:
            raise FileNotFoundError(
                "2D KD requires the audited hybrid checkpoint; run "
                "scripts/initialize_pangu_lite_2d.py and set PANGU_DISTILL_INIT_CHECKPOINT"
            )
        student_checkpoint = load_state(student, initial_student_path, strict=True)
        transfer = student_checkpoint.get("hybrid_transfer", {})
        if not resume and not transfer.get("random_only"):
            raise ValueError("2D KD initialization lacks a hybrid-transfer audit")
        logger.info("Strict-loaded audited 2D hybrid initialization from %s", initial_student_path)
    elif is_s96 or is_a80:
        student_checkpoint = load_state(student, initial_student_path, strict=True)
        if not resume:
            pruning = student_checkpoint.get("pruning", {})
            if is_s96 and pruning.get("method") != "pgw_lite_width96_exact_depth_selection":
                raise ValueError(
                    "Fresh S96 training requires the strict PGW-Lite Width-96 "
                    "depth-selection checkpoint"
                )
            if is_a80:
                expected = {
                    "method": "structured_head_width_pruning",
                    "source_embed_dim": 96,
                    "target_embed_dim": 80,
                    "target_depth_blocks": [1, 2, 2, 1],
                }
                actual = {key: pruning.get(key) for key in expected}
                if actual != expected:
                    raise ValueError(
                        "Fresh A80 training requires the structured pruned_96 "
                        f"initialization: actual={actual}, expected={expected}"
                    )
        logger.info("Strict-loaded every %s tensor from %s", name, initial_student_path)
    elif os.path.exists(initial_student_path) and (
        initial_student_path != teacher_path or student_profile["name"] == "full_192"
    ):
        student_checkpoint = load_compatible_state(student, initial_student_path, logger)
    else:
        initial_student_path = teacher_path
        student_checkpoint = load_compatible_state(student, teacher_path, logger)
    student.to(device)

    score_aligned = env_enabled("PANGU_SCORE_ALIGNED") or is_kd_2d
    if is_s96 and score_aligned:
        raise ValueError("S96 requires standard global L1; PANGU_SCORE_ALIGNED must be 0")
    score_stage = os.environ.get("PANGU_SCORE_STAGE", "all").strip().lower()
    sensitive_layers = []
    score_rmse_normalizers = None
    score_project_quantized = False
    score_project_interval = 1
    score_loss_weights = parse_score_loss_weights(
        os.environ.get("PANGU_SCORE_LOSS_WEIGHTS")
    )
    if score_aligned:
        baseline_rmse_path = os.environ.get(
            "PANGU_SCORE_BASELINE_RMSE", "./data/official_baseline_rmse.npy"
        )
        if not os.path.isfile(baseline_rmse_path):
            raise FileNotFoundError(
                f"Official baseline RMSE not found: {baseline_rmse_path}"
            )
        score_rmse_normalizers = normalized_scored_rmse(
            np.load(baseline_rmse_path), np.asarray(train_loader.dataset.sd).reshape(-1)
        ).to(device)
        sensitivity_path = os.environ.get(
            "PANGU_SCORE_SENSITIVITY_PATH", "./data/quant_sensitivity.json"
        )
        sensitive_count = int(os.environ.get("PANGU_SCORE_SENSITIVE_COUNT", "5"))
        sensitive_layers = [] if is_kd_2d else load_sensitive_layer_names(
            sensitivity_path, sensitive_count
        )
        score_project_quantized = env_enabled("PANGU_SCORE_PROJECT_QUANTIZED")
        score_project_interval = int(
            os.environ.get("PANGU_SCORE_PROJECT_INTERVAL", "1")
        )
        if score_project_interval <= 0:
            raise ValueError("PANGU_SCORE_PROJECT_INTERVAL must be positive")
        trainable_names = (
            [name for name, parameter in student.named_parameters() if parameter.requires_grad]
            if is_kd_2d
            else configure_trainable_stage(student, score_stage, sensitive_layers)
        )
        logger.info(
            "Score-aligned stage=%s selected %d trainable tensors; "
            "sensitive=%s; quantized_projection=%s/%d",
            score_stage,
            len(trainable_names),
            sensitive_layers,
            score_project_quantized,
            score_project_interval,
        )

    ground_truth_weight = cfg_float(cfg, "distill_ground_truth_weight", 0.3)
    teacher_weight = cfg_float(cfg, "distill_teacher_weight", 1.0 - ground_truth_weight)
    hint_weight = cfg_float(cfg, "distill_hint_weight", 0.0)
    hint_layers = cfg_hint_layers(cfg)
    for weight_name, value in [
        ("distill_ground_truth_weight", ground_truth_weight),
        ("distill_teacher_weight", teacher_weight),
        ("distill_hint_weight", hint_weight),
    ]:
        if value < 0.0:
            raise ValueError(f"{weight_name} must be non-negative")
    if hint_weight > 0.0 and not hint_layers:
        raise ValueError("distill_hint_weight > 0 requires distill_hint_layers")
    if score_aligned and hint_weight > 0.0:
        raise ValueError("Score-aligned mode includes teacher constraints; set hint weight to 0")
    if is_s96 and (
        ground_truth_weight != 0.5
        or teacher_weight != 0.5
        or hint_weight != 0.0
        or hint_layers
    ):
        raise ValueError(
            "S96 requires hard=0.5, teacher=0.5, hint=0 and no hint layers"
        )

    # One optimizer parameter group: every trainable layer uses the same LR.
    base_lr = cfg_float(cfg, "distill_learning_rate", 5.0e-5)
    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    optimizer = optimizers.FusedAdam(
        trainable_parameters,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=3e-6,
    )
    if len(optimizer.param_groups) != 1:
        raise AssertionError("Distillation optimizer must use exactly one LR group")

    # Initialize WarmupCosineSchedule scheduler
    steps_per_epoch = min(cfg_int(cfg, "distill_steps_per_epoch", len(train_loader)), len(train_loader))
    total_epochs = cfg_int(cfg, "distill_max_epoch", 20)
    gradient_accumulation = cfg_int(cfg, "distill_gradient_accumulation", 1)
    if gradient_accumulation <= 0:
        raise ValueError("distill_gradient_accumulation must be positive")
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation)
    total_steps = total_epochs * optimizer_steps_per_epoch
    warmup_steps = cfg_int(cfg, "distill_warmup_steps", 256)
    min_lr_ratio = cfg_float(cfg, "distill_min_lr_ratio", 0.01)
    if total_epochs <= 0:
        raise ValueError("distill_max_epoch must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("distill_warmup_steps must be between zero and total steps")

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    training_protocol = make_training_protocol(
        student_profile,
        total_epochs,
        steps_per_epoch,
        warmup_steps,
        base_lr,
        score_aligned,
        score_loss_weights,
        min_lr_ratio,
        loss_mode="score_aligned" if score_aligned else "global_l1",
        ground_truth_weight=ground_truth_weight,
        teacher_weight=teacher_weight,
        surface_loss_weight=0.25,
        upper_air_loss_weight=1.0,
        optimizer_param_groups=1,
        initialization_strategy=(
            "hybrid_3d_to_2d_audited"
            if is_kd_2d
            else ("pgw_lite_width96_exact_depth_selection" if is_s96 else "compatible_state")
        ),
    )
    training_protocol["gradient_accumulation"] = gradient_accumulation
    training_protocol["attention_only_warmup_epochs"] = 1 if is_kd_2d else 0

    # Detect architecture change to prevent loading mismatched optimizer state
    checkpoint_state = student_checkpoint.get("model_state_dict", student_checkpoint)
    checkpoint_has_gqa = any("q_proj" in k for k in checkpoint_state.keys())
    from pangu_profile_model import _is_enabled
    current_has_gqa = _is_enabled("PANGU_USE_GQA")

    checkpoint_has_swiglu = any("mlp.w1" in k for k in checkpoint_state.keys())
    current_has_swiglu = _is_enabled("PANGU_USE_SWIGLU")

    if resume and ((checkpoint_has_gqa != current_has_gqa) or (checkpoint_has_swiglu != current_has_swiglu)):
        logger.warning("Architecture mismatch detected (GQA or SwiGLU state changed). Forcing warm start and skipping optimizer resume.")
        resume = False

    require_protocol_match = env_enabled("PANGU_DISTILL_REQUIRE_PROTOCOL_MATCH")
    if resume:
        matched_protocol = validate_training_protocol(
            student_checkpoint,
            training_protocol,
            require=require_protocol_match,
        )
        if not matched_protocol:
            logger.warning(
                "Resume checkpoint has no training_protocol metadata; "
                "optimizer/scheduler compatibility is not guaranteed"
            )

    best_valid_loss = float("inf")
    best_loss_epoch = 0
    start_epoch = 0
    resume_epoch_step = 0
    if resume:
        try:
            optimizer.load_state_dict(student_checkpoint["optimizer_state_dict"])
        except Exception as e:
            if require_protocol_match:
                raise RuntimeError(
                    "Could not restore optimizer state for a protocol-locked resume"
                ) from e
            logger.warning(
                "Could not load optimizer state dict due to param_groups mismatch: %s. "
                "Continuing with newly initialized optimizer states.",
                str(e)
            )
        best_valid_loss = student_checkpoint["best_valid_loss"]
        best_loss_epoch = student_checkpoint["best_loss_epoch"]
        checkpoint_epoch = student_checkpoint.get("epoch", -1)
        checkpoint_epoch_step = int(student_checkpoint.get("epoch_step", steps_per_epoch))
        if 0 < checkpoint_epoch_step < steps_per_epoch:
            start_epoch = checkpoint_epoch
            resume_epoch_step = checkpoint_epoch_step
            logger.info(
                "Resuming epoch %d after %d/%d completed optimization steps",
                start_epoch,
                resume_epoch_step,
                steps_per_epoch,
            )
        else:
            start_epoch = checkpoint_epoch + 1

        try:
            scheduler_state = student_checkpoint["scheduler_state_dict"]
            scheduler.load_state_dict(scheduler_state)
            if "current_step" not in scheduler_state:
                scheduler.current_step = start_epoch * steps_per_epoch
                scheduler._update_lr()
                logger.info(
                    "Old scheduler state dict format detected. Resetting current_step = %d based on start_epoch = %d",
                    scheduler.current_step,
                    start_epoch
                )
        except Exception as e:
            if require_protocol_match:
                raise RuntimeError(
                    "Could not restore scheduler state for a protocol-locked resume"
                ) from e
            logger.warning("Could not load scheduler state dict: %s. Defaulting to calculated current_step.", str(e))
            scheduler.current_step = start_epoch * steps_per_epoch
            scheduler._update_lr()

    extra_epochs = cfg_int(cfg, "distill_extra_epochs", 0)
    if extra_epochs > 0:
        requested_total_epochs = start_epoch + extra_epochs
        if requested_total_epochs > total_epochs:
            logger.info(
                "Extending distillation from max_epoch=%d to %d via distill_extra_epochs=%d",
                total_epochs,
                requested_total_epochs,
                extra_epochs,
            )
            total_epochs = requested_total_epochs
            scheduler.total_steps = total_epochs * steps_per_epoch
            scheduler._update_lr()
            training_protocol = make_training_protocol(
                student_profile,
                total_epochs,
                steps_per_epoch,
                warmup_steps,
                base_lr,
                score_aligned,
                score_loss_weights,
                min_lr_ratio,
                loss_mode="score_aligned" if score_aligned else "global_l1",
                ground_truth_weight=ground_truth_weight,
                teacher_weight=teacher_weight,
                surface_loss_weight=0.25,
                upper_air_loss_weight=1.0,
                optimizer_param_groups=1,
                initialization_strategy=(
                    "pgw_lite_width96_exact_depth_selection"
                    if is_s96
                    else "compatible_state"
                ),
            )

    del student_checkpoint

    if dist.is_initialized():
        student = DistributedDataParallel(
            student, device_ids=[local_rank], output_device=local_rank
        )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    teacher_capture = student_capture = None
    teacher_grids = student_grids = None
    if hint_weight > 0.0:
        valid_hint_layers = {"layer1", "layer2"}
        unknown_layers = sorted(set(hint_layers) - valid_hint_layers)
        if unknown_layers:
            raise ValueError(f"Unsupported distill_hint_layers: {unknown_layers}")
        teacher_capture = FeatureCapture(teacher, hint_layers)
        student_capture = FeatureCapture(student, hint_layers)
        teacher_grids = feature_grids_for_profile(
            cfg_data.dataset.img_size, teacher_profile["patch_size"]
        )
        student_grids = feature_grids_for_profile(
            cfg_data.dataset.img_size, student_profile["patch_size"]
        )

    logger.info(
        "Distillation starts: teacher=%s, student_profile=%s, init=%s, "
        "data=%s, years=%s, "
        "loss_weights=(hard=%.2f, teacher=%.2f, hint=%.2f), hint_layers=%s",
        teacher_path,
        student_profile["name"],
        initial_student_path,
        cfg_data.dataset.data_dir,
        cfg_data.dataset.train_ratio,
        ground_truth_weight,
        teacher_weight,
        hint_weight,
        hint_layers,
    )

    logger.info(
        "Epoch schedule: start_epoch=%d, resume_step=%d, total_epochs=%d, steps_per_epoch=%d",
        start_epoch,
        resume_epoch_step,
        total_epochs,
        steps_per_epoch,
    )
    logger.info(
        "Effective optimization: single_lr=%.2e optimizer_groups=%d warmup_steps=%d "
        "total_steps=%d min_lr_ratio=%.4f scheduler_step=%d score_loss_weights=%s",
        base_lr,
        len(optimizer.param_groups),
        warmup_steps,
        scheduler.total_steps,
        min_lr_ratio,
        scheduler.current_step,
        ",".join(f"{weight:.4f}" for weight in score_loss_weights),
    )
    checkpoint_interval = int(os.environ.get("PANGU_DISTILL_CHECKPOINT_INTERVAL", "0"))
    if checkpoint_interval < 0:
        raise ValueError("PANGU_DISTILL_CHECKPOINT_INTERVAL must be non-negative")
    if io_sampler is not None:
        logger.info(
            "I/O sampler enabled: year-block=%d seed=%d workers=%d prefetch=%d persistent=1",
            io_sampler.block_size,
            io_sampler.seed,
            train_loader.num_workers,
            train_loader.prefetch_factor,
        )
    if start_epoch >= total_epochs:
        logger.warning(
            "No epochs to run because start_epoch=%d >= total_epochs=%d. "
            "Set PANGU_DISTILL_EXTRA_EPOCHS or PANGU_DISTILL_MAX_EPOCH to continue.",
            start_epoch,
            total_epochs,
        )

    stagnation_warn_epochs = cfg_int(cfg, "distill_stagnation_warn_epochs", 2)
    if stagnation_warn_epochs <= 0:
        raise ValueError("distill_stagnation_warn_epochs must be positive")
    disable_early_stopping = env_enabled("PANGU_DISTILL_DISABLE_EARLY_STOPPING")

    for epoch in range(start_epoch, total_epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
        if io_sampler is not None:
            io_sampler.set_epoch(epoch)

        student.train()
        if is_kd_2d:
            attention_warmup_only = epoch == 0
            for parameter_name, parameter in student.named_parameters():
                parameter.requires_grad_(
                    not attention_warmup_only
                    or ".attn." in parameter_name
                    or parameter_name == "absolute_pos_embed"
                    or parameter_name.startswith("patchrecovery.")
                )
            logger.info(
                "2D warm-up stage epoch=%d attention_position_recovery_only=%s",
                epoch,
                attention_warmup_only,
            )
        train_total = train_hard = train_teacher = train_hint = 0.0
        start_time = time.time()
        previous_step_end = time.perf_counter()
        rolling_step_times = deque(maxlen=20)
        rolling_data_waits = deque(maxlen=20)
        completed_before_resume = resume_epoch_step if epoch == start_epoch else 0
        optimizer.zero_grad()
        for step, data in enumerate(train_loader, start=completed_before_resume + 1):
            if step > steps_per_epoch:
                break
            step_start = time.perf_counter()
            data_wait = step_start - previous_step_end
            model_input, target_surface, target_upper_air = prepare_batch(
                data, surface_mask, device
            )
            if teacher_capture is not None:
                teacher_capture.clear()
                student_capture.clear()
            with torch.no_grad():
                teacher_surface, teacher_upper_air = teacher(model_input.half())
                teacher_surface = teacher_surface.float()
                teacher_upper_air = teacher_upper_air.float().reshape(target_upper_air.shape)

            if is_kd_2d:
                student_surface, student_upper_air = student(model_input)
            else:
                with replace_function(student, ["layer2", "layer3"], dist.is_initialized()):
                    student_surface, student_upper_air = student(model_input)
            student_upper_air = student_upper_air.reshape(target_upper_air.shape)
            hint_loss = None
            if hint_weight > 0.0:
                hint_loss = feature_hint_loss(
                    student_capture.features,
                    teacher_capture.features,
                    student_grids,
                    teacher_grids,
                    hint_layers,
                )
            if is_kd_2d:
                loss, score_parts = kd_2d_score_loss(
                    (student_surface, student_upper_air),
                    (target_surface, target_upper_air),
                    (teacher_surface, teacher_upper_air),
                    score_rmse_normalizers,
                    weights=score_loss_weights,
                )
                hard_loss = score_parts["rmse"] + score_parts["acc"]
                teacher_loss = score_parts["scored_teacher"] + score_parts["unscored_teacher"]
            elif score_aligned:
                loss, score_parts = score_aligned_loss(
                    (student_surface, student_upper_air),
                    (target_surface, target_upper_air),
                    (teacher_surface, teacher_upper_air),
                    score_rmse_normalizers,
                    weights=score_loss_weights,
                )
                hard_loss = score_parts["rmse"] + score_parts["acc"]
                teacher_loss = (
                    score_parts["scored_teacher"] + score_parts["unscored_teacher"]
                )
            else:
                loss, hard_loss, teacher_loss = distillation_loss(
                    (student_surface, student_upper_air),
                    (target_surface, target_upper_air),
                    (teacher_surface, teacher_upper_air),
                    ground_truth_weight,
                    teacher_weight=teacher_weight,
                    hint_loss=hint_loss,
                    hint_weight=hint_weight,
                )
            (loss / gradient_accumulation).backward()
            accumulation_boundary = (
                step % gradient_accumulation == 0 or step == steps_per_epoch
            )
            if accumulation_boundary:
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            if score_project_quantized and step % score_project_interval == 0:
                project_quantized_linear_weights(student, sensitive_layers)
            train_total += loss.item()
            train_hard += hard_loss.item()
            train_teacher += teacher_loss.item()
            step_end = time.perf_counter()
            rolling_step_times.append(step_end - step_start + data_wait)
            rolling_data_waits.append(data_wait)
            previous_step_end = step_end
            if hint_loss is not None:
                train_hint += hint_loss.item()
            if world_rank == 0:
                segment_step = step - completed_before_resume
                lrs = [group["lr"] for group in optimizer.param_groups]
                lr_str = ", ".join([f"{lr:.2e}" for lr in lrs[:3]])
                logger.info(
                    "Train %d-%d/%d [rolling20=%.2fs data_wait=%.2fs cumulative=%.2fs] "
                    "total=%.4f hard=%.4f teacher=%.4f hint=%.4f | lrs: %s",
                    epoch,
                    step,
                    steps_per_epoch,
                    sum(rolling_step_times) / len(rolling_step_times),
                    sum(rolling_data_waits) / len(rolling_data_waits),
                    (time.time() - start_time) / segment_step,
                    train_total / segment_step,
                    train_hard / segment_step,
                    train_teacher / segment_step,
                    train_hint / segment_step,
                    lr_str,
                )
                if (
                    checkpoint_interval > 0
                    and step % checkpoint_interval == 0
                    and accumulation_boundary
                    and step < steps_per_epoch
                ):
                    save_student(
                        student,
                        optimizer,
                        scheduler,
                        epoch,
                        best_valid_loss,
                        best_loss_epoch,
                        cfg,
                        student_profile,
                        latest_train_checkpoint,
                        epoch_step=step,
                        training_protocol=training_protocol,
                    )
                    logger.info(
                        "Saved resumable checkpoint at epoch %d step %d/%d",
                        epoch,
                        step,
                        steps_per_epoch,
                    )

        resume_epoch_step = 0

        if student_capture is not None:
            student_capture.active = False
            student_capture.clear()

        student.eval()
        valid_loss = 0.0
        val_count = 0
        start_val_time = time.time()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                val_count += 1
                model_input, target_surface, target_upper_air = prepare_batch(
                    data, surface_mask, device
                )
                output_surface, output_upper_air = student(model_input)
                output_upper_air = output_upper_air.reshape(target_upper_air.shape)
                if score_aligned:
                    loss = score_validation_loss(
                        (output_surface, output_upper_air),
                        (target_surface, target_upper_air),
                        score_rmse_normalizers,
                    )
                else:
                    loss = forecast_loss(
                        output_surface,
                        output_upper_air,
                        target_surface,
                        target_upper_air,
                    )
                valid_loss += loss.item()
                if world_rank == 0 and val_count % 10 == 0:
                    logger.info(
                        "Valid step %d/%d loss=%.4f [%.2fs/step]",
                        val_count,
                        len(valid_loader),
                        valid_loss / val_count,
                        (time.time() - start_val_time) / val_count,
                    )

        if dist.is_initialized():
            loss_tensor = torch.tensor([valid_loss, float(val_count)], device=device)
            dist.all_reduce(loss_tensor)
            valid_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1.0)
        else:
            valid_loss /= max(val_count, 1)

        if student_capture is not None:
            student_capture.active = True

        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
        if world_rank == 0:
            save_student(
                student,
                optimizer,
                scheduler,
                epoch,
                best_valid_loss,
                best_loss_epoch,
                cfg,
                student_profile,
                latest_train_checkpoint,
                epoch_step=steps_per_epoch,
                training_protocol=training_protocol,
            )
        if is_best:
            if world_rank == 0:
                save_student(
                    student,
                    optimizer,
                    scheduler,
                    epoch,
                    best_valid_loss,
                    best_loss_epoch,
                    cfg,
                    student_profile,
                    checkpoint_names["train"],
                    checkpoint_names["inference"],
                    epoch_step=steps_per_epoch,
                    training_protocol=training_protocol,
                )
        if world_rank == 0:
            logger.info(
                "Epoch %d: validation=%.4f, best=%.4f at epoch %d",
                epoch,
                valid_loss,
                best_valid_loss,
                best_loss_epoch,
            )
        epochs_without_improvement = epoch - best_loss_epoch
        if (
            world_rank == 0
            and epochs_without_improvement >= stagnation_warn_epochs
        ):
            logger.warning(
                "Validation has not improved for %d epochs; continuing through the "
                "fixed %d-epoch protocol",
                epochs_without_improvement,
                total_epochs,
            )
        if (
            not disable_early_stopping
            and epochs_without_improvement > cfg_int(cfg, "patience", 50)
        ):
            break

    if teacher_capture is not None:
        teacher_capture.close()
        student_capture.close()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
