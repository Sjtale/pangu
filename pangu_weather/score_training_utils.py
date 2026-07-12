"""Loss and parameter-freezing helpers for scored-channel recovery training."""

from __future__ import annotations

import json
import math
import random

import torch
import torch.nn.functional as F


SCORED_UPPER_INDICES = (2, 3, 5, 15, 16, 18, 28, 29, 31, 44, 57)
SCORED_CHANNEL_INDICES = (0, 1, 2, 3, 6, 7, 9, 19, 20, 22, 32, 33, 35, 48, 61)


def parse_score_loss_weights(value, default=(0.50, 0.25, 0.20, 0.05)):
    """Parse the four score-aligned loss weights before training starts."""
    if value is None or str(value).strip() == "":
        weights = tuple(float(weight) for weight in default)
    else:
        try:
            weights = tuple(float(item.strip()) for item in str(value).split(","))
        except ValueError as error:
            raise ValueError(
                "PANGU_SCORE_LOSS_WEIGHTS must contain four comma-separated numbers"
            ) from error
    if len(weights) != 4:
        raise ValueError(
            "PANGU_SCORE_LOSS_WEIGHTS must contain exactly four values"
        )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError(
            "PANGU_SCORE_LOSS_WEIGHTS values must be finite and non-negative"
        )
    return weights


def warmup_cosine_factor(step, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Return the LR multiplier for a 1-indexed optimization step."""
    step = int(step)
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    min_lr_ratio = float(min_lr_ratio)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between zero and one")
    step = min(total_steps, max(1, step))
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(1.0, max(0.0, progress))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def make_training_protocol(
    student_profile,
    total_epochs,
    steps_per_epoch,
    warmup_steps,
    base_lr,
    score_aligned,
    score_loss_weights,
    min_lr_ratio,
    loss_mode="global_l1",
    ground_truth_weight=0.5,
    teacher_weight=0.5,
    surface_loss_weight=0.25,
    upper_air_loss_weight=1.0,
    optimizer_param_groups=1,
    initialization_strategy="compatible_state",
):
    return {
        "version": 2,
        "student_profile": student_profile["name"],
        "total_epochs": int(total_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_steps": int(warmup_steps),
        "base_lr": float(base_lr),
        "score_aligned": bool(score_aligned),
        "score_loss_weights": [float(weight) for weight in score_loss_weights],
        "min_lr_ratio": float(min_lr_ratio),
        "loss_mode": str(loss_mode),
        "ground_truth_weight": float(ground_truth_weight),
        "teacher_weight": float(teacher_weight),
        "surface_loss_weight": float(surface_loss_weight),
        "upper_air_loss_weight": float(upper_air_loss_weight),
        "optimizer_param_groups": int(optimizer_param_groups),
        "initialization_strategy": str(initialization_strategy),
    }


def validate_training_protocol(checkpoint, expected, require=False):
    saved = checkpoint.get("training_protocol")
    if saved is None:
        if require:
            raise ValueError(
                "Resume checkpoint predates the fixed training protocol; "
                "use a new PANGU_UV_SCREEN_PREFIX and restart from the official teacher"
            )
        return False

    protocol_defaults = {
        "gradient_accumulation": 1,
        "attention_only_warmup_epochs": 0,
        "version": 2,
    }

    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = saved.get(key) if key in saved else protocol_defaults.get(key)
        if actual_value != expected_value:
            mismatches[key] = (actual_value, expected_value)

    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual!r}, requested={requested!r}"
            for key, (actual, requested) in sorted(mismatches.items())
        )
        raise ValueError(f"Resume training protocol mismatch: {details}")
    return True


class YearBlockSampler:
    """Shuffle ERA5 in contiguous per-year blocks to avoid random-file I/O."""

    def __init__(self, dataset, block_size=64, seed=20260711):
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive")
        self.dataset = dataset
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.epoch = 0
        if not hasattr(dataset, "year_offsets") or not hasattr(dataset, "sample_counts"):
            raise TypeError("YearBlockSampler requires ERA5 year_offsets/sample_counts")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        years = list(self.dataset.selected_years)
        blocks_by_year = {}
        for year_index, year in enumerate(years):
            start = int(self.dataset.year_offsets[year_index])
            count = int(self.dataset.sample_counts[year])
            year_blocks = [
                list(range(offset, min(offset + self.block_size, start + count)))
                for offset in range(start, start + count, self.block_size)
            ]
            rng.shuffle(year_blocks)
            blocks_by_year[year] = year_blocks
        active_years = list(years)
        while active_years:
            rng.shuffle(active_years)
            next_active = []
            for year in active_years:
                year_blocks = blocks_by_year[year]
                if year_blocks:
                    yield from year_blocks.pop()
                if year_blocks:
                    next_active.append(year)
            active_years = next_active


def _select_axis_by_magnitude(tensor, axis, count):
    if tensor.shape[axis] <= count:
        return tensor
    reduce_dims = tuple(index for index in range(tensor.ndim) if index != axis)
    if reduce_dims:
        # 多维张量：对非 axis 维度求和，得到每个 axis 切片的能量
        scores = tensor.float().square().sum(dim=reduce_dims)
    else:
        # 1D 张量：reduce_dims 为空，sum(dim=()) 在部分 PyTorch 版本会
        # 错误地规约为标量；直接用 element-wise square 作为 scores
        scores = tensor.float().square()
    indices = torch.topk(scores, k=int(count), largest=True, sorted=False).indices
    return tensor.index_select(axis, torch.sort(indices).values.to(tensor.device))


def magnitude_resize_tensor(source, target_shape, preserve_qkv=False):
    """Shrink an official tensor by retaining its highest-energy slices."""
    target_shape = tuple(int(value) for value in target_shape)
    if source.ndim != len(target_shape):
        raise ValueError("source and target tensor ranks must match")
    result = source
    for axis, target_size in enumerate(target_shape):
        source_size = result.shape[axis]
        if source_size == target_size:
            continue
        if source_size < target_size:
            raise ValueError(
                f"cannot expand axis {axis} from {source_size} to {target_size}"
            )
        if (
            preserve_qkv
            and axis == 0
            and source_size % 3 == 0
            and target_size % 3 == 0
        ):
            result = torch.cat(
                [
                    _select_axis_by_magnitude(group, axis, target_size // 3)
                    for group in torch.chunk(result, 3, dim=axis)
                ],
                dim=axis,
            )
        else:
            result = _select_axis_by_magnitude(result, axis, target_size)
    if tuple(result.shape) != target_shape:
        raise ValueError(f"mapped tensor shape {tuple(result.shape)} != {target_shape}")
    return result


def split_scored_channels(surface, upper_air):
    indices = torch.as_tensor(SCORED_UPPER_INDICES, device=upper_air.device)
    scored = torch.cat((surface, upper_air.index_select(1, indices)), dim=1)
    unscored_indices = torch.as_tensor(
        [index for index in range(upper_air.shape[1]) if index not in SCORED_UPPER_INDICES],
        device=upper_air.device,
    )
    return scored, upper_air.index_select(1, unscored_indices)


def _latitude_weights(reference):
    height = reference.shape[-2]
    latitudes = torch.linspace(-90.0, 90.0, height, device=reference.device)
    weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
    return weights.to(reference.dtype).view(1, 1, height, 1)


def latitude_weighted_rmse(prediction, target, channel_normalizers=None):
    weights = _latitude_weights(prediction)
    denominator = weights.sum() * prediction.shape[-1]
    channel_mse = ((prediction - target).square() * weights).sum(dim=(-2, -1))
    channel_rmse = torch.sqrt(channel_mse / denominator.clamp_min(1e-8) + 1e-12)
    if channel_normalizers is not None:
        normalizers = torch.as_tensor(
            channel_normalizers, device=prediction.device, dtype=prediction.dtype
        ).reshape(1, -1)
        if normalizers.shape[1] != prediction.shape[1]:
            raise ValueError("RMSE normalizer count must match scored channels")
        channel_rmse = channel_rmse / normalizers.clamp_min(1e-8)
    return channel_rmse.mean()


def latitude_weighted_acc_loss(prediction, target):
    weights = _latitude_weights(prediction)
    numerator = (prediction * target * weights).sum(dim=(-2, -1))
    pred_norm = (prediction.square() * weights).sum(dim=(-2, -1))
    target_norm = (target.square() * weights).sum(dim=(-2, -1))
    correlation = numerator / torch.sqrt(pred_norm * target_norm).clamp_min(1e-8)
    return (1.0 - correlation.clamp(-1.0, 1.0)).mean()


def latitude_weighted_anomaly_acc_loss(prediction, target):
    """Differentiable ACC loss after removing each weighted spatial mean."""
    weights = _latitude_weights(prediction)
    denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1e-8)
    prediction = prediction - (prediction * weights).sum(
        dim=(-2, -1), keepdim=True
    ) / denominator
    target = target - (target * weights).sum(dim=(-2, -1), keepdim=True) / denominator
    numerator = (prediction * target * weights).sum(dim=(-2, -1))
    prediction_norm = (prediction.square() * weights).sum(dim=(-2, -1))
    target_norm = (target.square() * weights).sum(dim=(-2, -1))
    correlation = numerator / torch.sqrt(prediction_norm * target_norm).clamp_min(1e-8)
    return (1.0 - correlation.clamp(-1.0, 1.0)).mean()


def kd_2d_score_loss(
    student,
    target,
    teacher,
    rmse_normalizers,
    weights=(0.55, 0.30, 0.10, 0.05),
):
    """Xiandao-focused 15-channel KD loss with a weak 54-channel guardrail."""
    student_scored, student_unscored = split_scored_channels(*student)
    target_scored, _ = split_scored_channels(*target)
    teacher_scored, teacher_unscored = split_scored_channels(*teacher)
    teacher_scored = teacher_scored.detach()
    teacher_unscored = teacher_unscored.detach()
    if len(weights) != 4 or any(float(weight) < 0.0 for weight in weights):
        raise ValueError("KD 2D weights must contain four non-negative values")
    normalizers = torch.as_tensor(rmse_normalizers)
    if normalizers.numel() != 15 or not torch.isfinite(normalizers).all() or (normalizers <= 0).any():
        raise ValueError("KD 2D requires exactly 15 finite positive baseline RMSE values")
    parts = {
        "rmse": latitude_weighted_rmse(student_scored, target_scored, rmse_normalizers),
        "acc": latitude_weighted_anomaly_acc_loss(student_scored, target_scored),
        "scored_teacher": F.mse_loss(student_scored, teacher_scored),
        "unscored_teacher": F.mse_loss(student_unscored, teacher_unscored),
    }
    total = sum(float(weight) * parts[name] for weight, name in zip(
        weights, ("rmse", "acc", "scored_teacher", "unscored_teacher")
    ))
    return total, parts


def score_aligned_loss(
    student,
    target,
    teacher,
    rmse_normalizers,
    weights=(0.50, 0.25, 0.20, 0.05),
):
    student_scored, student_unscored = split_scored_channels(*student)
    target_scored, _ = split_scored_channels(*target)
    teacher_scored, teacher_unscored = split_scored_channels(*teacher)

    rmse = latitude_weighted_rmse(
        student_scored, target_scored, rmse_normalizers
    )
    acc = latitude_weighted_acc_loss(student_scored, target_scored)
    scored_teacher = F.l1_loss(student_scored, teacher_scored)
    unscored_teacher = F.l1_loss(student_unscored, teacher_unscored)
    if len(weights) != 4 or any(float(weight) < 0.0 for weight in weights):
        raise ValueError("score-aligned weights must contain four non-negative values")
    rmse_weight, acc_weight, scored_teacher_weight, unscored_teacher_weight = (
        float(weight) for weight in weights
    )
    total = (
        rmse_weight * rmse
        + acc_weight * acc
        + scored_teacher_weight * scored_teacher
        + unscored_teacher_weight * unscored_teacher
    )
    return total, {
        "rmse": rmse,
        "acc": acc,
        "scored_teacher": scored_teacher,
        "unscored_teacher": unscored_teacher,
    }


def score_validation_loss(
    student,
    target,
    rmse_normalizers,
    weights=(0.50, 0.25),
):
    student_scored, _ = split_scored_channels(*student)
    target_scored, _ = split_scored_channels(*target)
    if len(weights) != 2 or any(float(weight) < 0.0 for weight in weights):
        raise ValueError("validation weights must contain two non-negative values")
    return float(weights[0]) * latitude_weighted_rmse(
        student_scored, target_scored, rmse_normalizers
    ) + float(weights[1]) * latitude_weighted_acc_loss(student_scored, target_scored)


def normalized_scored_rmse(baseline_rmse, channel_stds):
    baseline = torch.as_tensor(baseline_rmse, dtype=torch.float32).reshape(-1)
    stds = torch.as_tensor(channel_stds, dtype=torch.float32).reshape(-1)
    if stds.numel() != 69:
        raise ValueError("Expected 69 selected-channel standard deviations")
    indices = torch.as_tensor(SCORED_CHANNEL_INDICES)
    if baseline.numel() == 69:
        baseline = baseline.index_select(0, indices)
    elif baseline.numel() != 15:
        raise ValueError("Official baseline RMSE must contain 15 or 69 values")
    scored_stds = stds.index_select(0, indices)
    normalizers = baseline / scored_stds
    if not torch.isfinite(normalizers).all() or torch.any(normalizers <= 0):
        raise ValueError("Official baseline RMSE normalizers must be positive and finite")
    return normalizers


def load_sensitive_layer_names(path, count=5):
    with open(path, "r", encoding="utf-8") as handle:
        ranking = json.load(handle)
    names = [str(item["name"]) for item in ranking[: int(count)]]
    if len(names) != int(count):
        raise ValueError(f"Sensitivity ranking has fewer than {count} layers")
    return names


def configure_trainable_stage(model, stage, sensitive_layers=()):
    if stage not in {"head", "all"}:
        raise ValueError("score training stage must be 'head' or 'all'")
    if stage == "all":
        model.requires_grad_(True)
    else:
        sensitive = tuple(str(name) for name in sensitive_layers)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "patchrecovery" in name or any(
                layer in name for layer in sensitive
            )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters selected for score stage {stage}")
    return trainable


@torch.no_grad()
def project_quantized_linear_weights(model, fp16_layer_names):
    """Project non-sensitive Linear weights onto the accepted INT8 grid."""
    model = model.module if hasattr(model, "module") else model
    keep = set(str(name) for name in fp16_layer_names)
    projected = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name in keep:
            continue
        weight = module.weight
        flat = weight.reshape(weight.shape[0], -1)
        maximum = flat.abs().amax(dim=1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        flat.copy_(torch.clamp(torch.round(flat / scale), -128, 127) * scale)
        projected += 1
    return projected
