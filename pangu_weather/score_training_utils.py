"""Loss and parameter-freezing helpers for scored-channel recovery training."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F


SCORED_UPPER_INDICES = (2, 3, 5, 15, 16, 18, 28, 29, 31, 44, 57)
SCORED_CHANNEL_INDICES = (0, 1, 2, 3, 6, 7, 9, 19, 20, 22, 32, 33, 35, 48, 61)


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


def score_aligned_loss(student, target, teacher, rmse_normalizers):
    student_scored, student_unscored = split_scored_channels(*student)
    target_scored, _ = split_scored_channels(*target)
    teacher_scored, teacher_unscored = split_scored_channels(*teacher)

    rmse = latitude_weighted_rmse(
        student_scored, target_scored, rmse_normalizers
    )
    acc = latitude_weighted_acc_loss(student_scored, target_scored)
    scored_teacher = F.l1_loss(student_scored, teacher_scored)
    unscored_teacher = F.l1_loss(student_unscored, teacher_unscored)
    total = 0.50 * rmse + 0.25 * acc + 0.20 * scored_teacher + 0.05 * unscored_teacher
    return total, {
        "rmse": rmse,
        "acc": acc,
        "scored_teacher": scored_teacher,
        "unscored_teacher": unscored_teacher,
    }


def score_validation_loss(student, target, rmse_normalizers):
    student_scored, _ = split_scored_channels(*student)
    target_scored, _ = split_scored_channels(*target)
    return 0.50 * latitude_weighted_rmse(
        student_scored, target_scored, rmse_normalizers
    ) + 0.25 * latitude_weighted_acc_loss(student_scored, target_scored)


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
