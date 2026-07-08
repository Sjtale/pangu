"""Output calibration helpers for validation-fitted weather fields."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineCalibration:
    scale: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class GlobalMeanCorrection:
    target_mean: np.ndarray
    channel_mask: np.ndarray


def fit_anomaly_scale(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    lower: float = 0.2,
    upper: float = 2.0,
) -> np.ndarray:
    coeffs = np.ones_like(numerator, dtype=np.float32)
    valid = denominator > 1e-8
    coeffs[valid] = numerator[valid] / denominator[valid]
    return np.clip(coeffs, lower, upper).astype(np.float32)


def fit_affine_from_sums(
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_xx: np.ndarray,
    sum_xy: np.ndarray,
    count: int,
    channel_stds: np.ndarray,
    *,
    scale_bounds: tuple[float, float] = (0.5, 1.5),
    bias_std_clip: float = 0.25,
) -> AffineCalibration:
    scale = np.ones_like(sum_x, dtype=np.float64)
    bias = np.zeros_like(sum_x, dtype=np.float64)
    if count <= 0:
        return AffineCalibration(scale.astype(np.float32), bias.astype(np.float32))

    denom = sum_xx - (sum_x * sum_x / count)
    numer = sum_xy - (sum_x * sum_y / count)
    valid = np.abs(denom) > 1e-8
    scale[valid] = numer[valid] / denom[valid]
    scale = np.clip(scale, scale_bounds[0], scale_bounds[1])
    bias = (sum_y - scale * sum_x) / count

    std = np.asarray(channel_stds, dtype=np.float64).reshape(-1)
    if std.shape[0] == bias.shape[0] and bias_std_clip >= 0:
        limit = np.maximum(std * bias_std_clip, 1e-8)
        bias = np.clip(bias, -limit, limit)

    return AffineCalibration(scale.astype(np.float32), bias.astype(np.float32))


def save_affine_calibration(path: str, affine: AffineCalibration) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, scale=affine.scale.astype(np.float32), bias=affine.bias.astype(np.float32))


def load_affine_calibration(path: str, num_channels: int) -> AffineCalibration:
    data = np.load(path)
    scale = np.asarray(data["scale"], dtype=np.float32).reshape(num_channels)
    bias = np.asarray(data["bias"], dtype=np.float32).reshape(num_channels)
    return AffineCalibration(scale=scale, bias=bias)


def apply_affine_calibration(
    pred: np.ndarray,
    climatology: np.ndarray,
    affine: AffineCalibration | None,
) -> np.ndarray:
    if affine is None:
        return pred
    scale = affine.scale.reshape(1, -1, 1, 1)
    bias = affine.bias.reshape(1, -1, 1, 1)
    return climatology + scale * (pred - climatology) + bias


def latitude_weights(height: int) -> np.ndarray:
    latitudes = np.linspace(-90.0, 90.0, int(height), dtype=np.float64)
    weights = np.cos(np.deg2rad(latitudes))
    return np.maximum(weights, 0.0).astype(np.float32)


def weighted_channel_mean(fields: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if fields.ndim == 4:
        fields = fields[0]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("latitude weights must have positive sum")
    return np.sum(fields * weights.reshape(1, -1, 1), axis=(1, 2)) / (
        weight_sum * fields.shape[-1]
    )


def save_global_mean_correction(path: str, correction: GlobalMeanCorrection) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        target_mean=correction.target_mean.astype(np.float32),
        channel_mask=correction.channel_mask.astype(bool),
    )


def load_global_mean_correction(path: str, num_channels: int) -> GlobalMeanCorrection:
    data = np.load(path)
    target_mean = np.asarray(data["target_mean"], dtype=np.float32).reshape(num_channels)
    channel_mask = np.asarray(data["channel_mask"], dtype=bool).reshape(num_channels)
    return GlobalMeanCorrection(target_mean=target_mean, channel_mask=channel_mask)


def apply_global_mean_correction(
    pred: np.ndarray,
    correction: GlobalMeanCorrection | None,
) -> np.ndarray:
    if correction is None:
        return pred
    weights = latitude_weights(pred.shape[-2])
    current_mean = weighted_channel_mean(pred, weights)
    delta = correction.target_mean - current_mean
    delta = np.where(correction.channel_mask, delta, 0.0).reshape(1, -1, 1, 1)
    return pred + delta
