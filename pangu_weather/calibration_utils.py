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


SCORED_CHANNELS = (
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "geopotential_850",
    "geopotential_700",
    "geopotential_500",
    "specific_humidity_850",
    "specific_humidity_700",
    "specific_humidity_500",
    "temperature_850",
    "temperature_700",
    "temperature_500",
    "u_component_of_wind_500",
    "v_component_of_wind_500",
)


@dataclass(frozen=True)
class BlockedSlopeResult:
    candidate_coeffs: np.ndarray
    fold_reports: tuple[dict, ...]
    promotion_eligible: bool
    worst_relative_w: float
    mean_relative_w: float


def scored_channel_indices(channels) -> np.ndarray:
    """Return the official 15 scored channels in competition order."""
    positions = {name: idx for idx, name in enumerate(channels)}
    missing = [name for name in SCORED_CHANNELS if name not in positions]
    if missing:
        raise ValueError(f"Missing official scored channels: {missing}")
    return np.asarray([positions[name] for name in SCORED_CHANNELS], dtype=np.int64)


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


def _fit_candidate_slopes(
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
    climatology: np.ndarray,
    accepted_coeffs: np.ndarray,
    scored_indices: np.ndarray,
    coefficient_bounds: tuple[float, float],
) -> np.ndarray:
    pred_anom = baseline_predictions[:, scored_indices] - climatology[scored_indices]
    target_anom = targets[:, scored_indices] - climatology[scored_indices]
    axes = (0, 2, 3)
    adjustment = fit_anomaly_scale(
        np.sum(pred_anom * target_anom, axis=axes),
        np.sum(pred_anom * pred_anom, axis=axes),
        lower=0.0,
        upper=np.finfo(np.float32).max,
    )
    candidate = np.asarray(accepted_coeffs, dtype=np.float32).copy()
    candidate[scored_indices] = np.clip(
        candidate[scored_indices] * adjustment,
        coefficient_bounds[0],
        coefficient_bounds[1],
    )
    return candidate


def _apply_relative_slopes(
    baseline_predictions: np.ndarray,
    climatology: np.ndarray,
    accepted_coeffs: np.ndarray,
    candidate_coeffs: np.ndarray,
    scored_indices: np.ndarray,
) -> np.ndarray:
    accepted = np.asarray(accepted_coeffs, dtype=np.float64)
    if np.any(np.abs(accepted[scored_indices]) <= 1e-8):
        raise ValueError("Accepted scored-channel coefficients must be non-zero")
    adjustment = candidate_coeffs[scored_indices] / accepted[scored_indices]
    baseline_anom = baseline_predictions[:, scored_indices] - climatology[scored_indices]
    return climatology[scored_indices] + adjustment.reshape(1, -1, 1, 1) * baseline_anom


def _channel_metrics(predictions: np.ndarray, targets: np.ndarray, climatology: np.ndarray):
    errors = targets - predictions
    rmse = np.mean(np.sqrt(np.mean(errors * errors, axis=(2, 3))), axis=0)
    pred_anom = predictions - climatology
    target_anom = targets - climatology
    axes = (0, 2, 3)
    numerator = np.sum(pred_anom * target_anom, axis=axes)
    denominator = np.sqrt(
        np.sum(pred_anom * pred_anom, axis=axes)
        * np.sum(target_anom * target_anom, axis=axes)
    )
    acc = numerator / np.maximum(denominator, 1e-8)
    return rmse, acc


def _relative_w_proxy(
    baseline_rmse: np.ndarray,
    baseline_acc: np.ndarray,
    candidate_rmse: np.ndarray,
    candidate_acc: np.ndarray,
) -> float:
    """Return a baseline-relative W proxy; the accepted baseline equals 40."""
    epsilon = 1e-12
    rmse_ratio = baseline_rmse / np.maximum(np.abs(candidate_rmse), epsilon)
    rmse_ratio = np.where(
        (np.abs(baseline_rmse) <= epsilon) & (np.abs(candidate_rmse) <= epsilon),
        1.0,
        rmse_ratio,
    )
    acc_ratio = candidate_acc / np.maximum(np.abs(baseline_acc), epsilon)
    acc_ratio = np.where(
        (np.abs(baseline_acc) <= epsilon) & (np.abs(candidate_acc) <= epsilon),
        1.0,
        acc_ratio,
    )
    return float(20.0 * np.mean(rmse_ratio * rmse_ratio + acc_ratio * acc_ratio))


def blocked_slope_calibration(
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
    climatology: np.ndarray,
    accepted_coeffs: np.ndarray,
    scored_indices: np.ndarray,
    *,
    num_blocks: int = 4,
    coefficient_bounds: tuple[float, float] = (0.2, 2.0),
    minimum_block_gain: float = 0.001,
) -> BlockedSlopeResult:
    """Fit scored slopes and validate them on held-out contiguous sample blocks.

    ``baseline_predictions`` must already include ``accepted_coeffs``. Each
    fold refits on all other time blocks and evaluates only its held-out block.
    The final candidate is fit on all samples, while promotion eligibility is
    determined exclusively from the held-out fold reports.
    """
    predictions = np.asarray(baseline_predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    clim = np.asarray(climatology, dtype=np.float64)
    accepted = np.asarray(accepted_coeffs, dtype=np.float32).reshape(-1)
    scored = np.asarray(scored_indices, dtype=np.int64).reshape(-1)

    if predictions.shape != truth.shape or predictions.ndim != 4:
        raise ValueError("predictions and targets must share [sample, channel, height, width]")
    if clim.shape != predictions.shape[1:]:
        raise ValueError("climatology must have [channel, height, width] shape")
    if accepted.shape[0] != predictions.shape[1]:
        raise ValueError("accepted coefficient count must match prediction channels")
    if len(np.unique(scored)) != 15:
        raise ValueError("exactly 15 unique scored-channel indices are required")
    if predictions.shape[0] < 2:
        raise ValueError("at least two chronological samples are required")
    if not 2 <= int(num_blocks) <= predictions.shape[0]:
        raise ValueError("num_blocks must be between 2 and the sample count")

    folds = np.array_split(np.arange(predictions.shape[0]), int(num_blocks))
    reports = []
    for block_id, held_out in enumerate(folds):
        train = np.setdiff1d(np.arange(predictions.shape[0]), held_out)
        fold_coeffs = _fit_candidate_slopes(
            predictions[train], truth[train], clim, accepted, scored, coefficient_bounds
        )
        candidate_pred = _apply_relative_slopes(
            predictions[held_out], clim, accepted, fold_coeffs, scored
        )
        baseline_scored = predictions[held_out][:, scored]
        truth_scored = truth[held_out][:, scored]
        clim_scored = clim[scored]
        base_rmse, base_acc = _channel_metrics(baseline_scored, truth_scored, clim_scored)
        cand_rmse, cand_acc = _channel_metrics(candidate_pred, truth_scored, clim_scored)
        relative_w = _relative_w_proxy(base_rmse, base_acc, cand_rmse, cand_acc)
        reports.append(
            {
                "block": block_id,
                "held_out_indices": held_out.tolist(),
                "baseline_rmse_mean": float(np.mean(base_rmse)),
                "candidate_rmse_mean": float(np.mean(cand_rmse)),
                "baseline_acc_mean": float(np.mean(base_acc)),
                "candidate_acc_mean": float(np.mean(cand_acc)),
                "relative_w": relative_w,
                "relative_w_gain": relative_w - 40.0,
            }
        )

    candidate = _fit_candidate_slopes(
        predictions, truth, clim, accepted, scored, coefficient_bounds
    )
    unscored = np.setdiff1d(np.arange(accepted.shape[0]), scored)
    if not np.array_equal(candidate[unscored], accepted[unscored]):
        raise AssertionError("unscored calibration coefficients changed")

    scores = np.asarray([report["relative_w"] for report in reports])
    threshold = 40.0 + float(minimum_block_gain)
    return BlockedSlopeResult(
        candidate_coeffs=candidate,
        fold_reports=tuple(reports),
        promotion_eligible=bool(np.all(scores >= threshold)),
        worst_relative_w=float(np.min(scores)),
        mean_relative_w=float(np.mean(scores)),
    )


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
