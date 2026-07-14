"""Pure-NumPy blocked validator for the inferred official W score.

The validator is evaluation-only: it compares saved predictions with targets
and never fits or modifies model parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCORED_CHANNEL_INDICES = (0, 1, 2, 3, 6, 7, 9, 19, 20, 22, 32, 33, 35, 48, 61)
CHECKPOINT_NAMES = ("step1024", "step2048", "step3072")
BLOCK_COUNT = 4
BLOCK_SIZE = 32
MEAN_DELTA_GATE = 0.15
WORST_BLOCK_DELTA_GATE = 0.03


def separated_contiguous_blocks(
    sample_count: int,
    *,
    block_starts=None,
    block_count: int = BLOCK_COUNT,
    block_size: int = BLOCK_SIZE,
) -> tuple[tuple[int, int], ...]:
    """Choose deterministic, non-overlapping contiguous blocks across time."""

    required = int(block_count) * int(block_size)
    if sample_count < required:
        raise ValueError(
            f"at least {required} chronological samples are required, got {sample_count}"
        )
    if block_starts is None:
        starts = np.rint(
            np.linspace(0, sample_count - block_size, block_count)
        ).astype(int)
    else:
        starts = np.asarray(block_starts, dtype=int).reshape(-1)
        if starts.size != block_count:
            raise ValueError(f"exactly {block_count} block starts are required")

    if np.any(starts < 0) or np.any(starts + block_size > sample_count):
        raise ValueError("block starts must keep every block inside the sample range")
    if np.any(starts[1:] < starts[:-1] + block_size):
        raise ValueError("blocks must be ordered and non-overlapping")
    return tuple((int(start), int(start + block_size)) for start in starts)


def channel_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    climatology: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute result.py-compatible per-channel RMSE and anomaly ACC."""

    pred = np.asarray(predictions)
    truth = np.asarray(targets)
    clim = np.asarray(climatology)
    if pred.shape != truth.shape or pred.ndim != 4:
        raise ValueError("predictions and targets must share [sample,channel,H,W]")
    if clim.shape != pred.shape[1:]:
        raise ValueError("climatology must have shape [channel,H,W]")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and targets must be finite")
    if not np.isfinite(clim).all():
        raise ValueError("climatology must be finite")

    error = truth - pred
    per_sample_mse = np.mean(error * error, axis=(2, 3), dtype=np.float64)
    rmse = np.mean(np.sqrt(per_sample_mse), axis=0, dtype=np.float64)

    pred_anomaly = pred - clim
    truth_anomaly = truth - clim
    axes = (0, 2, 3)
    numerator = np.sum(
        pred_anomaly * truth_anomaly, axis=axes, dtype=np.float64
    )
    pred_square = np.sum(pred_anomaly * pred_anomaly, axis=axes, dtype=np.float64)
    truth_square = np.sum(
        truth_anomaly * truth_anomaly, axis=axes, dtype=np.float64
    )
    acc = numerator / (np.sqrt(pred_square * truth_square) + 1.0e-8)
    return rmse, acc


def official_min_clipped_w(
    full192_rmse: np.ndarray,
    full192_acc: np.ndarray,
    model_rmse: np.ndarray,
    model_acc: np.ndarray,
) -> float:
    """Apply the empirically inferred official min-clipped W formula."""

    full_rmse = np.asarray(full192_rmse, dtype=np.float64).reshape(-1)
    full_acc = np.asarray(full192_acc, dtype=np.float64).reshape(-1)
    candidate_rmse = np.asarray(model_rmse, dtype=np.float64).reshape(-1)
    candidate_acc = np.asarray(model_acc, dtype=np.float64).reshape(-1)
    if not (
        full_rmse.size
        == full_acc.size
        == candidate_rmse.size
        == candidate_acc.size
        == 15
    ):
        raise ValueError("W scoring requires exactly 15 RMSE and ACC values")

    rmse_ratio = np.divide(
        full_rmse,
        candidate_rmse,
        out=np.ones_like(full_rmse),
        where=np.abs(candidate_rmse) > 1.0e-12,
    )
    acc_ratio = np.divide(
        candidate_acc,
        full_acc,
        out=np.ones_like(full_acc),
        where=np.abs(full_acc) > 1.0e-12,
    )
    rmse_term = np.minimum(rmse_ratio * rmse_ratio, 1.0)
    acc_term = np.minimum(acc_ratio * acc_ratio, 1.0)
    return float(20.0 * np.mean(rmse_term + acc_term))


def _normalize_climatology(climatology: np.ndarray, channels: int) -> np.ndarray:
    clim = np.asarray(climatology)
    if clim.ndim == 4 and clim.shape[0] == 1:
        clim = clim[0]
    if clim.ndim != 3 or clim.shape[0] != channels:
        raise ValueError("climatology must have shape [C,H,W] or [1,C,H,W]")
    return clim


def _validate_prediction_shapes(arrays: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
    shapes = {name: tuple(np.asarray(value).shape) for name, value in arrays.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"all predictions and targets must share one shape: {shapes}")
    shape = unique.pop()
    if len(shape) != 4:
        raise ValueError("all predictions and targets must have shape [sample,C,H,W]")
    return shape


def _normalize_scored_indices(scored_indices, channels=None) -> np.ndarray:
    scored = np.asarray(scored_indices, dtype=int).reshape(-1)
    if scored.size != 15 or np.unique(scored).size != 15:
        raise ValueError("exactly 15 unique scored indices are required")
    if np.any(scored < 0) or (
        channels is not None and np.any(scored >= int(channels))
    ):
        raise ValueError("scored indices are outside the channel range")
    return scored


def _validate_metric_pair(metrics, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        rmse, acc = metrics
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} metrics must be an (rmse, acc) pair") from error
    rmse = np.asarray(rmse, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    expected_shape = (BLOCK_COUNT, len(SCORED_CHANNEL_INDICES))
    if rmse.shape != expected_shape or acc.shape != expected_shape:
        raise ValueError(
            f"{name} rmse and acc must both have shape {expected_shape}"
        )
    if not np.isfinite(rmse).all() or not np.isfinite(acc).all():
        raise ValueError(f"{name} rmse and acc must be finite")
    if np.any(rmse < 0.0):
        raise ValueError(f"{name} rmse must be non-negative")
    return rmse, acc


def _score_metric_blocks(
    candidate_metrics,
    pruned96_metrics,
    full192_metrics,
    *,
    scored_indices,
    block_metadata,
    input_mode: str,
    mean_delta_gate: float,
    worst_block_delta_gate: float,
) -> dict:
    if set(candidate_metrics) != set(CHECKPOINT_NAMES):
        raise ValueError(
            "candidates must contain exactly step1024, step2048, and step3072"
        )
    scored = _normalize_scored_indices(scored_indices)
    blocks = [dict(item) for item in block_metadata]
    if len(blocks) != BLOCK_COUNT:
        raise ValueError(f"exactly {BLOCK_COUNT} block descriptions are required")

    full_rmse, full_acc = _validate_metric_pair(full192_metrics, "full192")
    pruned_rmse, pruned_acc = _validate_metric_pair(
        pruned96_metrics, "pruned96"
    )
    normalized_candidates = {
        name: _validate_metric_pair(candidate_metrics[name], name)
        for name in CHECKPOINT_NAMES
    }

    pruned_w = np.empty(BLOCK_COUNT, dtype=np.float64)
    full_w = np.empty(BLOCK_COUNT, dtype=np.float64)
    for block in range(BLOCK_COUNT):
        full_pair = (full_rmse[block], full_acc[block])
        full_w[block] = official_min_clipped_w(*full_pair, *full_pair)
        pruned_w[block] = official_min_clipped_w(
            *full_pair, pruned_rmse[block], pruned_acc[block]
        )

    reports = {}
    for name in CHECKPOINT_NAMES:
        candidate_rmse, candidate_acc = normalized_candidates[name]
        block_reports = []
        for block, metadata in enumerate(blocks):
            candidate_w = official_min_clipped_w(
                full_rmse[block],
                full_acc[block],
                candidate_rmse[block],
                candidate_acc[block],
            )
            delta = candidate_w - pruned_w[block]
            block_reports.append(
                {
                    **metadata,
                    "full192_w": float(full_w[block]),
                    "pruned96_w": float(pruned_w[block]),
                    "candidate_w": candidate_w,
                    "delta_vs_pruned96": float(delta),
                }
            )
        deltas = np.asarray(
            [item["delta_vs_pruned96"] for item in block_reports],
            dtype=np.float64,
        )
        mean_delta = float(np.mean(deltas))
        worst_delta = float(np.min(deltas))
        mean_pass = mean_delta >= float(mean_delta_gate)
        worst_pass = worst_delta >= float(worst_block_delta_gate)
        reports[name] = {
            "blocks": block_reports,
            "mean_delta": mean_delta,
            "worst_block_delta": worst_delta,
            "mean_gate_pass": bool(mean_pass),
            "worst_block_gate_pass": bool(worst_pass),
            "passes": bool(mean_pass and worst_pass),
        }

    order = {name: index for index, name in enumerate(CHECKPOINT_NAMES)}
    best_observed = max(
        CHECKPOINT_NAMES,
        key=lambda name: (
            reports[name]["mean_delta"],
            reports[name]["worst_block_delta"],
            -order[name],
        ),
    )
    eligible = [name for name in CHECKPOINT_NAMES if reports[name]["passes"]]
    selected = (
        max(
            eligible,
            key=lambda name: (
                reports[name]["mean_delta"],
                reports[name]["worst_block_delta"],
                -order[name],
            ),
        )
        if eligible
        else None
    )
    ranking = sorted(
        CHECKPOINT_NAMES,
        key=lambda name: (
            not reports[name]["passes"],
            -reports[name]["mean_delta"],
            -reports[name]["worst_block_delta"],
            order[name],
        ),
    )
    return {
        "input_mode": input_mode,
        "formula": "20/15 * sum(min((rmse_full/rmse_model)^2,1) + min((acc_model/acc_full)^2,1))",
        "scored_indices": scored.tolist(),
        "metric_shape": [BLOCK_COUNT, len(SCORED_CHANNEL_INDICES)],
        "blocks": blocks,
        "gates": {
            "mean_delta_min": float(mean_delta_gate),
            "worst_block_delta_min": float(worst_block_delta_gate),
        },
        "candidates": reports,
        "ranking": ranking,
        "best_observed_checkpoint": best_observed,
        "selected_checkpoint": selected,
    }


def validate_blocked_w_metrics(
    candidate_metrics,
    pruned96_metrics,
    full192_metrics,
    *,
    scored_indices=SCORED_CHANNEL_INDICES,
    mean_delta_gate: float = MEAN_DELTA_GATE,
    worst_block_delta_gate: float = WORST_BLOCK_DELTA_GATE,
) -> dict:
    """Validate precomputed ``(rmse, acc)`` arrays shaped ``[4,15]``."""

    blocks = [{"block": block} for block in range(BLOCK_COUNT)]
    return _score_metric_blocks(
        candidate_metrics,
        pruned96_metrics,
        full192_metrics,
        scored_indices=scored_indices,
        block_metadata=blocks,
        input_mode="metrics",
        mean_delta_gate=mean_delta_gate,
        worst_block_delta_gate=worst_block_delta_gate,
    )


def validate_blocked_w(
    candidates: dict[str, np.ndarray],
    pruned96: np.ndarray,
    full192: np.ndarray,
    targets: np.ndarray,
    climatology: np.ndarray,
    *,
    scored_indices=SCORED_CHANNEL_INDICES,
    block_starts=None,
    mean_delta_gate: float = MEAN_DELTA_GATE,
    worst_block_delta_gate: float = WORST_BLOCK_DELTA_GATE,
) -> dict:
    """Compare three checkpoints with pruned96 over four held-out time blocks."""

    if set(candidates) != set(CHECKPOINT_NAMES):
        raise ValueError(
            "candidates must contain exactly step1024, step2048, and step3072"
        )
    all_arrays = {
        "targets": np.asarray(targets),
        "full192": np.asarray(full192),
        "pruned96": np.asarray(pruned96),
        **{name: np.asarray(candidates[name]) for name in CHECKPOINT_NAMES},
    }
    sample_count, channels, height, width = _validate_prediction_shapes(all_arrays)
    scored = _normalize_scored_indices(scored_indices, channels)
    clim = _normalize_climatology(climatology, channels)
    if tuple(clim.shape[1:]) != (height, width):
        raise ValueError("climatology spatial shape does not match predictions")

    blocks = separated_contiguous_blocks(
        sample_count, block_starts=block_starts
    )
    model_metrics = {
        name: ([], [])
        for name in ("full192", "pruned96", *CHECKPOINT_NAMES)
    }
    for start, stop in blocks:
        truth_block = all_arrays["targets"][start:stop, scored]
        climate_scored = clim[scored]
        for name in model_metrics:
            rmse, acc = channel_metrics(
                all_arrays[name][start:stop, scored],
                truth_block,
                climate_scored,
            )
            model_metrics[name][0].append(rmse)
            model_metrics[name][1].append(acc)

    stacked_metrics = {
        name: (np.stack(rmse), np.stack(acc))
        for name, (rmse, acc) in model_metrics.items()
    }
    report = _score_metric_blocks(
        {name: stacked_metrics[name] for name in CHECKPOINT_NAMES},
        stacked_metrics["pruned96"],
        stacked_metrics["full192"],
        scored_indices=scored,
        block_metadata=[
            {"block": index, "start": start, "stop": stop}
            for index, (start, stop) in enumerate(blocks)
        ],
        input_mode="predictions",
        mean_delta_gate=mean_delta_gate,
        worst_block_delta_gate=worst_block_delta_gate,
    )
    report["sample_shape"] = [sample_count, channels, height, width]
    return report


def load_npz_array(path, key: str) -> np.ndarray:
    """Load one named array, allowing a single-array NPZ as a convenience."""

    with np.load(path, allow_pickle=False) as data:
        if key in data.files:
            return np.asarray(data[key])
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]])
        raise KeyError(f"{path} does not contain key {key!r}; keys={data.files}")


def load_npz_metrics(
    path, rmse_key: str = "rmse", acc_key: str = "acc"
) -> tuple[np.ndarray, np.ndarray]:
    """Load one compact ``[4,15]`` RMSE/ACC pair from an NPZ file."""

    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in (rmse_key, acc_key) if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing metric keys {missing}; keys={data.files}")
        return np.asarray(data[rmse_key]), np.asarray(data[acc_key])


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare step1024/2048/3072 with pruned96 over four W blocks. "
            "Default mode reads chronological prediction arrays; "
            "--metric-inputs reads tiny NPZ files whose rmse and acc arrays "
            "share the same four block rows and 15 official channel columns."
        )
    )
    parser.add_argument(
        "--metric-inputs",
        action="store_true",
        help=(
            "read compact rmse/acc [4,15] NPZ files; targets and climatology "
            "are not needed"
        ),
    )
    parser.add_argument("--targets", help="prediction mode: NPZ with key 'targets'")
    parser.add_argument(
        "--climatology", help="prediction mode: NPZ with key 'climatology'"
    )
    parser.add_argument("--full192", required=True, help="full192 NPZ")
    parser.add_argument("--pruned96", required=True, help="pruned96 NPZ")
    parser.add_argument("--step1024", required=True, help="step1024 NPZ")
    parser.add_argument("--step2048", required=True, help="step2048 NPZ")
    parser.add_argument("--step3072", required=True, help="step3072 NPZ")
    parser.add_argument("--prediction-key", default="predictions")
    parser.add_argument("--targets-key", default="targets")
    parser.add_argument("--climatology-key", default="climatology")
    parser.add_argument("--rmse-key", default="rmse")
    parser.add_argument("--acc-key", default="acc")
    parser.add_argument(
        "--scored-indices",
        default=",".join(str(index) for index in SCORED_CHANNEL_INDICES),
        help="exactly 15 comma-separated channel indices",
    )
    parser.add_argument(
        "--block-starts",
        help="optional four comma-separated starts; default spans the timeline",
    )
    parser.add_argument("--output", help="write JSON report; otherwise print stdout")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    scored_indices = _parse_int_list(args.scored_indices)
    if args.metric_inputs:
        if args.targets or args.climatology or args.block_starts:
            parser.error(
                "--metric-inputs cannot be combined with --targets, "
                "--climatology, or --block-starts"
            )
        metric_keys = (args.rmse_key, args.acc_key)
        report = validate_blocked_w_metrics(
            {
                "step1024": load_npz_metrics(args.step1024, *metric_keys),
                "step2048": load_npz_metrics(args.step2048, *metric_keys),
                "step3072": load_npz_metrics(args.step3072, *metric_keys),
            },
            load_npz_metrics(args.pruned96, *metric_keys),
            load_npz_metrics(args.full192, *metric_keys),
            scored_indices=scored_indices,
        )
    else:
        if not args.targets or not args.climatology:
            parser.error(
                "prediction mode requires both --targets and --climatology; "
                "use --metric-inputs for compact rmse/acc NPZ files"
            )
        prediction_key = args.prediction_key
        report = validate_blocked_w(
            {
                "step1024": load_npz_array(args.step1024, prediction_key),
                "step2048": load_npz_array(args.step2048, prediction_key),
                "step3072": load_npz_array(args.step3072, prediction_key),
            },
            load_npz_array(args.pruned96, prediction_key),
            load_npz_array(args.full192, prediction_key),
            load_npz_array(args.targets, args.targets_key),
            load_npz_array(args.climatology, args.climatology_key),
            scored_indices=scored_indices,
            block_starts=(
                _parse_int_list(args.block_starts) if args.block_starts else None
            ),
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite report: {output}")
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report["selected_checkpoint"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
