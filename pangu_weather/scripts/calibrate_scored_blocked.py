#!/usr/bin/env python3
"""Non-destructive blocked calibration for the 15 official scored channels."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from calibration_utils import blocked_slope_calibration, scored_channel_indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit only the 15 scored slope coefficients with contiguous blocked "
            "cross-validation. The accepted calibration is never modified."
        )
    )
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--predictions", default="result/output")
    parser.add_argument("--accepted", default="data/checkpoints/calibration_coeffs.npy")
    parser.add_argument(
        "--candidate", default="result/calibration_candidates/scored_blocked.npy"
    )
    parser.add_argument(
        "--report", default="result/calibration_candidates/scored_blocked.json"
    )
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--lower", type=float, default=0.2)
    parser.add_argument("--upper", type=float, default=2.0)
    parser.add_argument("--minimum-block-gain", type=float, default=0.001)
    parser.add_argument(
        "--prediction-space",
        choices=("accepted-calibrated", "uncalibrated"),
        default="accepted-calibrated",
        help=(
            "Use accepted-calibrated for normal inference outputs. For raw "
            "outputs, the accepted slopes are applied before cross-validation."
        ),
    )
    return parser.parse_args()


def _load_data(config_path: str, predictions_dir: Path):
    import h5py
    from onescience.utils.YParams import YParams

    cfg_data = YParams(config_path, "datapipe")
    dataset = cfg_data.dataset
    channels = list(dataset.channels)
    data_dir = dataset.data_dir

    meta_path = Path(data_dir) / "metadata.json"
    if not meta_path.exists():
        meta_path = Path(
            "/public/home/xdzs2026_c271/xiandao2026-AI4S/"
            "onedatasets/ERA5_test/metadata.json"
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        variables = json.load(handle)["variables"]
    channel_indices = [variables.index(name) for name in channels]

    truth_paths = {}
    for year in dataset.test_ratio:
        pattern = os.path.join(data_dir, "data", str(year), "*.h5")
        for path in sorted(glob.glob(pattern)):
            truth_paths[Path(path).stem] = Path(path)

    filenames = sorted(path for path in predictions_dir.glob("*.npy") if path.stem in truth_paths)
    if len(filenames) < 2:
        raise RuntimeError("Need at least two prediction files matched to truth HDF5 files")

    predictions = []
    targets = []
    for prediction_path in filenames:
        pred = np.asarray(np.load(prediction_path)).squeeze()
        with h5py.File(truth_paths[prediction_path.stem], "r") as handle:
            target = np.asarray(handle["fields"][:]).squeeze()[channel_indices]
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {prediction_path.name}: {pred.shape} != {target.shape}"
            )
        predictions.append(pred)
        targets.append(target)

    means = np.load(Path(dataset.stats_dir) / "global_means.npy")
    climatology = np.asarray(means[0, channel_indices])
    return filenames, channels, np.stack(predictions), np.stack(targets), climatology


def main():
    args = _parse_args()
    accepted_path = Path(args.accepted).resolve()
    candidate_path = Path(args.candidate).resolve()
    report_path = Path(args.report).resolve()
    if accepted_path in {candidate_path, report_path}:
        raise ValueError("Candidate and report paths must not overwrite accepted calibration")
    if candidate_path == report_path:
        raise ValueError("Candidate and report paths must differ")
    if not accepted_path.is_file():
        raise FileNotFoundError(f"Accepted calibration not found: {accepted_path}")
    if args.lower <= 0 or args.upper <= args.lower:
        raise ValueError("Require 0 < lower < upper")

    accepted_hash_before = _sha256(accepted_path)
    accepted = np.asarray(np.load(accepted_path), dtype=np.float32).reshape(-1)
    files, channels, predictions, targets, climatology = _load_data(
        args.config, Path(args.predictions)
    )
    scored = scored_channel_indices(channels)

    if args.prediction_space == "uncalibrated":
        predictions = climatology + accepted.reshape(1, -1, 1, 1) * (
            predictions - climatology
        )

    result = blocked_slope_calibration(
        predictions,
        targets,
        climatology,
        accepted,
        scored,
        num_blocks=args.blocks,
        coefficient_bounds=(args.lower, args.upper),
        minimum_block_gain=args.minimum_block_gain,
    )

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(candidate_path, result.candidate_coeffs)
    accepted_hash_after = _sha256(accepted_path)
    if accepted_hash_before != accepted_hash_after:
        raise RuntimeError("Accepted calibration changed during evaluation")

    unscored = np.setdiff1d(np.arange(len(channels)), scored)
    report = {
        "status": "promotion-eligible" if result.promotion_eligible else "rejected",
        "prediction_space": args.prediction_space,
        "sample_count": len(files),
        "first_sample": files[0].stem,
        "last_sample": files[-1].stem,
        "blocks": args.blocks,
        "minimum_block_gain": args.minimum_block_gain,
        "worst_relative_w": result.worst_relative_w,
        "mean_relative_w": result.mean_relative_w,
        "accepted_path": str(accepted_path),
        "accepted_sha256": accepted_hash_after,
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "unscored_coefficients_exact": bool(
            np.array_equal(result.candidate_coeffs[unscored], accepted[unscored])
        ),
        "scored_channels": [
            {"index": int(index), "name": channels[index]} for index in scored
        ],
        "folds": list(result.fold_reports),
        "warning": (
            "The relative W proxy ranks calibration candidates but is not an "
            "official platform W score. Only a promotion-eligible candidate "
            "should proceed to a platform A/B."
        ),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
