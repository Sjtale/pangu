import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pangu_weather.blocked_w_validator import (
    CHECKPOINT_NAMES,
    SCORED_CHANNEL_INDICES,
    channel_metrics,
    main,
    official_min_clipped_w,
    separated_contiguous_blocks,
    validate_blocked_w,
    validate_blocked_w_metrics,
)


def synthetic_predictions():
    rng = np.random.default_rng(17)
    shape = (160, 69, 1, 2)
    climatology = rng.normal(size=(69, 1, 2)).astype(np.float32)
    anomalies = rng.normal(size=shape).astype(np.float32)
    targets = climatology[None] + anomalies
    error = rng.normal(size=shape).astype(np.float32)
    full192 = targets + 0.05 * error
    pruned96 = targets + 0.20 * error
    candidates = {
        "step1024": targets + 0.16 * error,
        "step2048": targets + 0.08 * error,
        "step3072": targets + 0.24 * error,
    }
    return candidates, pruned96, full192, targets, climatology


def precomputed_metrics(predictions, targets, climatology):
    rmse_blocks = []
    acc_blocks = []
    scored = np.asarray(SCORED_CHANNEL_INDICES)
    for start, stop in separated_contiguous_blocks(targets.shape[0]):
        rmse, acc = channel_metrics(
            predictions[start:stop, scored],
            targets[start:stop, scored],
            climatology[scored],
        )
        rmse_blocks.append(rmse)
        acc_blocks.append(acc)
    return np.stack(rmse_blocks), np.stack(acc_blocks)


class BlockedWValidatorTests(unittest.TestCase):
    def test_inferred_formula_min_clips_better_metrics(self):
        full_rmse = np.full(15, 2.0)
        full_acc = np.full(15, 0.8)

        self.assertAlmostEqual(
            official_min_clipped_w(
                full_rmse,
                full_acc,
                np.full(15, 1.0),
                np.full(15, 0.9),
            ),
            40.0,
        )
        self.assertAlmostEqual(
            official_min_clipped_w(
                full_rmse,
                full_acc,
                np.full(15, 4.0),
                np.full(15, 0.4),
            ),
            10.0,
        )

    def test_blocks_are_spread_contiguous_and_non_overlapping(self):
        blocks = separated_contiguous_blocks(160)

        self.assertEqual(blocks[0], (0, 32))
        self.assertEqual(blocks[-1], (128, 160))
        self.assertEqual(len(blocks), 4)
        self.assertTrue(all(stop - start == 32 for start, stop in blocks))
        self.assertTrue(
            all(blocks[index][1] <= blocks[index + 1][0] for index in range(3))
        )

    def test_selects_best_checkpoint_only_after_both_gates(self):
        candidates, pruned96, full192, targets, climatology = synthetic_predictions()

        report = validate_blocked_w(
            candidates, pruned96, full192, targets, climatology
        )

        self.assertEqual(report["scored_indices"], list(SCORED_CHANNEL_INDICES))
        self.assertEqual(report["selected_checkpoint"], "step2048")
        self.assertEqual(report["best_observed_checkpoint"], "step2048")
        self.assertTrue(report["candidates"]["step2048"]["passes"])
        self.assertGreaterEqual(report["candidates"]["step2048"]["mean_delta"], 0.15)
        self.assertGreaterEqual(
            report["candidates"]["step2048"]["worst_block_delta"], 0.03
        )
        self.assertFalse(report["candidates"]["step3072"]["passes"])
        self.assertEqual(set(report["candidates"]), set(CHECKPOINT_NAMES))

    def test_compact_metrics_match_direct_array_scoring(self):
        candidates, pruned96, full192, targets, climatology = synthetic_predictions()
        direct = validate_blocked_w(
            candidates, pruned96, full192, targets, climatology
        )
        compact = validate_blocked_w_metrics(
            {
                name: precomputed_metrics(value, targets, climatology)
                for name, value in candidates.items()
            },
            precomputed_metrics(pruned96, targets, climatology),
            precomputed_metrics(full192, targets, climatology),
        )

        self.assertEqual(compact["input_mode"], "metrics")
        self.assertEqual(compact["metric_shape"], [4, 15])
        self.assertEqual(compact["ranking"], direct["ranking"])
        self.assertEqual(
            compact["selected_checkpoint"], direct["selected_checkpoint"]
        )
        for name in CHECKPOINT_NAMES:
            self.assertAlmostEqual(
                compact["candidates"][name]["mean_delta"],
                direct["candidates"][name]["mean_delta"],
            )
            self.assertAlmostEqual(
                compact["candidates"][name]["worst_block_delta"],
                direct["candidates"][name]["worst_block_delta"],
            )

    def test_compact_metrics_require_four_by_fifteen_arrays(self):
        valid = (np.ones((4, 15)), np.ones((4, 15)))
        candidates = {name: valid for name in CHECKPOINT_NAMES}
        candidates["step1024"] = (np.ones((3, 15)), np.ones((4, 15)))

        with self.assertRaisesRegex(ValueError, "shape"):
            validate_blocked_w_metrics(candidates, valid, valid)

    def test_cli_returns_nonzero_when_no_checkpoint_clears_both_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = (np.ones((4, 15)), np.full((4, 15), 0.8))
            pruned = (np.full((4, 15), 2.0), np.full((4, 15), 0.4))

            def save_metrics(name, metrics):
                path = root / f"{name}.npz"
                np.savez(path, rmse=metrics[0], acc=metrics[1])
                return path

            full_path = save_metrics("full", full)
            pruned_path = save_metrics("pruned", pruned)
            candidate_paths = {
                name: save_metrics(name, pruned) for name in CHECKPOINT_NAMES
            }
            output = root / "rejected.json"
            code = main(
                [
                    "--metric-inputs",
                    "--full192",
                    str(full_path),
                    "--pruned96",
                    str(pruned_path),
                    "--step1024",
                    str(candidate_paths["step1024"]),
                    "--step2048",
                    str(candidate_paths["step2048"]),
                    "--step3072",
                    str(candidate_paths["step3072"]),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIsNone(
                json.loads(output.read_text(encoding="utf-8"))[
                    "selected_checkpoint"
                ]
            )

    def test_npz_cli_writes_same_selection_without_overwrite(self):
        candidates, pruned96, full192, targets, climatology = synthetic_predictions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def save(name, key, value):
                path = root / f"{name}.npz"
                np.savez(path, **{key: value})
                return path

            paths = {
                "targets": save("targets", "targets", targets),
                "climatology": save("climatology", "climatology", climatology),
                "full192": save("full192", "predictions", full192),
                "pruned96": save("pruned96", "predictions", pruned96),
                **{
                    name: save(name, "predictions", candidates[name])
                    for name in CHECKPOINT_NAMES
                },
            }
            output = root / "report.json"
            argv = [
                "--targets",
                str(paths["targets"]),
                "--climatology",
                str(paths["climatology"]),
                "--full192",
                str(paths["full192"]),
                "--pruned96",
                str(paths["pruned96"]),
                "--step1024",
                str(paths["step1024"]),
                "--step2048",
                str(paths["step2048"]),
                "--step3072",
                str(paths["step3072"]),
                "--output",
                str(output),
            ]

            self.assertEqual(main(argv), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["selected_checkpoint"], "step2048")
            with self.assertRaises(FileExistsError):
                main(argv)

    def test_compact_metric_npz_cli_needs_no_full_prediction_arrays(self):
        candidates, pruned96, full192, targets, climatology = synthetic_predictions()
        candidate_metrics = {
            name: precomputed_metrics(value, targets, climatology)
            for name, value in candidates.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def save_metrics(name, metrics):
                path = root / f"{name}.npz"
                np.savez(path, rmse=metrics[0], acc=metrics[1])
                return path

            paths = {
                "full192": save_metrics(
                    "full192", precomputed_metrics(full192, targets, climatology)
                ),
                "pruned96": save_metrics(
                    "pruned96", precomputed_metrics(pruned96, targets, climatology)
                ),
                **{
                    name: save_metrics(name, candidate_metrics[name])
                    for name in CHECKPOINT_NAMES
                },
            }
            output = root / "metric_report.json"
            argv = [
                "--metric-inputs",
                "--full192",
                str(paths["full192"]),
                "--pruned96",
                str(paths["pruned96"]),
                "--step1024",
                str(paths["step1024"]),
                "--step2048",
                str(paths["step2048"]),
                "--step3072",
                str(paths["step3072"]),
                "--output",
                str(output),
            ]

            self.assertEqual(main(argv), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["input_mode"], "metrics")
            self.assertEqual(report["metric_shape"], [4, 15])
            self.assertEqual(report["selected_checkpoint"], "step2048")


if __name__ == "__main__":
    unittest.main()
