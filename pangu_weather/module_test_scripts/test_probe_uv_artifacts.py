"""Focused artifact and A/B tests for the U/V runtime probe."""

import importlib.util
import os
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_uv_runtime_sweep.py"
SPEC = importlib.util.spec_from_file_location("probe_uv_runtime_sweep", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProbeUvArtifactTests(unittest.TestCase):
    def test_official_timer_hash_is_frozen(self):
        inference = SCRIPT.parents[1] / "inference.py"
        self.assertEqual(
            PROBE.official_timer_block_sha256(inference),
            PROBE.FROZEN_OFFICIAL_TIMER_SHA256,
        )

    def test_new_p2_presets_are_explicit_isolated_interleaved_pairs(self):
        region = list(PROBE.iter_candidates("p2-region-release"))
        self.assertEqual([item["kind"] for item in region], ["baseline", "p2-region-release"])
        self.assertEqual(
            [item["env"]["PANGU_P2_REGION_RELEASE"] for item in region],
            ["0", "1"],
        )
        self.assertNotIn("PANGU_P2_PRECOMPUTE_REGION_IDS", PROBE.BASE_ENV)
        self.assertNotIn("PANGU_P2_RELEASE_ORIGINAL_MASKS", PROBE.BASE_ENV)
        for item in region:
            self.assertEqual(item["env"]["PANGU_P2_TILED_ATTENTION"], "1")
            self.assertEqual(item["env"]["PANGU_P2_RELEASE_ORIGINAL_BIAS"], "1")
            self.assertEqual(
                item["env"]["PANGU_TILED_HIP_EXTRA_FLAGS"],
                PROBE.P2_PLATFORM_KERNEL_FLAGS,
            )

        prebuild = list(PROBE.iter_candidates("p2-hip-prebuild"))
        self.assertEqual([item["kind"] for item in prebuild], ["baseline", "p2-hip-prebuild"])
        self.assertEqual(
            [item["env"]["PANGU_P2_PREBUILD_HIP"] for item in prebuild],
            ["0", "1"],
        )
        self.assertTrue(all(item["fresh_hip_build"] for item in prebuild))
        self.assertTrue(PROBE.should_run_interleaved("p2-region-release"))
        self.assertTrue(PROBE.should_run_interleaved("p2-hip-prebuild"))
        self.assertTrue(PROBE.should_run_interleaved("baseline", "candidate.pth"))
        self.assertFalse(PROBE.should_run_interleaved("baseline"))

        stacked = PROBE.checkpoint_ab_candidates(
            region,
            "baseline.pth",
            "candidate.pth",
        )
        self.assertEqual([item["kind"] for item in stacked], ["baseline", "checkpoint_candidate"])
        self.assertTrue(
            all(item["env"]["PANGU_P2_REGION_RELEASE"] == "1" for item in stacked)
        )
        self.assertEqual(
            [item["env"]["PANGU_FP16_CHECKPOINT"] for item in stacked],
            ["baseline.pth", "candidate.pth"],
        )

    def test_stdout_parser_retains_region_and_prebuild_json(self):
        stdout = "\n".join(
            [
                "[MEM] sample allocated=10.0 MB, reserved=512.0 MB, peak=20.0 MB",
                "[P2_REGION_SETUP] {\"net_unique_bytes_reclaimed\": 50200000}",
                "⚡ PANGU_P2_PREBUILD_HIP=1, prepared HIP library: "
                "{\"fingerprint\": \"abc\", \"occupancy\": {\"active_blocks_per_multiprocessor\": 8}}",
                "Max VRAM: 483.2 MB",
                "Current VRAM: 40.4 MB",
            ]
        )
        parsed = PROBE.parse_stdout(stdout)
        self.assertEqual(parsed["max_vram_mb"], 483.2)
        self.assertEqual(parsed["current_vram_mb"], 40.4)
        self.assertEqual(parsed["reserved_mb"], 512.0)
        self.assertEqual(
            parsed["p2_region_setup"]["net_unique_bytes_reclaimed"], 50200000
        )
        self.assertEqual(parsed["hip_prebuild_report"]["fingerprint"], "abc")
        self.assertEqual(parsed["report_parse_errors"], [])

    def test_exact_output_checks_cover_set_shape_dtype_values_sha_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            values = np.arange(6, dtype=np.float32).reshape(2, 3)
            np.save(baseline / "a.npy", values)
            np.save(candidate / "a.npy", values)

            exact = PROBE.compare_outputs(candidate, baseline)
            self.assertTrue(exact["output_exact"])
            self.assertTrue(exact["output_set_equal"])
            self.assertTrue(exact["output_shapes_equal"])
            self.assertTrue(exact["output_dtypes_equal"])
            self.assertTrue(exact["output_array_equal"])
            self.assertTrue(exact["output_sha256_equal"])
            self.assertTrue(exact["output_all_finite"])

            np.save(candidate / "a.npy", np.asfortranarray(values))
            layout = PROBE.compare_outputs(candidate, baseline)
            self.assertTrue(layout["output_array_equal"])
            self.assertFalse(layout["output_sha256_equal"])
            self.assertFalse(layout["output_exact"])

            np.save(candidate / "a.npy", values.astype(np.float64))
            dtype = PROBE.compare_outputs(candidate, baseline)
            self.assertFalse(dtype["output_dtypes_equal"])
            self.assertFalse(dtype["output_exact"])

            np.save(candidate / "a.npy", values[:, :2])
            shape = PROBE.compare_outputs(candidate, baseline)
            self.assertFalse(shape["output_shapes_equal"])

            np.save(candidate / "a.npy", values + 1)
            np.save(candidate / "extra.npy", values)
            output_set = PROBE.compare_outputs(candidate, baseline)
            self.assertFalse(output_set["output_set_equal"])
            self.assertEqual(output_set["output_extra_files"], ["extra.npy"])
            self.assertFalse(output_set["output_array_equal"])

            (candidate / "extra.npy").unlink()
            with_nan = values.copy()
            with_nan[0, 0] = np.nan
            np.save(baseline / "a.npy", with_nan)
            np.save(candidate / "a.npy", with_nan)
            nonfinite = PROBE.compare_outputs(candidate, baseline)
            self.assertEqual(nonfinite["output_nan_count"], 1)
            self.assertFalse(nonfinite["output_all_finite"])
            self.assertFalse(nonfinite["output_array_equal"])
            self.assertFalse(nonfinite["output_exact"])

    def test_fresh_build_dirs_are_unique_and_only_used_by_prebuild_pair(self):
        seen = []

        def fake_run(_command, **kwargs):
            build_dir = kwargs["env"].get("PANGU_TILED_HIP_BUILD_DIR")
            seen.append((build_dir, Path(build_dir).is_dir() if build_dir else None))
            return types.SimpleNamespace(
                returncode=0,
                stdout="Max VRAM: 1.0 MB\nCurrent VRAM: 1.0 MB\n",
            )

        args = Namespace(
            repeat=2,
            max_batches=1,
            python="python",
            fp16_checkpoint=None,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            PROBE.subprocess, "run", side_effect=fake_run
        ), mock.patch.dict(
            os.environ, {"PANGU_TILED_HIP_BUILD_DIR": "inherited-build-dir"}
        ):
            root = Path(directory)
            prebuild = list(PROBE.iter_candidates("p2-hip-prebuild"))[0]
            result = PROBE.run_one(
                prebuild,
                args=args,
                pangu_dir=root,
                output_dir=root / "result" / "output",
                baseline_dir=None,
            )
            ordinary = list(PROBE.iter_candidates("baseline"))[0]
            ordinary_args = Namespace(**vars(args))
            ordinary_args.repeat = 1
            PROBE.run_one(
                ordinary,
                args=ordinary_args,
                pangu_dir=root,
                output_dir=root / "result" / "output",
                baseline_dir=None,
            )

        fresh_paths = [item[0] for item in seen[:2]]
        self.assertEqual(len(set(fresh_paths)), 2)
        self.assertTrue(all(item[1] for item in seen[:2]))
        self.assertTrue(all(not Path(path).exists() for path in fresh_paths))
        self.assertEqual(seen[2][0], "inherited-build-dir")
        self.assertTrue(result["fresh_hip_build"])
        self.assertEqual(len(result["fresh_hip_build_dirs"]), 2)

    def test_aggregate_and_summary_retain_round_artifacts_and_are_deterministic(self):
        def output_round(round_index, baseline):
            return {
                "round": round_index,
                "available": not (baseline and round_index == 1),
                "exact": None if baseline and round_index == 1 else True,
                "set_equal": None if baseline and round_index == 1 else True,
                "shapes_equal": None if baseline and round_index == 1 else True,
                "dtypes_equal": None if baseline and round_index == 1 else True,
                "array_equal": None if baseline and round_index == 1 else True,
                "sha256_equal": None if baseline and round_index == 1 else True,
                "all_finite": True,
                "nan_count": 0,
                "inf_count": 0,
                "candidate_files": 2,
                "baseline_files": None if baseline and round_index == 1 else 2,
                "file_checks": [],
            }

        baseline = {
            "label": "base",
            "kind": "baseline",
            "repeat": 2,
            "max_batches": 2,
            "latency_rounds_ms": [[11.0, 9.0], [12.0, 10.0]],
            "steady_latency_rounds_ms": [[9.0], [10.0]],
            "max_vram_rounds_mb": [480.0, 482.0],
            "current_vram_rounds_mb": [90.0, 91.0],
            "reserved_rounds_mb": [512.0, 512.0],
            "p2_region_setup_rounds": [None, None],
            "hip_prebuild_report_rounds": [None, None],
            "output_comparison_rounds": [
                output_round(1, True),
                output_round(2, True),
            ],
        }
        candidate = {
            "label": "candidate",
            "kind": "p2-region-release",
            "repeat": 2,
            "max_batches": 2,
            "latency_rounds_ms": [[10.0, 8.0], [10.0, 8.0]],
            "steady_latency_rounds_ms": [[8.0], [8.0]],
            "max_vram_rounds_mb": [470.0, 471.0],
            "current_vram_rounds_mb": [42.0, 43.0],
            "reserved_rounds_mb": [464.0, 464.0],
            "p2_region_setup_rounds": [
                {"net_unique_bytes_reclaimed": 50000000},
                {"net_unique_bytes_reclaimed": 50000000},
            ],
            "hip_prebuild_report_rounds": [None, None],
            "output_comparison_rounds": [
                output_round(1, False),
                output_round(2, False),
            ],
        }
        first = PROBE.build_interleaved_summary(
            [baseline, candidate], seed=7, samples=200
        )
        second = PROBE.build_interleaved_summary(
            [baseline, candidate], seed=7, samples=200
        )
        self.assertEqual(first, second)
        comparison = first["comparisons"][0]
        self.assertTrue(comparison["all_batches"]["complete"])
        self.assertEqual(comparison["all_batches"]["paired_delta_mean_ms"], -1.5)
        self.assertEqual(comparison["steady"]["paired_delta_mean_ms"], -1.5)
        self.assertEqual(
            comparison["current_vram_mb"]["paired_round_deltas"],
            [-48.0, -48.0],
        )
        self.assertTrue(comparison["exact_output_all_rounds"])
        self.assertEqual(
            comparison["candidate_region_setup_rounds"][0][
                "net_unique_bytes_reclaimed"
            ],
            50000000,
        )

    def test_region_stage1_hard_gate_passes_only_complete_5x4_evidence(self):
        def output_round(round_index, baseline):
            unavailable = baseline and round_index == 1
            return {
                "round": round_index,
                "available": not unavailable,
                "exact": None if unavailable else True,
                "set_equal": None if unavailable else True,
                "shapes_equal": None if unavailable else True,
                "dtypes_equal": None if unavailable else True,
                "array_equal": None if unavailable else True,
                "sha256_equal": None if unavailable else True,
                "all_finite": True,
                "nan_count": 0,
                "inf_count": 0,
                "candidate_files": 4,
                "baseline_files": None if unavailable else 4,
                "file_checks": [],
            }

        baseline = {
            "label": "base",
            "kind": "baseline",
            "returncode": 0,
            "repeat": 5,
            "max_batches": 4,
            "latency_rounds_ms": [[101.0, 100.0, 100.0, 100.0]] * 5,
            "steady_latency_avg_ms": 100.0,
            "steady_latency_p90_ms": 100.0,
            "max_vram_rounds_mb": [500.0] * 5,
            "current_vram_rounds_mb": [100.0] * 5,
            "output_comparison_rounds": [
                output_round(index, True) for index in range(1, 6)
            ],
            "report_parse_errors": [],
        }
        region_report = {
            "attention_modules": 16,
            "shifted_mask_owners": 8,
            "dense_mask_logical_bytes_after": 0,
            "dense_mask_unique_bytes_after": 0,
            "actual_cuda_allocated_reclaimed_bytes": 50 * 2**20,
        }
        candidate = {
            "label": "candidate",
            "kind": "p2-region-release",
            "returncode": 0,
            "repeat": 5,
            "max_batches": 4,
            "latency_rounds_ms": [[101.0, 100.4, 100.4, 100.4]] * 5,
            "steady_latency_avg_ms": 100.4,
            "steady_latency_p90_ms": 100.8,
            "max_vram_rounds_mb": [500.5] * 5,
            "current_vram_rounds_mb": [50.0] * 5,
            "p2_region_setup_rounds": [region_report] * 5,
            "output_comparison_rounds": [
                output_round(index, False) for index in range(1, 6)
            ],
            "report_parse_errors": [],
        }
        gate = PROBE.build_stage1_hard_gate(
            [baseline, candidate],
            preset="p2-region-release",
            timer_sha256=PROBE.FROZEN_OFFICIAL_TIMER_SHA256,
        )
        self.assertTrue(gate["passed"], gate)
        candidate["output_comparison_rounds"][2]["candidate_files"] = 3
        rejected = PROBE.build_stage1_hard_gate(
            [baseline, candidate],
            preset="p2-region-release",
            timer_sha256=PROBE.FROZEN_OFFICIAL_TIMER_SHA256,
        )
        self.assertFalse(rejected["passed"])


if __name__ == "__main__":
    unittest.main()
