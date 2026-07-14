import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "diagnose_uv_next.py"
SPEC = importlib.util.spec_from_file_location("diagnose_uv_next", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnoseUVNextTests(unittest.TestCase):
    def test_score_mapping_uses_semantic_fields(self):
        reference = MODULE.SCORE_REFERENCE
        self.assertEqual(reference["metric_mapping"]["U"], "inference_time")
        self.assertEqual(reference["metric_mapping"]["V"], "lightweight")
        self.assertEqual(reference["score_inference_time"], 17.9758)
        self.assertEqual(reference["score_lightweight"], 35.9786)
        self.assertNotIn("u", reference)
        self.assertNotIn("v", reference)

    def test_sample_summary_separates_distribution_metrics(self):
        summary = MODULE.summarize_samples([70.0, 80.0, 90.0, 100.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean_ms"], 85.0)
        self.assertEqual(summary["p50_ms"], 85.0)
        self.assertEqual(summary["p90_ms"], 97.0)
        self.assertGreater(summary["cv"], 0)

    def test_paired_bootstrap_detects_clear_latency_win(self):
        candidate = [75.0, 76.0, 77.0, 78.0]
        oracle = [100.0, 101.0, 102.0, 103.0]
        result = MODULE.paired_bootstrap_ci(
            candidate, oracle, samples=500, seed=7
        )
        self.assertEqual(result["mean_delta_ms"], -25.0)
        self.assertLess(result["upper_95_ms"], 0)

    def test_latency_score_targets_match_guardrail_elasticity(self):
        targets = MODULE.latency_score_targets(77.2945)
        self.assertAlmostEqual(targets[0]["target_latency_ms"], 75.3621375)
        self.assertAlmostEqual(targets[1]["target_latency_ms"], 73.429775)
        self.assertEqual(targets[1]["required_latency_reduction_pct"], 5.0)

    def test_cache_classification_is_explicit(self):
        self.assertEqual(
            MODULE.classify_cache_attribute("_pangu_p2_tiled_bias_index", 1),
            "packed_bias",
        )
        self.assertEqual(
            MODULE.classify_cache_attribute("_pangu_p2_tiled_bias_index", 2),
            "compact_index",
        )
        self.assertEqual(
            MODULE.classify_cache_attribute("_pangu_p2_tiled_region_ids", 1),
            "region_ids",
        )

    def test_storage_summary_deduplicates_aliases(self):
        records = [
            {
                "module": "layer1.block0",
                "kind": "packed_bias",
                "shape": [10],
                "dtype": "torch.float16",
                "logical_bytes": 20,
                "storage_bytes": 32,
                "storage_key": "cuda:0:123:32",
            },
            {
                "module": "layer1.block1",
                "kind": "packed_bias",
                "shape": [10],
                "dtype": "torch.float16",
                "logical_bytes": 20,
                "storage_bytes": 32,
                "storage_key": "cuda:0:123:32",
            },
        ]
        summary = MODULE.dedupe_tensor_records(records)
        self.assertEqual(summary["unique_storage_bytes"], 32)
        self.assertEqual(summary["by_kind"]["packed_bias"]["storage_count"], 1)
        self.assertEqual(summary["share_groups"][0]["module_count"], 2)

    def test_baseline_drift_is_fail_closed(self):
        passed = MODULE.assess_baseline_drift(77.3, 504.6, 109.9)
        failed = MODULE.assess_baseline_drift(82.0, 504.6, 109.9)
        missing = MODULE.assess_baseline_drift(77.3, None, 109.9)
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertFalse(missing["passed"])

    def test_correctness_gate_requires_exact_69_channel_outputs(self):
        valid = {
            "exact": True,
            "output_files": 5,
            "all_outputs_have_69_channels": True,
            "mismatch_count": 0,
            "nan_count": 0,
            "inf_count": 0,
        }
        self.assertTrue(MODULE.correctness_gate([valid])["passed"])
        invalid = dict(valid, all_outputs_have_69_channels=False)
        self.assertFalse(MODULE.correctness_gate([invalid])["passed"])

    def test_diagnosis_blocks_ranking_when_static_profile_drifts(self):
        static = {
            "provenance": {"status": "user_confirmed"},
            "checkpoint": {
                "profile": {
                    "patch_size": [2, 8, 8],
                    "embed_dim": 80,
                    "num_heads": [3, 6, 6, 3],
                    "depth_blocks": [2, 6, 6, 2],
                }
            },
        }
        runtime = {
            "integrity": {"passed": True},
            "variants": {
                "p2_on": {"steady": {"mean_ms": 77.0}},
                "p2_off": {"steady": {"mean_ms": 100.0}},
            },
            "event_attribution": {"summary": []},
            "p2_cache_inventory": {"by_kind": {}},
        }
        diagnosis = MODULE.build_diagnosis(static, runtime)
        self.assertFalse(diagnosis["diagnosis"]["valid_for_ranking"])
        self.assertIsNone(diagnosis["diagnosis"]["projections"])

    def test_memory_parser_and_safe_delta_support_failed_reports(self):
        parsed = MODULE.parse_memory_stdout(
            "[MEM] after load: allocated=88.8 MB, reserved=536.0 MB, "
            "peak=483.5 MB\nMax VRAM: 483.5 MB\nCurrent VRAM: 88.8 MB\n"
        )
        self.assertEqual(parsed["lifecycle"][0]["tag"], "after load")
        self.assertEqual(parsed["max_vram_mb"], 483.5)
        self.assertIsNone(MODULE._numeric_delta(None, 88.8))


if __name__ == "__main__":
    unittest.main()
