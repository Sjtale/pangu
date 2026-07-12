import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_pruned96_uv.py"
SPEC = importlib.util.spec_from_file_location("audit_pruned96_uv", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditPruned96UVTest(unittest.TestCase):
    def test_static_audit_accepts_full_pruned96_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pruned96.pth"
            torch.save(
                {
                    "model_profile": dict(MODULE.EXPECTED_PROFILE),
                    "model_state_dict": {
                        "patchembed2d.embedder.proj.weight": torch.zeros(
                            96, 7, 1, 8, 8, dtype=torch.float16
                        )
                    },
                },
                path,
            )
            audit = MODULE.audit_checkpoint(path)
            self.assertEqual(audit["profile"]["depth_blocks"], [2, 6, 6, 2])
            self.assertGreater(audit["file_size_mb"], 0)

    def test_profile_validation_rejects_shallow_s96(self):
        profile = dict(MODULE.EXPECTED_PROFILE)
        profile["depth_blocks"] = [1, 2, 2, 1]
        with self.assertRaisesRegex(ValueError, "not full pgw_lite_pruned_96"):
            MODULE.validate_profile(profile)

    def test_report_ranks_measured_stage_bottlenecks(self):
        static = {
            "file_size_mb": 30.0,
            "logical_tensor_mb": 40.0,
            "unique_storage_mb": 28.0,
            "alias_view_savings_mb": 12.0,
        }
        vram = [
            {
                "tag": "layer2_forward",
                "delta_peak_mb": 80.0,
                "peak_mb": 500.0,
                "delta_alloc_mb": 20.0,
                "allocated_mb": 300.0,
                "elapsed_ms": 30.0,
            },
            {
                "tag": "recovery",
                "delta_peak_mb": 20.0,
                "peak_mb": 520.0,
                "delta_alloc_mb": 100.0,
                "allocated_mb": 400.0,
                "elapsed_ms": 40.0,
            },
            {
                "tag": "steady.layer2_forward",
                "peak_mb": 510.0,
                "delta_peak_mb": 10.0,
                "delta_alloc_mb": 0.0,
                "allocated_mb": 300.0,
                "elapsed_ms": 25.0,
            },
            {
                "tag": "steady.recovery",
                "peak_mb": 520.0,
                "delta_peak_mb": 10.0,
                "delta_alloc_mb": 100.0,
                "allocated_mb": 400.0,
                "elapsed_ms": 10.0,
            },
        ]
        runtime = {
            "max_vram_mb": 520.0,
            "reserved_mb": 600.0,
            "current_vram_mb": 180.0,
            "latency_avg_ms": 110.0,
            "steady_latency_avg_ms": 100.0,
            "steady_latency_p50_ms": 99.0,
            "steady_latency_p90_ms": 102.0,
            "latency_ms_values": [140.0, 100.0],
        }
        report = MODULE.build_report(static, vram, runtime)
        self.assertIn("`recovery` high-water mark", report)
        self.assertIn("Cold-start V attribution is led by `recovery`", report)
        self.assertIn("steady V attribution is led by `layer2_forward`", report)

    def test_load_runtime_baseline_ignores_failed_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.jsonl"
            rows = [
                {"kind": "baseline", "returncode": 1},
                {"kind": "baseline", "returncode": 0, "max_vram_mb": 123.0},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.load_runtime_baseline(path)["max_vram_mb"], 123.0)


if __name__ == "__main__":
    unittest.main()
