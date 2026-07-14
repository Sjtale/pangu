import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PANGU = Path(__file__).resolve().parents[1]
SCRIPT = PANGU / "scripts" / "validate_selective_mlp96_runtime.py"
SPEC = importlib.util.spec_from_file_location("validate_selective_mlp96_runtime", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def runtime_row(kind, mean, p90, vram):
    return {
        "kind": kind,
        "returncode": 0,
        "repeat": 5,
        "max_batches": 5,
        "steady_latency_ms_values": [mean] * 20,
        "steady_latency_avg_ms": mean,
        "steady_latency_p90_ms": p90,
        "max_vram_mb": vram,
    }


class SelectiveMLP96RuntimeGateTests(unittest.TestCase):
    def test_all_runtime_and_size_gates_pass(self):
        report = MODULE.validate_runtime_rows(
            runtime_row("baseline", 100.0, 101.0, 500.0),
            runtime_row("checkpoint_candidate", 94.9, 100.0, 499.0),
            checkpoint_size_bytes=int(29.0 * MODULE.MIB),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["protocol"]["steady_points"], 20)
        self.assertEqual(report["gates"]["steady_mean"]["limit_ms"], 95.0)

    def test_relative_mean_p90_vram_and_size_each_fail_closed(self):
        baseline = runtime_row("baseline", 90.0, 91.0, 500.0)
        cases = (
            (runtime_row("checkpoint_candidate", 85.6, 90.0, 499.0), 29.0),
            (runtime_row("checkpoint_candidate", 85.0, 91.1, 499.0), 29.0),
            (runtime_row("checkpoint_candidate", 85.0, 90.0, 500.1), 29.0),
            (runtime_row("checkpoint_candidate", 85.0, 90.0, 499.0), 29.11),
        )
        for candidate, size_mib in cases:
            with self.subTest(candidate=candidate, size_mib=size_mib):
                report = MODULE.validate_runtime_rows(
                    baseline,
                    candidate,
                    checkpoint_size_bytes=int(size_mib * MODULE.MIB),
                )
                self.assertFalse(report["passed"])

    def test_protocol_requires_five_processes_and_twenty_steady_points(self):
        baseline = runtime_row("baseline", 100.0, 101.0, 500.0)
        candidate = runtime_row("checkpoint_candidate", 94.0, 100.0, 499.0)
        candidate["repeat"] = 4
        with self.assertRaisesRegex(ValueError, "5 independent processes"):
            MODULE.validate_runtime_rows(
                baseline, candidate, checkpoint_size_bytes=1
            )
        candidate = runtime_row("checkpoint_candidate", 94.0, 100.0, 499.0)
        candidate["steady_latency_ms_values"] = candidate[
            "steady_latency_ms_values"
        ][:-1]
        with self.assertRaisesRegex(ValueError, "20 steady latency points"):
            MODULE.validate_runtime_rows(
                baseline, candidate, checkpoint_size_bytes=1
            )

    def test_log_loader_requires_exact_ab_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.jsonl"
            rows = [
                runtime_row("baseline", 100.0, 101.0, 500.0),
                runtime_row("checkpoint_candidate", 94.0, 100.0, 499.0),
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            actual = MODULE.load_probe_rows(path)
            self.assertEqual(actual[0]["kind"], "baseline")
            self.assertEqual(actual[1]["kind"], "checkpoint_candidate")


if __name__ == "__main__":
    unittest.main()
