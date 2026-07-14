import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "run_selective_mlp96.sh"


class SelectiveMLP96LauncherTests(unittest.TestCase):
    def test_default_source_is_the_unquantized_pruned96_export(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'PANGU_SELECTIVE_MLP96_SOURCE:-data/checkpoints/model_pgw_lite_pruned_96_fp16.pth',
            source,
        )
        self.assertNotIn(
            'PANGU_SELECTIVE_MLP96_SOURCE:-data/checkpoints/model_fp16.pth',
            source,
        )

    def test_exact_two_stage_recipe_and_compliant_runtime_are_locked(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'run_stage source_recovery "$init_checkpoint" "$recovery_prefix" 1 512 2e-5 64',
            source,
        )
        self.assertIn(
            'run_stage full_teacher "$recovery_train" "$teacher_prefix" 3 1024 5e-6 128',
            source,
        )
        self.assertIn("PANGU_DISTILL_GROUND_TRUTH_WEIGHT=0", source)
        self.assertIn("PANGU_DISTILL_TEACHER_WEIGHT=1", source)
        self.assertIn("PANGU_DISTILL_CHECKPOINT_INTERVAL=256", source)
        self.assertIn("PANGU_COMPLIANT_FULL69_BOUNDARY=1", source)
        self.assertIn("--repeat 5", source)
        self.assertIn("--max-batches 5", source)
        self.assertIn("validate_selective_mlp96_runtime.py", source)
        self.assertIn('runtime_report="${runtime_log%.jsonl}_gate.json"', source)

    def test_unknown_action_fails_without_mutation(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "unknown"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
