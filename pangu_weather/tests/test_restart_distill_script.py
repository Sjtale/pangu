"""Static checks for the pruned96 distillation restart helper."""

import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "restart_pruned96_distill.sh"


class RestartDistillScriptTests(unittest.TestCase):
    def test_script_defaults_restart_from_structural_pruned96_checkpoint(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("PANGU_RESTART_PROFILE:-pgw_lite_pruned_96", source)
        self.assertIn(
            "PANGU_RESTART_INIT_CHECKPOINT:-model_pgw_lite_pruned_96.pth",
            source,
        )
        self.assertIn("PANGU_RESTART_GROUND_TRUTH_WEIGHT:-0.3", source)
        self.assertIn("PANGU_RESTART_TEACHER_WEIGHT:-0.5", source)
        self.assertIn("PANGU_RESTART_HINT_WEIGHT:-0", source)
        self.assertIn("PANGU_RESTART_ALLOW_QUANTIZED_INIT:-0", source)

    def test_script_rejects_quantized_init_by_default(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("int8_count = sum", source)
        self.assertIn('value.dtype == torch.int8', source)
        self.assertIn('str(key).endswith("_scale")', source)
        self.assertIn("Refusing quantized init for structural restart", source)
        self.assertIn("PANGU_RESTART_ALLOW_QUANTIZED_INIT=1", source)

    def test_script_has_explicit_quantized_recovery_modes(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("quant-check", source)
        self.assertIn("quant-smoke", source)
        self.assertIn("quant-full", source)
        self.assertIn("quant-smoke-full", source)
        self.assertIn("init_checkpoint=\"${PANGU_RESTART_INIT_CHECKPOINT:-model_fp16.pth}\"", source)
        self.assertIn("allow_quantized_init=\"${PANGU_RESTART_ALLOW_QUANTIZED_INIT:-1}\"", source)
        self.assertIn("Quantized recovery mode expected INT8 tensors", source)

    def test_script_has_smoke_and_full_schedules(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("PANGU_RESTART_SMOKE_STEPS:-20", source)
        self.assertIn("PANGU_RESTART_FULL_STEPS:-512", source)
        self.assertIn("PANGU_RESTART_FULL_EPOCHS:-30", source)
        self.assertIn("smoke-full", source)

    def test_script_backs_up_resume_and_export_artifacts(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("model_${profile}_latest.pth", source)
        self.assertIn("model_${profile}_train.pth", source)
        self.assertIn("model_${profile}_fp16.pth", source)
        self.assertIn("restart_backup_${profile}", source)

    def test_script_validates_expected_log_fields(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("student_profile=${profile}", source)
        self.assertIn("expected_init=", source)
        self.assertIn("expected_loss=", source)
        self.assertIn("Epoch schedule: start_epoch=0", source)
        self.assertIn('if [[ "$mode" == "smoke" || "$mode" == "quant-smoke" ]]', source)
        self.assertIn('elif [[ "$mode" == "full" || "$mode" == "quant-full" ]]', source)


if __name__ == "__main__":
    unittest.main()
