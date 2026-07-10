"""Static smoke tests for distillation training wiring.

These tests intentionally parse source instead of importing distill_train.py so
they can run without PyTorch, OneScience, a GPU, or ERA5 data.
"""

import ast
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "distill_train.py"


def _parse_script():
    return ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


class DistillTrainStaticTests(unittest.TestCase):
    def test_save_student_calls_supply_required_arguments(self):
        tree = _parse_script()
        save_student = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "save_student"
        )
        positional_args = [arg.arg for arg in save_student.args.args]
        required_count = len(positional_args) - len(save_student.args.defaults)
        required_names = positional_args[:required_count]

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_student"
        ]
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            supplied = set(required_names[: len(call.args)])
            supplied.update(keyword.arg for keyword in call.keywords if keyword.arg)
            missing = [name for name in required_names if name not in supplied]
            self.assertFalse(
                missing,
                f"save_student call at line {call.lineno} misses {missing}",
            )

    def test_latest_and_best_checkpoint_paths_are_separate(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn('"model_distilled_latest.pth"', source)
        self.assertIn("PANGU_STUDENT_PROFILE", source)
        self.assertIn("pgw_lite_distilled_latest_checkpoint", source)
        self.assertIn("load_compatible_state", source)
        self.assertIn("latest_train_checkpoint", source)
        self.assertIn("cfg.distilled_train_checkpoint", source)
        self.assertIn("cfg.distilled_checkpoint", source)

    def test_hint_loss_feature_capture_is_wired(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("class FeatureCapture", source)
        self.assertIn("register_forward_hook", source)
        self.assertIn("feature_hint_loss", source)
        self.assertIn("tokens_to_feature_grid", source)
        self.assertIn("F.adaptive_avg_pool3d", source)
        self.assertIn("distill_hint_layers", source)
        self.assertIn("distill_hint_weight", source)
        self.assertIn("teacher_weight=teacher_weight", source)

    def test_resume_can_extend_completed_training(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("def cfg_int", source)
        self.assertIn("distill_extra_epochs", source)
        self.assertIn("PANGU_DISTILL_EXTRA_EPOCHS", source)
        self.assertIn("Extending distillation", source)
        self.assertIn("No epochs to run", source)

    def test_resume_can_start_from_best_checkpoint(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("PANGU_DISTILL_RESUME_FROM", source)
        self.assertIn("Resuming from best training checkpoint", source)
        self.assertIn("PANGU_DISTILL_RESUME_FROM=best requested", source)

    def test_score_aligned_mode_is_opt_in_and_uses_separate_checkpoints(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        launcher = (
            SCRIPT_PATH.parent / "scripts" / "run_score_aligned_finetune.sh"
        ).read_text(encoding="utf-8")
        quantizer = (
            SCRIPT_PATH.parent / "scripts" / "quantize_mixed_precision.py"
        ).read_text(encoding="utf-8")
        self.assertIn('env_enabled("PANGU_SCORE_ALIGNED")', source)
        self.assertIn("score_aligned_loss", source)
        self.assertIn("score_validation_loss", source)
        self.assertIn("PANGU_DISTILL_CHECKPOINT_PREFIX", source)
        self.assertIn("PANGU_SCORE_STAGE=head", launcher)
        self.assertIn("PANGU_SCORE_STAGE=all", launcher)
        self.assertIn("official_baseline_rmse.npy", launcher)
        self.assertIn('parser.add_argument(\n        "--output"', quantizer)
        self.assertIn("Refusing to overwrite candidate output", quantizer)


if __name__ == "__main__":
    unittest.main()
