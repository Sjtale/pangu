"""Static smoke tests for distillation training wiring.

These tests intentionally parse source instead of importing distill_train.py so
they can run without PyTorch, OneScience, a GPU, or ERA5 data.
"""

import ast
import subprocess
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

    def test_official_uv_screening_is_fresh_and_unquantized(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        launcher = (
            SCRIPT_PATH.parent / "scripts" / "run_official_uv_screen.sh"
        ).read_text(encoding="utf-8")
        preparer = (
            SCRIPT_PATH.parent / "scripts" / "prepare_official_uv_student.py"
        ).read_text(encoding="utf-8")
        config = (SCRIPT_PATH.parent / "conf" / "config.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("PANGU_DISTILL_FRESH_OFFICIAL", source)
        self.assertIn("Fresh-official mode", source)
        self.assertIn('"PANGU_DISTILL_FRESH_OFFICIAL=$fresh_official"', launcher)
        self.assertIn('"PANGU_DISTILL_STEPS_PER_EPOCH=2048"', launcher)
        self.assertIn('epochs="${3:-1}"', launcher)
        self.assertIn('"PANGU_DISTILL_MAX_EPOCH=$epochs"', launcher)
        self.assertIn('"PANGU_DISTILL_CHECKPOINT_INTERVAL=256"', launcher)
        self.assertIn('fresh_official=0', launcher)
        self.assertIn('epoch_step', source)
        self.assertIn('Saved resumable checkpoint', source)
        self.assertIn('"PANGU_SCORE_LOSS_WEIGHTS=0.45,0.20,0.25,0.10"', launcher)
        self.assertIn('"PANGU_DISTILL_REQUIRE_PROTOCOL_MATCH=1"', launcher)
        self.assertIn('"PANGU_DISTILL_DISABLE_EARLY_STOPPING=1"', launcher)
        self.assertIn('"PANGU_SCORE_PROJECT_QUANTIZED=0"', launcher)
        self.assertNotIn("PANGU_IO_BLOCK_SAMPLER=1", launcher)
        self.assertIn("trained_epochs", preparer)
        for profile in (
            "uv_a_patch8_w80_shallow",
            "uv_s96_patch8_w96_shallow",
            "uv_b_patch8_w64_shallow",
            "uv_e_patch8_w80_ultrashallow",
            "uv_c_patch12_w80_shallow",
            "uv_d_patch16_w80_shallow",
        ):
            self.assertIn(profile, config)
        self.assertIn('S96) profile="uv_s96_patch8_w96_shallow"', launcher)
        self.assertIn("window_size: [2, 6, 12]", config)

    def test_a80_is_structurally_initialized_from_pruned96(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        launcher = (
            SCRIPT_PATH.parent / "scripts" / "run_official_uv_screen.sh"
        ).read_text(encoding="utf-8")
        pruner = (
            SCRIPT_PATH.parent / "scripts" / "prune_structured.py"
        ).read_text(encoding="utf-8")
        self.assertIn("${profile}_pgw96_structured_init_fp16.pth", launcher)
        self.assertIn('[[ "$candidate" == "S96" || "$candidate" == "A" ]]', launcher)
        self.assertIn("--require-unquantized-source", launcher)
        self.assertIn('elif [[ "$candidate" == "A" ]]', launcher)
        self.assertIn(
            '"PANGU_DISTILL_INIT_CHECKPOINT=$(basename "$screen_checkpoint")"',
            launcher,
        )
        self.assertIn('is_a80 = name == "uv_a_patch8_w80_shallow"', source)
        self.assertIn('"source_embed_dim": 96', source)
        self.assertIn('"target_embed_dim": 80', source)
        self.assertIn('"target_depth_blocks": [1, 2, 2, 1]', source)
        self.assertIn("Structured initialization source must be unquantized", pruner)

    def test_uv_candidate_has_lossless_pack_and_probe_phases(self):
        launcher = (
            SCRIPT_PATH.parent / "scripts" / "run_official_uv_screen.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("prepare|probe|train|pack|probe-packed", launcher)
        self.assertIn('trained_fp16="data/checkpoints/${prefix}_fp16.pth"', launcher)
        self.assertIn('packed_fp16="data/checkpoints/${prefix}_fp16_compact.pth"', launcher)
        self.assertIn("scripts/compact_fuser_alias_checkpoint.py", launcher)
        self.assertIn('elif [[ "$candidate" == "A" ]]', launcher)
        self.assertIn("uv_arch_${candidate_slug}_trained_compact.jsonl", launcher)

    def test_s96_uses_exact_pgw96_init_and_standard_l1(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        launcher = (
            SCRIPT_PATH.parent / "scripts" / "run_official_uv_screen.sh"
        ).read_text(encoding="utf-8")
        pruner = (
            SCRIPT_PATH.parent / "scripts" / "prune_structured.py"
        ).read_text(encoding="utf-8")
        self.assertIn("model_pgw_lite_pruned_96_fp16.pth", launcher)
        self.assertIn("--strict-exact-depth", launcher)
        self.assertIn(
            '"PANGU_DISTILL_INIT_CHECKPOINT=$(basename "$screen_checkpoint")"',
            launcher,
        )
        self.assertIn("normalized.startswith(checkpoint_dir + os.sep)", source)
        self.assertIn('"PANGU_DISTILL_GROUND_TRUTH_WEIGHT=0.5"', launcher)
        self.assertIn('"PANGU_DISTILL_TEACHER_WEIGHT=0.5"', launcher)
        self.assertIn('"PANGU_DISTILL_HINT_LAYERS="', launcher)
        self.assertIn('"PANGU_SCORE_ALIGNED=0"', launcher)
        self.assertIn("S96_SOURCE_DEPTHS = [2, 6, 6, 2]", pruner)
        self.assertIn("S96_TARGET_DEPTHS = [1, 2, 2, 1]", pruner)
        self.assertIn("expected_map = [[0], [0, 5], [0, 5], [0]]", pruner)
        self.assertIn('"interpolated_tensors": 0', pruner)
        self.assertIn('"resized_tensors": 0', pruner)
        self.assertIn("student_checkpoint = load_state(student, initial_student_path, strict=True)", source)

    def test_global_l1_and_single_lr_are_wired(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        forecast = source[
            source.index("def forecast_loss") : source.index(
                "def weighted_recovery_loss"
            )
        ]
        self.assertIn("F.l1_loss(upper_air, target_upper_air)", forecast)
        self.assertIn("0.25 * F.l1_loss", forecast)
        self.assertNotIn("dataset.weights", forecast)
        self.assertIn('getattr(cfg_data.dataset, "weights")', source)
        self.assertNotIn("get_llrd_param_groups", source)
        self.assertIn("trainable_parameters", source)
        self.assertIn("optimizer_param_groups=1", source)
        self.assertIn("len(optimizer.param_groups) != 1", source)

    def test_uv_launcher_rejects_invalid_epoch_counts(self):
        launcher = SCRIPT_PATH.parent / "scripts" / "run_official_uv_screen.sh"
        for value in ("0", "-1", "not-a-number"):
            with self.subTest(value=value):
                result = subprocess.run(
                    ["bash", str(launcher), "A", "train", value],
                    cwd=SCRIPT_PATH.parent,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("epochs must be a positive integer", result.stderr)

    def test_uv_launcher_rejects_invalid_candidate_and_prefix(self):
        launcher = SCRIPT_PATH.parent / "scripts" / "run_official_uv_screen.sh"
        invalid_candidate = subprocess.run(
            ["bash", str(launcher), "unknown", "train", "8"],
            cwd=SCRIPT_PATH.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(invalid_candidate.returncode, 2)
        invalid_prefix = subprocess.run(
            ["bash", str(launcher), "A", "prepare"],
            cwd=SCRIPT_PATH.parent,
            env={"PATH": "/usr/bin:/bin", "PANGU_UV_SCREEN_PREFIX": "../unsafe"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(invalid_prefix.returncode, 2)
        self.assertIn("must be a safe basename", invalid_prefix.stderr)

    def test_effective_uv_training_protocol_is_wired(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('cfg_float(cfg, "distill_learning_rate", 5.0e-5)', source)
        self.assertIn("parse_score_loss_weights", source)
        self.assertIn("weights=score_loss_weights", source)
        self.assertNotIn("weights=(0.45, 0.20, 0.25, 0.10)", source)
        self.assertIn("training_protocol", source)
        self.assertIn("validate_training_protocol", source)
        loop = source[source.index("optimizer.zero_grad()") : source.index("train_total +=")]
        self.assertLess(loop.index("scheduler.step()"), loop.index("optimizer.step()"))

    def test_validation_loader_uses_one_low_prefetch_worker(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        validation_loader = source[
            source.index("valid_loader = torch.utils.data.DataLoader(") :
            source.index("static_dir = cfg_data.dataset.static_dir")
        ]
        self.assertIn("num_workers=1", validation_loader)
        self.assertIn("prefetch_factor=1", validation_loader)
        self.assertIn("persistent_workers=False", validation_loader)


if __name__ == "__main__":
    unittest.main()
