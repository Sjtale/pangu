import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import types
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


PANGU = Path(__file__).parents[1]
INITIALIZER = PANGU / "scripts" / "initialize_selective_1441_full192.py"
TRAINER = PANGU / "distill_train.py"
CONFIG = PANGU / "conf" / "config.yaml"
LAUNCHER = PANGU / "scripts" / "run_selective_1441_full192.sh"
ARCHIVER = PANGU / "scripts" / "archive_selective_1441_full192.py"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        )
    ]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Selective1441InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = {
            "PROFILE_NAME",
            "SOURCE_PATCH_SIZE",
            "SOURCE_EMBED_DIM",
            "SOURCE_NUM_HEADS",
            "SOURCE_DEPTHS",
            "TARGET_PATCH_SIZE",
            "TARGET_EMBED_DIM",
            "TARGET_NUM_HEADS",
            "TARGET_DEPTHS",
            "TARGET_WINDOW_SIZE",
            "TARGET_MLP_RATIO",
            "EXPECTED_PARAMETER_COUNT",
            "BLOCK_MAP",
            "BLOCK_PATTERN",
            "clean_state_dict",
            "validate_full192_source",
            "validate_full192_structure",
            "validate_target_model",
            "source_key_for_target",
            "resize_patch_weight",
            "interpolate_earth_bias",
            "sha256_file",
            "atomic_deterministic_save",
        }
        cls.namespace = load_functions(
            INITIALIZER,
            names,
            {
                "OrderedDict": OrderedDict,
                "re": re,
                "torch": torch,
                "F": F,
                "_state_depths": lambda _state: [2, 6, 6, 2],
                "hashlib": hashlib,
                "os": os,
                "Path": Path,
            },
        )

    def test_profile_and_fixed_shift_compatible_block_map(self):
        self.assertEqual(
            self.namespace["PROFILE_NAME"], "pangu_selective_1441_full192"
        )
        self.assertEqual(self.namespace["TARGET_PATCH_SIZE"], [2, 8, 8])
        self.assertEqual(self.namespace["TARGET_NUM_HEADS"], [3, 6, 6, 3])
        self.assertEqual(self.namespace["TARGET_DEPTHS"], [1, 4, 4, 1])
        self.assertEqual(self.namespace["TARGET_WINDOW_SIZE"], [2, 6, 12])
        self.assertEqual(self.namespace["TARGET_MLP_RATIO"], 4)
        self.assertEqual(
            self.namespace["BLOCK_MAP"],
            [[0], [0, 1, 4, 5], [0, 1, 4, 5], [0]],
        )
        source_key = self.namespace["source_key_for_target"]
        self.assertEqual(
            source_key("layer2.fuser.blocks.2.transformer.norm1.weight"),
            "layer2.fuser.blocks.4.transformer.norm1.weight",
        )
        self.assertEqual(
            source_key("layer3.Fuser.blocks.3.transformer.mlp.fc2.weight"),
            "layer3.Fuser.blocks.5.transformer.mlp.fc2.weight",
        )
        self.assertEqual(
            source_key("patchrecovery3d.recovery.proj.weight"),
            "patchrecovery3d.recovery.proj.weight",
        )

    def test_patch_resize_is_deterministic_and_preserves_embedding_scale(self):
        resize = self.namespace["resize_patch_weight"]
        source = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
        first = resize(source, (1, 1, 1, 8, 8), True)
        second = resize(source, (1, 1, 1, 8, 8), True)
        recovery = resize(source, (1, 1, 1, 8, 8), False)
        self.assertEqual(tuple(first.shape), (1, 1, 1, 8, 8))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first * 4.0, recovery))

    def test_earth_bias_interpolates_windows_after_head_selection(self):
        interpolate = self.namespace["interpolate_earth_bias"]
        source = torch.arange(2 * 64 * 3, dtype=torch.float32).reshape(2, 64, 3)
        first = interpolate(source, (2, 32, 3))
        second = interpolate(source, (2, 32, 3))
        self.assertEqual(tuple(first.shape), (2, 32, 3))
        self.assertTrue(torch.equal(first, second))

    def test_source_audit_accepts_only_unquantized_full192(self):
        validate = self.namespace["validate_full192_source"]
        state = OrderedDict(
            {
                "patchembed2d.embedder.proj.weight": torch.zeros(192, 7, 1, 4, 4),
                "patchembed3d.embedder.proj.weight": torch.zeros(192, 5, 2, 4, 4),
            }
        )
        validate({}, state)
        with self.assertRaisesRegex(ValueError, "unquantized"):
            validate({}, OrderedDict(state, quantized=torch.zeros(1, dtype=torch.int8)))
        with self.assertRaisesRegex(ValueError, "not the official full_192"):
            validate({"distillation": {}}, state)
        with self.assertRaisesRegex(ValueError, "model_profile mismatch"):
            validate(
                {
                    "model_profile": {
                        "patch_size": [2, 8, 8],
                        "embed_dim": 96,
                        "num_heads": [3, 6, 6, 3],
                        "depth_blocks": [2, 6, 6, 2],
                    }
                },
                state,
            )

    def test_target_structure_has_ten_blocks_shift_phases_and_69_outputs(self):
        validate = self.namespace["validate_target_model"]
        self.namespace["_state_depths"] = lambda _state: [1, 4, 4, 1]
        shifts = [
            [(0, 0, 0)],
            [(0, 0, 0), (1, 3, 6), (0, 0, 0), (1, 3, 6)],
            [(0, 0, 0), (1, 3, 6), (0, 0, 0), (1, 3, 6)],
            [(0, 0, 0)],
        ]
        model = types.SimpleNamespace(
            state_dict=lambda: {
                "patchembed2d.embedder.proj.weight": torch.empty(1, 1, 8, 8),
                "patchembed3d.embedder.proj.weight": torch.empty(1, 2, 8, 8),
                "patchrecovery2d.recovery.proj.bias": torch.empty(4),
                "patchrecovery3d.recovery.proj.bias": torch.empty(5),
            },
            parameters=lambda: [
                types.SimpleNamespace(
                    numel=lambda: self.namespace["EXPECTED_PARAMETER_COUNT"]
                )
            ],
        )
        for stage, stage_shifts in enumerate(shifts, start=1):
            setattr(
                model,
                f"layer{stage}",
                types.SimpleNamespace(
                    fuser=types.SimpleNamespace(
                        blocks=[
                            types.SimpleNamespace(shift_size=shift)
                            for shift in stage_shifts
                        ]
                    )
                ),
            )
        validate(model)
        self.assertEqual(sum(len(stage) for stage in shifts), 10)

    def test_full_source_structure_rejects_missing_and_shape_mismatch(self):
        validate = self.namespace["validate_full192_structure"]
        expected = OrderedDict(
            weight=torch.zeros(2, 3), bias=torch.zeros(2)
        )
        validate(OrderedDict(expected), expected)
        with self.assertRaisesRegex(ValueError, "structure mismatch"):
            validate(OrderedDict(weight=torch.zeros(2, 3)), expected)
        with self.assertRaisesRegex(ValueError, "structure mismatch"):
            validate(
                OrderedDict(weight=torch.zeros(3, 2), bias=torch.zeros(2)),
                expected,
            )

    def test_checkpoint_save_is_byte_repeatable_and_refuses_overwrite(self):
        save = self.namespace["atomic_deterministic_save"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.pth"
            payload = {"model_state_dict": OrderedDict(weight=torch.arange(8))}
            reported_hash = save(payload, output)
            self.assertEqual(reported_hash, self.namespace["sha256_file"](output))
            with self.assertRaises(FileExistsError):
                save(payload, output)


class Selective1441RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_functions(
            TRAINER,
            {
                "SELECTIVE_1441_PROFILE",
                "SELECTIVE_1441_SPEC",
                "weighted_recovery_loss",
                "validate_recovery_only_profile",
                "validate_recovery_only_schedule",
            },
            {
                "F": F,
                "env_enabled": lambda _name: False,
            },
        )

    def test_recovery_profile_is_exact(self):
        validate = self.namespace["validate_recovery_only_profile"]
        profile = {
            "name": self.namespace["SELECTIVE_1441_PROFILE"],
            **self.namespace["SELECTIVE_1441_SPEC"],
        }
        validate(profile)
        profile["depth_blocks"] = [1, 3, 3, 1]
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            validate(profile)

    def test_recovery_schedule_is_fixed_and_teacher_free(self):
        validate = self.namespace["validate_recovery_only_schedule"]
        valid = {
            "ground_truth_weight": 1.0,
            "teacher_weight": 0.0,
            "hint_weight": 0.0,
            "hint_layers": [],
            "score_aligned": False,
            "score_project_quantized": False,
            "total_epochs": 4,
            "steps_per_epoch": 2048,
            "warmup_steps": 256,
            "base_lr": 1.0e-5,
            "min_lr_ratio": 0.1,
            "gradient_accumulation": 1,
            "checkpoint_interval": 256,
        }
        validate(**valid)
        invalid = dict(valid, teacher_weight=0.1)
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            validate(**invalid)

    def test_recovery_loss_uses_all_channel_weights(self):
        loss_fn = self.namespace["weighted_recovery_loss"]
        surface = torch.ones(1, 4, 1, 1)
        upper = torch.ones(1, 65, 1, 1)
        target_surface = torch.zeros_like(surface)
        target_upper = torch.zeros_like(upper)
        surface_weights = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1)
        upper_weights = torch.arange(1, 66, dtype=torch.float32).view(1, 65, 1, 1)
        expected = upper_weights.mean() + 0.25 * surface_weights.mean()
        self.assertAlmostEqual(
            float(
                loss_fn(
                    surface,
                    upper,
                    target_surface,
                    target_upper,
                    surface_weights,
                    upper_weights,
                )
            ),
            float(expected),
        )

    def test_config_launcher_and_trainer_are_isolated(self):
        config = CONFIG.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        trainer = TRAINER.read_text(encoding="utf-8")
        initializer = INITIALIZER.read_text(encoding="utf-8")
        self.assertIn("pangu_selective_1441_full192:", config)
        self.assertIn("ARCHIVED/REJECTED", config)
        self.assertIn("depth_blocks: [1, 4, 4, 1]", config)
        self.assertIn("mlp_ratio: 4", config)
        self.assertIn("PANGU_RECOVERY_ONLY=1", launcher)
        self.assertIn("PANGU_DISTILL_MAX_EPOCH=4", launcher)
        self.assertIn("PANGU_DISTILL_TEACHER_WEIGHT=0.0", launcher)
        self.assertIn("PANGU_SCORE_ALIGNED=0", launcher)
        self.assertIn("if not recovery_only:", trainer)
        self.assertIn("teacher = None", trainer)
        self.assertIn('"teacher_forward": False', trainer)
        self.assertIn("random_initialized_parameters=0", initializer)
        self.assertIn("load_state_dict(saved[\"model_state_dict\"], strict=True)", initializer)
        self.assertIn('source_path.name != "model_bak.pth"', initializer)
        self.assertIn("existing_outputs", initializer)
        self.assertIn("PANGU_ALLOW_REJECTED_SELECTIVE_1441", trainer)

    def test_launcher_rejects_invalid_phase_without_touching_data(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "invalid"],
            cwd=PANGU,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_probe_forces_guardrail_interning_and_runs_once(self):
        launcher = LAUNCHER.read_text(encoding="utf-8")
        probe_region = launcher.split('if [[ "$phase" == "probe" ]]', 1)[1]
        probe_region = probe_region.split('if [[ "$phase" == "pack" ]]', 1)[0]
        self.assertIn("--buffer-intern 1", probe_region)
        self.assertIn("--repeat 1", probe_region)

        result = subprocess.run(
            [
                "python",
                "scripts/probe_uv_runtime_sweep.py",
                "--preset",
                "baseline",
                "--buffer-intern",
                "1",
                "--candidate-fp16-checkpoint",
                "synthetic.pth",
                "--dry-run",
            ],
            cwd=PANGU,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all('"PANGU_INTERN_IMMUTABLE_BUFFERS": "1"' in row for row in rows))
        self.assertTrue(all("_intern1_" in row for row in rows))

    def test_rejected_launcher_blocks_accidental_reuse(self):
        environment = os.environ.copy()
        environment.pop("PANGU_ALLOW_REJECTED_SELECTIVE_1441", None)
        result = subprocess.run(
            ["bash", str(LAUNCHER), "probe"],
            cwd=PANGU,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("archived/rejected", result.stderr)

    def test_archive_moves_only_selective_artifacts_and_writes_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            logs = root / "logs"
            archive = root / "archive"
            checkpoints.mkdir()
            logs.mkdir()
            init = checkpoints / "pangu_selective_1441_full192_init_train.pth"
            latest = checkpoints / "trial_latest.pth"
            baseline = checkpoints / "model_fp16_alias_compact.pth"
            training_log = logs / "distill_train_20260713.log"
            init.write_bytes(b"full192-init")
            latest.write_bytes(b"recovery-state")
            baseline.write_bytes(b"accepted-baseline")
            training_log.write_text("validation=0.2003\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    str(ARCHIVER),
                    "--prefix",
                    "trial",
                    "--checkpoint-dir",
                    str(checkpoints),
                    "--logs-dir",
                    str(logs),
                    "--archive-dir",
                    str(archive),
                ],
                cwd=PANGU,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(init.exists())
            self.assertFalse(latest.exists())
            self.assertFalse(training_log.exists())
            self.assertTrue(baseline.exists())
            manifest = json.loads(
                (archive / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "rejected_archived")
            self.assertEqual(
                manifest["rejection"]["weighted_validation_loss"], 0.2003
            )
            records = {
                record["archived_path"]: record
                for record in manifest["artifacts"]
            }
            archived_init = archive / "checkpoints" / init.name
            self.assertEqual(
                records[f"checkpoints/{init.name}"]["sha256"],
                hashlib.sha256(archived_init.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
