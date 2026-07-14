import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from selective_mlp96 import (
    EXPECTED_PARAMETER_COUNT,
    MLP_RATIO_BLOCKS,
    PROFILE_NAME,
    PROFILE_SPEC,
    STAGE_PROTOCOLS,
    TARGET_MLP_BLOCKS,
    TARGET_MLP_PREFIXES,
    source_recovery_gradient_boundary,
    teacher_only_loss,
    validate_initialization_metadata,
    validate_inference_state_load,
    validate_profile,
    validate_stage_protocol,
)


class SelectiveMLP96TrainingTests(unittest.TestCase):
    def test_profile_and_target_blocks_are_exact(self):
        validate_profile({"name": PROFILE_NAME, **PROFILE_SPEC})
        self.assertEqual(MLP_RATIO_BLOCKS[1], [4, 2, 2, 2, 2, 2])
        self.assertEqual(MLP_RATIO_BLOCKS[2], [2, 2, 2, 2, 2, 2])
        self.assertEqual(len(TARGET_MLP_BLOCKS), 11)
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 14_768_265)
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            validate_profile(
                {"name": PROFILE_NAME, **PROFILE_SPEC, "embed_dim": 80}
            )

    def test_teacher_only_loss_uses_all_69_channels_with_fixed_weights(self):
        student_surface = torch.zeros(1, 4, 2, 2)
        teacher_surface = torch.ones_like(student_surface)
        student_upper = torch.zeros(1, 65, 2, 2)
        teacher_upper = torch.ones_like(student_upper) * 2
        loss, scored, unscored = teacher_only_loss(
            (student_surface, student_upper),
            (teacher_surface, teacher_upper),
        )
        expected_scored = (4.0 * 1.0 + 11.0 * 2.0) / 15.0
        self.assertAlmostEqual(scored.item(), expected_scored, places=6)
        self.assertEqual(unscored.item(), 2.0)
        self.assertAlmostEqual(
            loss.item(), 0.70 * expected_scored + 0.30 * 2.0, places=6
        )

    def test_training_protocols_are_locked_and_teacher_only(self):
        common = {
            "checkpoint_interval": 256,
            "gradient_accumulation": 1,
            "ground_truth_weight": 0.0,
            "teacher_weight": 1.0,
            "hint_weight": 0.0,
            "hint_layers": [],
            "score_aligned": False,
            "score_project_quantized": False,
        }
        for stage, protocol in STAGE_PROTOCOLS.items():
            with self.subTest(stage=stage):
                validate_stage_protocol(stage, **protocol, **common)
        with self.assertRaisesRegex(ValueError, "teacher-only"):
            validate_stage_protocol(
                "full_teacher",
                **STAGE_PROTOCOLS["full_teacher"],
                **{**common, "ground_truth_weight": 0.1},
            )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_stage_protocol(
                "fallback",
                **STAGE_PROTOCOLS["full_teacher"],
                **common,
            )

    def test_source_recovery_checkpoint_supports_frozen_inputs(self):
        class ToyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(4, 4)
                self.downsample = nn.Linear(4, 4)
                self.layer2 = nn.Linear(4, 4)
                self.layer3 = nn.Linear(4, 4)

            def forward(self, value):
                value = self.layer1(value)
                value = self.downsample(value)
                value = checkpoint(self.layer2, value, use_reentrant=True)
                return checkpoint(self.layer3, value, use_reentrant=True)

        student = ToyStudent()
        student.requires_grad_(False)
        student.layer2.weight.requires_grad_(True)
        student.layer2.bias.requires_grad_(True)
        student.layer3.weight.requires_grad_(True)
        student.layer3.bias.requires_grad_(True)
        model_input = torch.randn(2, 4)
        self.assertFalse(model_input.requires_grad)

        with source_recovery_gradient_boundary(student):
            output = student(model_input)

        self.assertTrue(output.requires_grad)
        output.square().mean().backward()
        for layer in (student.layer2, student.layer3):
            for parameter in layer.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNone(student.layer1.weight.grad)
        self.assertIsNone(student.downsample.weight.grad)
        self.assertFalse(student.downsample._forward_hooks)

    def test_initializer_metadata_schema_is_fail_closed(self):
        metadata = {
            "method": "pruned96_activation_aware_mlp_pair_selection",
            "profile_name": PROFILE_NAME,
            "source": "full_depth_ratio4_pruned96",
            "teacher": "official_full192",
            "source_sha256": "a" * 64,
            "teacher_sha256": "b" * 64,
            "init_sha256": "c" * 64,
            "mlp_ratio_blocks": MLP_RATIO_BLOCKS,
            "strict_coverage": True,
            "random_initialized_parameters": 0,
            "state_tensor_keys": 250,
            "covered_state_tensor_keys": 250,
            "parameter_state_keys": 218,
            "covered_parameter_state_keys": 218,
            "neuron_indices": {
                prefix: list(range(384)) for prefix in TARGET_MLP_PREFIXES
            },
        }
        validate_initialization_metadata(metadata)
        metadata["random_initialized_parameters"] = 1
        with self.assertRaisesRegex(ValueError, "initialization mismatch"):
            validate_initialization_metadata(metadata)

    def test_parameter_only_alias_load_is_lossless_and_fail_closed(self):
        class Aliased(nn.Module):
            def __init__(self):
                super().__init__()
                self.fuser = nn.Linear(3, 2)
                self.Fuser = self.fuser
                self.register_buffer("attn_mask", torch.zeros(1))

        model = Aliased()
        source = {name: value for name, value in model.named_parameters()}
        missing = [key for key in model.state_dict() if key not in source]
        validate_inference_state_load(model, source, missing, [])

        duplicate = dict(source)
        duplicate["Fuser.weight"] = source["fuser.weight"]
        with self.assertRaisesRegex(RuntimeError, "duplicate_aliases"):
            validate_inference_state_load(model, duplicate, missing, [])

        incomplete = dict(source)
        del incomplete["fuser.bias"]
        with self.assertRaisesRegex(RuntimeError, "missing_parameters"):
            validate_inference_state_load(model, incomplete, missing, [])

        with_buffer = dict(source)
        with_buffer["attn_mask"] = model.attn_mask
        with self.assertRaisesRegex(RuntimeError, "parameter-only"):
            validate_inference_state_load(model, with_buffer, [], [])

    def test_distill_entrypoint_wires_teacher_only_stages_without_labels(self):
        source = (
            Path(__file__).parents[1] / "distill_train.py"
        ).read_text(encoding="utf-8")
        self.assertIn("PANGU_SELECTIVE_MLP96_STAGE", source)
        self.assertIn("PANGU_SELECTIVE_MLP96_SOURCE_CHECKPOINT", source)
        self.assertIn("selective_mlp96_teacher_loss", source)
        self.assertIn("configure_selective_mlp96_trainable", source)
        self.assertIn("source_recovery_gradient_boundary", source)
        self.assertIn(
            '"layer2_layer3_reentrant_frozen_prefix_boundary"', source
        )
        self.assertIn('"ground_truth_backprop"] = False', source)
        self.assertIn("completed_steps in {1024, 2048, 3072}", source)
        self.assertIn("W-selection snapshot", source)
        loop_start = source.index("for step, data in enumerate")
        training_loop = source[
            loop_start : source.index("resume_epoch_step = 0", loop_start)
        ]
        self.assertIn("prepare_model_input(data, surface_mask, device)", training_loop)
        self.assertIn("target_surface = target_upper_air = None", training_loop)
        self.assertIn(
            'selective_mlp96_stage == "source_recovery"', training_loop
        )
        self.assertIn(
            "with source_recovery_gradient_boundary(student):", training_loop
        )
        self.assertIn(
            'student,\n                        ["layer2", "layer3"]',
            training_loop,
        )

        save_start = source.index("def save_student(")
        save_end = source.index("def prepare_model_input(", save_start)
        save_slice = source[save_start:save_end]
        self.assertIn("named_parameters()", save_slice)
        self.assertNotIn("remove_duplicate=False", save_slice)


if __name__ == "__main__":
    unittest.main()
