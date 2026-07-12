import json
import tempfile
import unittest
from pathlib import Path

import torch

from pangu_weather.score_training_utils import (
    SCORED_UPPER_INDICES,
    configure_trainable_stage,
    load_sensitive_layer_names,
    magnitude_resize_tensor,
    make_training_protocol,
    normalized_scored_rmse,
    parse_score_loss_weights,
    project_quantized_linear_weights,
    score_aligned_loss,
    score_validation_loss,
    split_scored_channels,
    validate_training_protocol,
    warmup_cosine_factor,
    YearBlockSampler,
)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patchrecovery = torch.nn.Linear(2, 2)
        self.layer1 = torch.nn.ModuleDict({"sensitive": torch.nn.Linear(2, 2)})
        self.layer2 = torch.nn.Linear(2, 2)


class ScoreTrainingUtilsTests(unittest.TestCase):
    def make_fields(self, fill=1.0):
        surface = torch.full((2, 4, 5, 6), fill)
        upper = torch.full((2, 65, 5, 6), fill)
        return surface, upper

    def test_scored_selection_matches_official_layout(self):
        surface = torch.arange(4.0).view(1, 4, 1, 1)
        upper = torch.arange(65.0).view(1, 65, 1, 1)
        scored, unscored = split_scored_channels(surface, upper)
        self.assertEqual(scored.shape[1], 15)
        self.assertEqual(unscored.shape[1], 54)
        self.assertEqual(scored[0, 4:, 0, 0].tolist(), list(SCORED_UPPER_INDICES))

    def test_perfect_prediction_has_near_zero_score_loss(self):
        fields = self.make_fields()
        normalizers = torch.ones(15)
        total, components = score_aligned_loss(fields, fields, fields, normalizers)
        self.assertLess(float(total), 1e-5)
        self.assertLess(float(components["acc"]), 1e-6)
        self.assertLess(float(score_validation_loss(fields, fields, normalizers)), 1e-5)

    def test_unscored_change_only_affects_unscored_teacher_term(self):
        target = self.make_fields()
        student_surface, student_upper = self.make_fields()
        student_upper[:, 0] += 2.0
        _, components = score_aligned_loss(
            (student_surface, student_upper), target, target, torch.ones(15)
        )
        self.assertLess(float(components["rmse"]), 2e-6)
        self.assertAlmostEqual(float(components["scored_teacher"]), 0.0, places=6)
        self.assertGreater(float(components["unscored_teacher"]), 0.0)

    def test_head_stage_selects_recovery_and_ranked_layer(self):
        model = ToyModel()
        trainable = configure_trainable_stage(model, "head", ["layer1.sensitive"])
        self.assertTrue(any("patchrecovery" in name for name in trainable))
        self.assertTrue(any("layer1.sensitive" in name for name in trainable))
        self.assertFalse(any("layer2" in name for name in trainable))
        configure_trainable_stage(model, "all")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_sensitive_ranking_loads_exact_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ranking.json"
            path.write_text(
                json.dumps([{"name": f"layer{i}"} for i in range(6)]),
                encoding="utf-8",
            )
            self.assertEqual(load_sensitive_layer_names(path, 5), [f"layer{i}" for i in range(5)])

    def test_physical_baseline_rmse_is_normalized_by_channel_std(self):
        baseline = torch.arange(1.0, 70.0)
        stds = baseline * 2.0
        normalizers = normalized_scored_rmse(baseline, stds)
        self.assertEqual(normalizers.shape, (15,))
        self.assertTrue(torch.allclose(normalizers, torch.full((15,), 0.5)))

    def test_quantization_projection_preserves_sensitive_layer(self):
        model = ToyModel()
        before_sensitive = model.layer1["sensitive"].weight.detach().clone()
        before_other = model.layer2.weight.detach().clone()
        projected = project_quantized_linear_weights(model, ["layer1.sensitive"])
        self.assertEqual(projected, 2)
        self.assertTrue(torch.equal(model.layer1["sensitive"].weight, before_sensitive))
        self.assertFalse(torch.equal(model.layer2.weight, before_other))

    def test_magnitude_mapping_preserves_qkv_groups(self):
        source = torch.zeros(12, 6)
        source[3, :] = 30.0
        source[7, :] = 20.0
        source[11, :] = 10.0
        mapped = magnitude_resize_tensor(source, (6, 4), preserve_qkv=True)
        self.assertEqual(tuple(mapped.shape), (6, 4))
        self.assertTrue(torch.any(mapped[:2] == 30.0))
        self.assertTrue(torch.any(mapped[2:4] == 20.0))
        self.assertTrue(torch.any(mapped[4:] == 10.0))

    def test_score_weights_are_configurable(self):
        fields = self.make_fields()
        student = (fields[0] + 1.0, fields[1] + 1.0)
        total, components = score_aligned_loss(
            student,
            fields,
            fields,
            torch.ones(15),
            weights=(0.0, 0.0, 1.0, 0.0),
        )
        self.assertTrue(torch.allclose(total, components["scored_teacher"]))

    def test_score_weight_parser_rejects_invalid_values(self):
        self.assertEqual(
            parse_score_loss_weights("0.45,0.20,0.25,0.10"),
            (0.45, 0.20, 0.25, 0.10),
        )
        for value in ("1,2,3", "1,2,3,4,5", "1,broken,3,4", "1,-2,3,4", "1,nan,3,4"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_score_loss_weights(value)

    def test_multi_epoch_warmup_cosine_schedule(self):
        total_steps = 8 * 2048
        self.assertAlmostEqual(warmup_cosine_factor(1, 256, total_steps), 1.0 / 256.0)
        self.assertAlmostEqual(warmup_cosine_factor(256, 256, total_steps), 1.0)
        self.assertGreater(
            warmup_cosine_factor(2048, 256, total_steps),
            warmup_cosine_factor(4096, 256, total_steps),
        )
        self.assertAlmostEqual(warmup_cosine_factor(total_steps, 256, total_steps), 0.01)
        completed = 1714
        self.assertAlmostEqual(
            warmup_cosine_factor(completed + 1, 256, total_steps),
            warmup_cosine_factor(1715, 256, total_steps),
        )

    def test_resume_protocol_rejects_legacy_and_mismatched_runs(self):
        protocol = make_training_protocol(
            {"name": "uv_c_patch12_w80_shallow"},
            8,
            2048,
            256,
            1.0e-5,
            True,
            (0.45, 0.20, 0.25, 0.10),
            0.01,
        )
        self.assertTrue(
            validate_training_protocol({"training_protocol": protocol}, protocol, require=True)
        )
        with self.assertRaisesRegex(ValueError, "predates the fixed training protocol"):
            validate_training_protocol({}, protocol, require=True)
        mismatched = dict(protocol)
        mismatched["total_epochs"] = 1
        with self.assertRaisesRegex(ValueError, "total_epochs"):
            validate_training_protocol(
                {"training_protocol": mismatched}, protocol, require=True
            )

        # Test protocol backward compatibility with default fallback values
        legacy_saved_protocol = dict(protocol)
        legacy_saved_protocol.pop("version", None)
        legacy_saved_protocol.pop("gradient_accumulation", None)
        legacy_saved_protocol.pop("attention_only_warmup_epochs", None)

        expected_protocol_with_defaults = dict(protocol)
        expected_protocol_with_defaults["gradient_accumulation"] = 1
        expected_protocol_with_defaults["attention_only_warmup_epochs"] = 0
        expected_protocol_with_defaults["version"] = 2  # Current version is 2

        self.assertTrue(
            validate_training_protocol(
                {"training_protocol": legacy_saved_protocol},
                expected_protocol_with_defaults,
                require=True
            )
        )

    def test_year_block_sampler_is_deterministic_and_year_local(self):
        class Dataset:
            selected_years = [2000, 2001]
            year_offsets = [0, 6]
            sample_counts = {2000: 6, 2001: 6}

            def __len__(self):
                return 12

        first = list(YearBlockSampler(Dataset(), block_size=3, seed=7))
        second = list(YearBlockSampler(Dataset(), block_size=3, seed=7))
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(12)))
        for offset in range(0, len(first), 3):
            block = first[offset : offset + 3]
            self.assertEqual(block, list(range(block[0], block[0] + len(block))))
            self.assertTrue(all(index < 6 for index in block) or all(index >= 6 for index in block))


if __name__ == "__main__":
    unittest.main()
