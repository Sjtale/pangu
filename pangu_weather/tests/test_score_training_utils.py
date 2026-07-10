import json
import tempfile
import unittest
from pathlib import Path

import torch

from pangu_weather.score_training_utils import (
    SCORED_UPPER_INDICES,
    configure_trainable_stage,
    load_sensitive_layer_names,
    normalized_scored_rmse,
    project_quantized_linear_weights,
    score_aligned_loss,
    score_validation_loss,
    split_scored_channels,
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


if __name__ == "__main__":
    unittest.main()
