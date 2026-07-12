import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pangu_lite_2d import EarthSpecificLayer2DNoBias, PanguLite2DAttentionPosEmbed
from score_training_utils import kd_2d_score_loss


class PanguLite2DKDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PanguLite2DAttentionPosEmbed()

    def test_architecture_contract(self):
        self.assertEqual(tuple(self.model.absolute_pos_embed.shape), (1, 91, 288))
        self.assertEqual(self.model.patch_size, (8, 8))
        self.assertEqual(
            sum(isinstance(module, EarthSpecificLayer2DNoBias) for module in self.model.modules()),
            4,
        )
        self.assertFalse(any("earth_position_bias" in key for key in self.model.state_dict()))
        self.assertEqual(self.model.patchrecovery.out_channels, 69)

    def test_kd_loss_splits_and_detaches_teacher(self):
        student = (
            torch.randn(1, 4, 9, 16, requires_grad=True),
            torch.randn(1, 65, 9, 16, requires_grad=True),
        )
        target = (torch.randn_like(student[0]), torch.randn_like(student[1]))
        teacher = (
            torch.randn_like(student[0], requires_grad=True),
            torch.randn_like(student[1], requires_grad=True),
        )
        loss, parts = kd_2d_score_loss(student, target, teacher, torch.ones(15))
        loss.backward()
        self.assertEqual(set(parts), {"rmse", "acc", "scored_teacher", "unscored_teacher"})
        self.assertIsNotNone(student[0].grad)
        self.assertIsNotNone(student[1].grad)
        self.assertIsNone(teacher[0].grad)
        self.assertIsNone(teacher[1].grad)

    def test_baseline_contract_is_strict(self):
        fields = (torch.zeros(1, 4, 3, 4), torch.zeros(1, 65, 3, 4))
        with self.assertRaisesRegex(ValueError, "15 finite positive"):
            kd_2d_score_loss(fields, fields, fields, torch.ones(14))

    def test_launcher_locks_long_run_budget(self):
        source = (ROOT / "scripts/run_pangu_lite_2d_kd_100e.sh").read_text()
        for expected in (
            "PANGU_DISTILL_MAX_EPOCH=100",
            "PANGU_DISTILL_GRADIENT_ACCUMULATION=4",
            "PANGU_DISTILL_WARMUP_STEPS=1024",
            "PANGU_SCORE_LOSS_WEIGHTS=0.55,0.30,0.10,0.05",
            "WORLD_SIZE=1",
        ):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
