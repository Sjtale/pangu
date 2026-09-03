import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE = (ROOT / "distill_train.py").read_text(encoding="utf-8")


class DistillTrainStaticTests(unittest.TestCase):
    def test_fixed_profile_and_losses(self):
        self.assertIn('"name": "pgw_lite_pruned_96"', SOURCE)
        self.assertIn('"depth_blocks": [2, 6, 6, 2]', SOURCE)
        self.assertIn("GROUND_TRUTH_WEIGHT = 0.5", SOURCE)
        self.assertIn("TEACHER_WEIGHT = 0.5", SOURCE)
        for marker in ("HINT_WEIGHT", "HINT_LAYERS", "feature_hint_loss", "FeatureCapture"):
            self.assertNotIn(marker, SOURCE)

    def test_all_69_non_residual_metadata(self):
        self.assertIn('"teacher_source": "organizer_pangu_full_model"', SOURCE)
        self.assertIn('"all_69_channels": True', SOURCE)
        self.assertIn('"predict_residual": False', SOURCE)

    def test_no_historical_profiles_or_runtime_switches(self):
        for marker in (
            "selective_mlp96",
            "PANGU_STUDENT_PROFILE",
            "PANGU_SCORE_ALIGNED",
            "PANGU_RECOVERY_ONLY",
            "DistributedDataParallel",
        ):
            self.assertNotIn(marker, SOURCE)

    def test_source_parses(self):
        ast.parse(SOURCE)


if __name__ == "__main__":
    unittest.main()
