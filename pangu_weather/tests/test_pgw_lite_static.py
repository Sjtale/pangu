import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


class PGWLiteStaticTests(unittest.TestCase):
    def test_config_contains_only_submission_student(self):
        config = yaml.safe_load((ROOT / "conf/config.yaml").read_text(encoding="utf-8"))
        profiles = config["model"]["student_profiles"]
        self.assertEqual(list(profiles), ["pgw_lite_pruned_96"])
        self.assertEqual(profiles["pgw_lite_pruned_96"]["patch_size"], [2, 8, 8])
        self.assertEqual(profiles["pgw_lite_pruned_96"]["embed_dim"], 96)
        self.assertEqual(profiles["pgw_lite_pruned_96"]["depth_blocks"], [2, 6, 6, 2])

    def test_config_has_69_channels_and_weights(self):
        config = yaml.safe_load((ROOT / "conf/config.yaml").read_text(encoding="utf-8"))
        dataset = config["datapipe"]["dataset"]
        self.assertEqual(len(dataset["channels"]), 69)
        self.assertEqual(len(dataset["weights"]), 69)

    def test_fixed_p2_adapter_has_no_debug_or_rollback(self):
        source = (ROOT / "p2_tiled_attention.py").read_text(encoding="utf-8")
        self.assertIn('mode="full-row-fast"', source)
        self.assertIn("if patched != 16", source)
        self.assertNotIn("PANGU_P2_TILED_DEBUG", source)
        self.assertNotIn("disable_p2_tiled_attention", source)
        ast.parse(source)


if __name__ == "__main__":
    unittest.main()
