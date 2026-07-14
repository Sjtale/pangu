"""Regression tests for storage-preserving checkpoint conversion."""

import ast
import importlib.util
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "convert_fp16.py"
INFERENCE_PATH = Path(__file__).parents[1] / "inference.py"


class ConvertFp16StaticTests(unittest.TestCase):
    def test_script_parses_and_contains_safety_checks(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for required in (
            "weights_only=False",
            "untyped_storage",
            "as_strided",
            "torch.equal",
            "os.replace",
            "--audit-only",
        ):
            self.assertIn(required, source)

    def test_inference_uses_only_the_submitted_checkpoint(self):
        source = INFERENCE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertNotIn("PANGU_FP16_CHECKPOINT", source)
        self.assertIn('"model_fp16.pth"', source)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local test environment")
class ConvertFp16TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("convert_fp16", SCRIPT_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_conversion_preserves_values_views_and_minimal_payload(self):
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        state = OrderedDict((
            ("weight", base),
            ("weight_alias", base),
            ("weight_view", base[:, 1:5]),
            ("counter", torch.arange(3, dtype=torch.int64)),
        ))

        converted = self.module.convert_state_dict_to_fp16(state)
        self.assertEqual(converted["weight"].dtype, torch.float16)
        self.assertTrue(torch.equal(converted["weight"], state["weight"].half()))
        self.assertTrue(torch.equal(converted["weight_view"], state["weight_view"].half()))
        storage_ids = {
            converted[name].untyped_storage()._cdata
            for name in ("weight", "weight_alias", "weight_view")
        }
        self.assertEqual(len(storage_ids), 1)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            torch.save(
                {
                    "model_state_dict": state,
                    "optimizer_state_dict": {"x": 1},
                    "model_profile": {"name": "pgw_lite_pruned_96"},
                },
                source,
            )
            self.module.convert_to_fp16(str(source), str(output))
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            self.assertNotIn("optimizer_state_dict", checkpoint)
            self.assertEqual(
                checkpoint["model_profile"], {"name": "pgw_lite_pruned_96"}
            )
            self.module.verify_checkpoint(state, str(output))


if __name__ == "__main__":
    unittest.main()
