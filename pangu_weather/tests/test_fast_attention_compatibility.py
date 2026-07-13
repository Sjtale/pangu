import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_fast_attention_compatibility.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_fast_attention_compatibility", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastAttentionCompatibilityTests(unittest.TestCase):
    def test_representative_layout_combines_bias_and_shifted_mask(self):
        probe = _load_probe()
        args = SimpleNamespace(
            width_windows=2,
            pressure_height_windows=3,
            heads=2,
            tokens=4,
            head_dim=2,
            seed=7,
        )
        q, k, v, mask = probe._make_inputs(args, torch.device("cpu"))
        self.assertEqual(tuple(q.shape), (6, 2, 4, 2))
        self.assertEqual(tuple(k.shape), tuple(q.shape))
        self.assertEqual(tuple(v.shape), tuple(q.shape))
        self.assertEqual(tuple(mask.shape), (6, 2, 4, 4))
        self.assertTrue(torch.all(mask[..., :2, 2:] < -99.0))

    def test_flash_path_explicitly_disables_backend_fallback(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])", source)
        self.assertNotIn("SDPBackend.MATH", source)
        self.assertNotIn("SDPBackend.EFFICIENT_ATTENTION", source)


if __name__ == "__main__":
    unittest.main()
