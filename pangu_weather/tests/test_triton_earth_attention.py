import importlib.util
import unittest
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[1] / "triton_earth_attention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("triton_earth_attention", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TritonEarthAttentionTests(unittest.TestCase):
    def test_accepts_real_model_layout_without_contiguous_copy(self):
        module = _load_module()
        q = torch.randn(3, 3, 4, 8, 32, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        earth = torch.randn(1, 3, 4, 8, 8, dtype=torch.float16)
        shifted = torch.zeros(3, 4, 8, 8, dtype=torch.float16)
        normalized_earth, normalized_mask, shape = module._validate_inputs(
            q, k, v, earth, shifted
        )
        self.assertEqual(shape, (3, 3, 4, 8, 32))
        self.assertEqual(tuple(normalized_earth.shape), (3, 4, 8, 8))
        self.assertEqual(tuple(normalized_mask.shape), (3, 4, 8, 8))
        self.assertEqual(
            normalized_earth.untyped_storage().data_ptr(),
            earth.untyped_storage().data_ptr(),
        )

    def test_rejects_non_pangu_head_dimension(self):
        module = _load_module()
        q = torch.randn(1, 1, 1, 4, 16, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "head_dim=32"):
            module._validate_inputs(
                q,
                q,
                q,
                torch.zeros(1, 1, 1, 4, 4, dtype=torch.float16),
                torch.zeros(1, 1, 4, 4, dtype=torch.float16),
            )

    def test_kernel_is_direct_triton_and_uses_online_softmax(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("@triton.jit", source)
        self.assertIn("tl.dot", source)
        self.assertIn("running_max", source)
        self.assertIn("running_sum", source)
        self.assertNotIn("tl.trans", source)
        self.assertNotIn("torch.compile(", source)


if __name__ == "__main__":
    unittest.main()
