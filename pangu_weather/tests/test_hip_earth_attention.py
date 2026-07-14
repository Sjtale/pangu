import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "hip_earth_attention.py"
KERNEL = ROOT / "hip_kernels" / "earth_attention_fwd.hip"
PROBE = ROOT / "scripts" / "probe_hip_earth_attention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hip_earth_attention", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HipEarthAttentionTests(unittest.TestCase):
    def test_validation_accepts_strided_pangu_layout(self):
        module = _load_module()
        qkv_base = torch.randn(2, 4, 8, 3, 3, 32, dtype=torch.float16)
        qkv = qkv_base.permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        earth = torch.randn(1, 3, 4, 8, 8, dtype=torch.float16)
        shifted = torch.zeros(2, 4, 8, 8, dtype=torch.float16)
        normalized_earth, normalized_mask, shape = module._validate_inputs(
            q, k, v, earth, shifted
        )
        self.assertEqual(shape, (2, 3, 4, 8, 32))
        self.assertEqual(tuple(normalized_earth.shape), (3, 4, 8, 8))
        self.assertEqual(tuple(normalized_mask.shape), (2, 4, 8, 8))
        self.assertFalse(q.is_contiguous())
        self.assertEqual(q.stride(), (9216, 32, 2304, 288, 1))

    def test_validation_rejects_unsupported_geometry(self):
        module = _load_module()
        q = torch.randn(1, 1, 1, 257, 32, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "at most 256"):
            module._validate_inputs(
                q,
                q,
                q,
                torch.zeros(1, 1, 1, 257, 257, dtype=torch.float16),
                torch.zeros(1, 1, 257, 257, dtype=torch.float16),
            )

    def test_kernel_fuses_bias_softmax_and_pv(self):
        source = KERNEL.read_text(encoding="utf-8")
        self.assertIn('extern "C" int pangu_earth_attention_fwd_fp16', source)
        self.assertIn("__half2float(earth_bias[earth_offset])", source)
        self.assertIn("__half2float(shifted_mask[mask_offset])", source)
        self.assertIn("expf(score - row_max)", source)
        self.assertIn("accumulator += scratch[key_index]", source)
        self.assertIn("hipLaunchKernelGGL", source)

    def test_build_is_local_and_probe_covers_l32_l144(self):
        wrapper = MODULE.read_text(encoding="utf-8")
        probe = PROBE.read_text(encoding="utf-8")
        self.assertIn('"/opt/dtk/bin/hipcc"', wrapper)
        self.assertIn('"--shared"', wrapper)
        self.assertIn('"logs" / "hip_earth_attention_build"', wrapper)
        self.assertNotIn("pip install", wrapper)
        self.assertNotIn("conda", wrapper)
        self.assertIn('"small_l32"', probe)
        self.assertIn('"representative_l144"', probe)
        self.assertIn("subprocess.run", probe)
        self.assertIn("preexec_fn=_disable_core_dumps", probe)
        self.assertIn('"signal": signal_name', probe)


if __name__ == "__main__":
    unittest.main()
