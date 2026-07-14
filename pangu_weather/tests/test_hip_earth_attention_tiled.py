"""Static contract for the single submitted P2/HIP implementation."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
P2 = ROOT / "p2_tiled_attention.py"
WRAPPER = ROOT / "hip_earth_attention_tiled.py"
KERNEL = ROOT / "hip_kernels" / "earth_attention_tiled_fwd.hip"


class FixedP2HipTests(unittest.TestCase):
    def test_python_adapter_is_fixed_to_full_row_fast(self):
        source = P2.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('mode="full-row-fast"', source)
        wrapper = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("score_stride = 144", wrapper)
        self.assertIn("qk_tile = 16", wrapper)
        for forbidden in ("debug", "rollback", "online", "expf", "qk32"):
            self.assertNotIn(forbidden, source.lower())

    def test_kernel_contains_scored_fast_path(self):
        source = KERNEL.read_text(encoding="utf-8")
        for required in (
            "PANGU_FULL_ROW_SCORE_STRIDE",
            "PANGU_FULL_ROW_QK_TILE",
            "PANGU_FULL_ROW_DIRECT_SCORE_STORE",
            "PANGU_FULL_ROW_PV_DOUBLE_BUFFER",
            "pangu_earth_attention_tiled_full_row_fwd_fp16",
        ):
            self.assertIn(required, source)

    def test_wrapper_and_kernel_parse_or_exist(self):
        ast.parse(WRAPPER.read_text(encoding="utf-8"))
        self.assertTrue(KERNEL.is_file())


if __name__ == "__main__":
    unittest.main()
