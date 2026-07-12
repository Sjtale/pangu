"""Tests for packed groupwise INT4 checkpoint storage."""

import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "pack_groupwise_int4.py"
SPEC = importlib.util.spec_from_file_location("pack_groupwise_int4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GroupwiseInt4Tests(unittest.TestCase):
    def test_pack_round_trip_shape_and_error(self):
        torch.manual_seed(20260711)
        weight = torch.randn(7, 130)
        source_scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        qweight = torch.round(weight / source_scale).clamp(-127, 127).to(torch.int8)
        packed, scales = MODULE.pack_groupwise_int4(qweight, source_scale, 64)
        restored = MODULE.unpack_groupwise_int4(packed, scales, weight.shape, 64)
        self.assertEqual(tuple(packed.shape), (7, 96))
        self.assertEqual(tuple(restored.shape), tuple(weight.shape))
        self.assertLess((restored - weight).abs().mean().item(), 0.12)

    def test_int4_storage_is_smaller_than_int8_plus_scale(self):
        qweight = torch.randint(-127, 128, (64, 128), dtype=torch.int8)
        source_scale = torch.ones(64, 1, dtype=torch.float16)
        packed, scales = MODULE.pack_groupwise_int4(qweight, source_scale, 64)
        source_bytes = qweight.numel() + source_scale.numel() * 2
        packed_bytes = packed.numel() + scales.numel() * 2
        self.assertLess(packed_bytes, source_bytes * 0.6)


if __name__ == "__main__":
    unittest.main()
