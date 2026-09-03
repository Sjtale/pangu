"""Tests for lossless OneFuser checkpoint alias compaction."""

import importlib.util
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT = Path(__file__).parents[1] / "scripts" / "compact_fuser_alias_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("compact_fuser_alias_checkpoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AliasWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuser = nn.Linear(4, 3)
        self.Fuser = self.fuser


class AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = AliasWrapper()
        self.unique = nn.Linear(3, 2)


class CompactFuserAliasTests(unittest.TestCase):
    def make_source_state(self):
        target = AliasModel().state_dict()
        source = OrderedDict()
        for key, value in target.items():
            if ".fuser." in key and key.endswith("weight"):
                source[key] = torch.zeros_like(value, dtype=torch.int8)
                source[key + "_scale"] = torch.ones(value.shape[0], 1, dtype=torch.float16)
            else:
                source[key] = value.clone()
        return target, source

    def test_plan_drops_earlier_alias_and_scale(self):
        target, source = self.make_source_state()
        pairs, drop_keys = MODULE.plan_alias_compaction(source)
        MODULE.validate_runtime_aliases(target, pairs)
        compacted = MODULE.compact_state_dict(source, drop_keys)

        self.assertEqual(pairs, [("layer.fuser.weight", "layer.Fuser.weight"),
                                 ("layer.fuser.bias", "layer.Fuser.bias")])
        self.assertNotIn("layer.fuser.weight", compacted)
        self.assertNotIn("layer.fuser.weight_scale", compacted)
        self.assertNotIn("layer.fuser.bias", compacted)
        self.assertIn("layer.Fuser.weight", compacted)
        self.assertIn("unique.weight", compacted)

    def test_compacted_load_matches_current_final_writer(self):
        _, source = self.make_source_state()
        pairs, drop_keys = MODULE.plan_alias_compaction(source)
        compacted = MODULE.compact_state_dict(source, drop_keys)

        current = AliasModel()
        candidate = AliasModel()
        with torch.no_grad():
            for key, value in source.items():
                if key.endswith("_scale"):
                    continue
                target = current.state_dict()[key]
                loaded = value.to(target.dtype)
                target.copy_(loaded)
            for key, value in compacted.items():
                candidate.state_dict()[key].copy_(value)

        for current_value, candidate_value in zip(
            current.state_dict().values(), candidate.state_dict().values()
        ):
            self.assertTrue(torch.equal(current_value, candidate_value))

    def test_saved_candidate_preserves_kept_values(self):
        _, source = self.make_source_state()
        _, drop_keys = MODULE.plan_alias_compaction(source)
        compacted = MODULE.compact_state_dict(source, drop_keys)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pth"
            torch.save({"model_state_dict": compacted}, path)
            MODULE.verify_saved_candidate(compacted, str(path))

    def test_rejects_reversed_writer_order(self):
        _, source = self.make_source_state()
        reversed_state = OrderedDict(reversed(list(source.items())))
        with self.assertRaisesRegex(ValueError, "Final alias writer"):
            MODULE.plan_alias_compaction(reversed_state)

    def test_quantization_metadata_requires_actual_kept_int8_and_scale(self):
        state = OrderedDict(
            (
                ("layer.Fuser.weight", torch.ones(2, 2, dtype=torch.int8)),
                ("layer.Fuser.weight_scale", torch.ones(2, 1, dtype=torch.float16)),
            )
        )
        MODULE.validate_quantized_state(
            {"quantization": {"quantized_keys_count": 1}}, state
        )

        with self.assertRaisesRegex(ValueError, "declared representation"):
            MODULE.validate_quantized_state(
                {"quantization": {"quantized_keys_count": 1}},
                OrderedDict((('layer.Fuser.weight', torch.ones(2, 2)),)),
            )

    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"pangu-alias-candidate")
            self.assertEqual(
                MODULE.sha256_file(path),
                "f1f775c45e334d98de78294f1a0610b290f29b51da79d715366f19a65ca9808d",
            )


if __name__ == "__main__":
    unittest.main()
