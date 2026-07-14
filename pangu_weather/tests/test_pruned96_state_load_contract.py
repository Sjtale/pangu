"""Fail-closed contracts for the scored pruned96 checkpoint loader."""

import ast
import hashlib
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


INFERENCE = Path(__file__).parents[1] / "inference.py"
HELPERS = {
    "_tensor_sha256",
    "_runtime_alias_keys",
    "_resolve_pruned96_state_load_contract",
    "_validate_pruned96_state_load",
}
TIMER_SHA256 = "fa7d46a8ea3a3da93f5348bbb6b237409da16a68b20708331d4d9b0f4adb61ad"


def load_helpers():
    source = INFERENCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]
    namespace = {"hashlib": hashlib, "torch": torch}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(INFERENCE), "exec"), namespace)
    return namespace


class IndexedBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.register_buffer(
            "earth_position_index",
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        )


class AliasOwner(nn.Module):
    def __init__(self):
        super().__init__()
        body = IndexedBody()
        self.fuser = body
        self.Fuser = body


class AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = AliasOwner()


class Pruned96LoadContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_helpers()

    def setUp(self):
        self.model = AliasModel()
        self.model_state = self.model.state_dict()
        self.upper_weight = "layer.Fuser.weight"
        self.upper_index = "layer.Fuser.earth_position_index"
        self.lower_weight = "layer.fuser.weight"
        self.lower_index = "layer.fuser.earth_position_index"

    def test_alias_compacted_baseline_allows_only_proven_runtime_aliases(self):
        state = OrderedDict(
            (key, value.clone())
            for key, value in self.model_state.items()
            if ".Fuser." in key
        )
        allowed, report = self.helpers["_resolve_pruned96_state_load_contract"](
            self.model,
            {},
            state,
        )
        self.assertEqual(allowed, {self.lower_weight, self.lower_index})
        self.assertEqual(report["elided_index_keys"], [])
        self.helpers["_validate_pruned96_state_load"](
            [self.lower_weight, self.lower_index],
            [],
            allowed,
        )

        allowed_with_unrelated_metadata, _ = self.helpers[
            "_resolve_pruned96_state_load_contract"
        ](
            self.model,
            {"storage_optimization": {"existing_marker": "preserve"}},
            state,
        )
        self.assertEqual(allowed_with_unrelated_metadata, allowed)

    def make_elided_checkpoint(self):
        tensor_sha256 = self.helpers["_tensor_sha256"]
        expected = [self.lower_index, self.upper_index]
        generated = {
            key: {
                "shape": list(self.model_state[key].shape),
                "dtype": str(self.model_state[key].dtype),
                "sha256": tensor_sha256(self.model_state[key]),
            }
            for key in expected
        }
        return {
            "storage_optimization": {
                "schema_version": 1,
                "deterministic_buffer_elision": {
                    "method": "constructor-earth-position-index-v1",
                    "removed_checkpoint_keys": [self.upper_index],
                    "expected_runtime_missing_keys": expected,
                    "removed_logical_bytes": (
                        self.model_state[self.upper_index].numel()
                        * self.model_state[self.upper_index].element_size()
                    ),
                    "generated_indices": generated,
                },
            }
        }

    def test_elided_indices_are_verified_and_added_to_exact_allowlist(self):
        checkpoint = self.make_elided_checkpoint()
        state = OrderedDict(
            [(self.upper_weight, self.model_state[self.upper_weight].clone())]
        )
        allowed, report = self.helpers["_resolve_pruned96_state_load_contract"](
            self.model,
            checkpoint,
            state,
        )
        expected = {self.lower_weight, self.lower_index, self.upper_index}
        self.assertEqual(allowed, expected)
        self.assertEqual(set(report["elided_index_keys"]), expected - {self.lower_weight})
        self.helpers["_validate_pruned96_state_load"](expected, [], allowed)

    def test_wrong_generated_digest_is_rejected(self):
        checkpoint = self.make_elided_checkpoint()
        checkpoint["storage_optimization"]["deterministic_buffer_elision"][
            "generated_indices"
        ][self.upper_index]["sha256"] = "0" * 64
        state = OrderedDict(
            [(self.upper_weight, self.model_state[self.upper_weight].clone())]
        )
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.helpers["_resolve_pruned96_state_load_contract"](
                self.model,
                checkpoint,
                state,
            )

    def test_quantized_tensors_and_metadata_are_rejected_before_loading(self):
        for key, value in (
            ("layer.Fuser.weight_scale", torch.ones(2)),
            ("layer.Fuser.weight", torch.ones(2, 3, dtype=torch.int8)),
            ("layer.Fuser.weight.int4_group_size", torch.tensor(32)),
        ):
            state = OrderedDict(
                (name, tensor.clone())
                for name, tensor in self.model_state.items()
                if ".Fuser." in name
            )
            state[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError,
                "forbids undeclared quantized state",
            ):
                self.helpers["_resolve_pruned96_state_load_contract"](
                    self.model,
                    {},
                    state,
                )

    def test_missing_and_unexpected_keys_must_match_exactly(self):
        validate = self.helpers["_validate_pruned96_state_load"]
        with self.assertRaisesRegex(RuntimeError, "missing-key contract"):
            validate([self.lower_weight, "other"], [], {self.lower_weight})
        with self.assertRaisesRegex(RuntimeError, "unexpected keys"):
            validate([self.lower_weight], ["extra"], {self.lower_weight})

    def test_official_timer_block_is_byte_stable(self):
        source = INFERENCE.read_text(encoding="utf-8")
        loop = source.index("for batch_index, data in enumerate")
        marker = "#----------------------AI4S(时间度量不可更改)---------------------------"
        start = source.index(marker, loop)
        end_marker = "#---------------------------------------------------------------------"
        end = source.index(end_marker, start) + len(end_marker)
        actual = hashlib.sha256(source[start:end].encode()).hexdigest()
        self.assertEqual(actual, TIMER_SHA256)


if __name__ == "__main__":
    unittest.main()
