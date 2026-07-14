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
    "_verified_pruned96_generated_mask_keys",
    "_runtime_alias_keys",
    "_resolve_pruned96_state_load_contract",
    "_validate_pruned96_state_load",
}
HELPER_GLOBALS = {"_PRUNED96_GENERATED_MASK_LAYOUT"}
TIMER_SHA256 = "fa7d46a8ea3a3da93f5348bbb6b237409da16a68b20708331d4d9b0f4adb61ad"


def load_helpers():
    source = INFERENCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in HELPERS
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in HELPER_GLOBALS
                for target in node.targets
            )
        )
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


class StaticStateModel:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class Pruned96LoadContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_helpers()
        cls.production_layout = cls.helpers["_PRUNED96_GENERATED_MASK_LAYOUT"]
        outer = torch.tensor(
            [[[[0.0, -100.0], [-100.0, 0.0]]]],
            dtype=torch.float16,
        )
        inner = torch.tensor(
            [[[[0.0, -100.0, 0.0], [-100.0, 0.0, -100.0]]]],
            dtype=torch.float16,
        )
        tensor_sha256 = cls.helpers["_tensor_sha256"]
        outer_digest = tensor_sha256(outer)
        inner_digest = tensor_sha256(inner)
        cls.test_masks = {
            outer_digest: outer,
            inner_digest: inner,
        }
        cls.helpers["_PRUNED96_GENERATED_MASK_LAYOUT"] = (
            ("layer1", (1,), tuple(outer.shape), outer_digest),
            ("layer2", (1, 3, 5), tuple(inner.shape), inner_digest),
            ("layer3", (1, 3, 5), tuple(inner.shape), inner_digest),
            ("layer4", (1,), tuple(outer.shape), outer_digest),
        )

    def setUp(self):
        self.model_state = OrderedDict(AliasModel().state_dict())
        self.mask_keys = set()
        for stage, block_indices, _shape, digest in self.helpers[
            "_PRUNED96_GENERATED_MASK_LAYOUT"
        ]:
            for block_index in block_indices:
                value = self.test_masks[digest].clone()
                for alias in ("Fuser", "fuser"):
                    key = (
                        f"{stage}.{alias}.blocks.{block_index}."
                        "transformer.attn_mask"
                    )
                    self.model_state[key] = value
                    self.mask_keys.add(key)
        self.model = StaticStateModel(self.model_state)
        self.upper_weight = "layer.Fuser.weight"
        self.upper_index = "layer.Fuser.earth_position_index"
        self.lower_weight = "layer.fuser.weight"
        self.lower_index = "layer.fuser.earth_position_index"

    def baseline_state(self):
        return OrderedDict(
            (key, value.clone())
            for key, value in self.model_state.items()
            if ".Fuser." in key and not key.endswith("attn_mask")
        )

    def test_production_constructor_mask_layout_is_frozen(self):
        self.assertEqual(
            self.production_layout,
            (
                (
                    "layer1",
                    (1,),
                    (15, 64, 144, 144),
                    "39ee00633b54a104ae928d7724a72afd84490b9067d05f878b0664baa5de1b07",
                ),
                (
                    "layer2",
                    (1, 3, 5),
                    (8, 32, 144, 144),
                    "7064b0fd983ea8966be281f089673fe86ee57bdee4ae9e8587f064887ae2f36c",
                ),
                (
                    "layer3",
                    (1, 3, 5),
                    (8, 32, 144, 144),
                    "7064b0fd983ea8966be281f089673fe86ee57bdee4ae9e8587f064887ae2f36c",
                ),
                (
                    "layer4",
                    (1,),
                    (15, 64, 144, 144),
                    "39ee00633b54a104ae928d7724a72afd84490b9067d05f878b0664baa5de1b07",
                ),
            ),
        )

    def test_alias_compacted_baseline_allows_only_proven_runtime_aliases(self):
        state = self.baseline_state()
        allowed, report = self.helpers["_resolve_pruned96_state_load_contract"](
            self.model,
            {},
            state,
        )
        expected = {self.lower_weight, self.lower_index} | self.mask_keys
        self.assertEqual(allowed, expected)
        self.assertEqual(report["elided_index_keys"], [])
        self.assertEqual(set(report["constructor_mask_missing_keys"]), self.mask_keys)
        self.assertEqual(report["constructor_mask_owner_count"], 8)
        self.helpers["_validate_pruned96_state_load"](
            expected,
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
        expected = {
            self.lower_weight,
            self.lower_index,
            self.upper_index,
        } | self.mask_keys
        self.assertEqual(allowed, expected)
        self.assertEqual(
            set(report["elided_index_keys"]),
            {self.lower_index, self.upper_index},
        )
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
            state = self.baseline_state()
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

    def test_constructor_masks_reject_source_topology_shape_alias_and_values(self):
        resolve = self.helpers["_resolve_pruned96_state_load_contract"]
        source = self.baseline_state()
        mask_key = sorted(self.mask_keys)[0]

        source_with_mask = OrderedDict(source)
        source_with_mask[mask_key] = self.model_state[mask_key].clone()
        with self.assertRaisesRegex(ValueError, "must omit constructor masks"):
            resolve(self.model, {}, source_with_mask)

        missing_state = OrderedDict(self.model_state)
        missing_state.pop(mask_key)
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            resolve(StaticStateModel(missing_state), {}, source)

        upper_key = mask_key.replace(".fuser.", ".Fuser.")
        lower_key = upper_key.replace(".Fuser.", ".fuser.")
        wrong_shape_state = OrderedDict(self.model_state)
        wrong_shape = torch.zeros(1, dtype=torch.float16)
        wrong_shape_state[upper_key] = wrong_shape
        wrong_shape_state[lower_key] = wrong_shape
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            resolve(StaticStateModel(wrong_shape_state), {}, source)

        broken_alias_state = OrderedDict(self.model_state)
        broken_alias_state[lower_key] = broken_alias_state[upper_key].clone()
        with self.assertRaisesRegex(ValueError, "does not share storage"):
            resolve(StaticStateModel(broken_alias_state), {}, source)

        wrong_value_state = OrderedDict(self.model_state)
        wrong_value = wrong_value_state[upper_key].clone()
        wrong_value.view(-1)[0] = 1.0
        wrong_value_state[upper_key] = wrong_value
        wrong_value_state[lower_key] = wrong_value
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            resolve(StaticStateModel(wrong_value_state), {}, source)

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
