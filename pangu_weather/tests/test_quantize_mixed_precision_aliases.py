import ast
import unittest
from collections import OrderedDict
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "quantize_mixed_precision.py"


def load_helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    wanted = {
        "quantize_per_output_channel",
        "canonical_fuser_key",
        "quantize_state_dict",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


class MixedPrecisionAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_helpers()

    def test_quantized_alias_survives_keep_later_compaction(self):
        weight = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        state = OrderedDict(
            (
                ("layer1.fuser.linear.weight", weight),
                ("layer1.Fuser.linear.weight", weight),
            )
        )
        mixed, quantized_count, fp16_count = self.helpers["quantize_state_dict"](
            state,
            {"layer1.fuser.linear.weight"},
            set(),
        )
        self.assertEqual((quantized_count, fp16_count), (1, 0))
        self.assertEqual(mixed["layer1.Fuser.linear.weight"].dtype, torch.int8)
        self.assertIn("layer1.Fuser.linear.weight_scale", mixed)
        self.assertIs(
            mixed["layer1.fuser.linear.weight"],
            mixed["layer1.Fuser.linear.weight"],
        )

    def test_fp16_keep_count_is_per_logical_linear(self):
        weight = torch.ones(2, 2)
        state = OrderedDict(
            (
                ("layer1.fuser.linear.weight", weight),
                ("layer1.Fuser.linear.weight", weight),
            )
        )
        mixed, quantized_count, fp16_count = self.helpers["quantize_state_dict"](
            state,
            {"layer1.fuser.linear.weight"},
            {"layer1.fuser.linear"},
        )
        self.assertEqual((quantized_count, fp16_count), (0, 1))
        self.assertEqual(mixed["layer1.Fuser.linear.weight"].dtype, torch.float16)

    def test_script_preserves_available_provenance_metadata(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '("distillation", "pruning", "initialization")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
