"""Contracts for releasing transformer-block intermediates at last use."""

import ast
import hashlib
import re
import unittest
from pathlib import Path


MODEL = Path(__file__).parents[1] / "pangu_profile_model.py"
BASELINE_WITHOUT_DELETES_SHA256 = (
    "5cc79b27fdf0b586fef322ab4f1e1e2a8af1ed6f93d3b2644c10b8a6824b47a9"
)


def load_forward_node():
    tree = ast.parse(MODEL.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_chunked_mlp_block"
    )


class StripDeletes(ast.NodeTransformer):
    def visit_Delete(self, node):
        return None


class StripDirectResidualCandidate(ast.NodeTransformer):
    """Recover the pre-candidate fallback AST for lifecycle regression checks."""

    def visit_Assign(self, node):
        if any(
            isinstance(target, ast.Name) and target.id == "direct_residual"
            for target in node.targets
        ):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Name) and node.test.id == "direct_residual":
            return node.orelse
        return node


class BlockIntermediateReleaseTests(unittest.TestCase):
    def test_default_fallback_arithmetic_ast_stays_locked(self):
        node = StripDirectResidualCandidate().visit(load_forward_node())
        node = StripDeletes().visit(node)
        ast.fix_missing_locations(node)
        payload = ast.dump(node, annotate_fields=True, include_attributes=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        self.assertEqual(digest, BASELINE_WITHOUT_DELETES_SHA256)

    def test_partition_and_attention_inputs_are_released_at_last_use(self):
        source = ast.get_source_segment(
            MODEL.read_text(encoding="utf-8"), load_forward_node()
        )
        partition_calls = [
            match.start()
            for match in re.finditer(r"x_windows = window_partition\(", source)
        ]
        roll_call = source.index("shifted_x = torch.roll(")
        release_partition_inputs = [
            match.start() for match in re.finditer(r"del x\n", source)
        ]
        release_shifted_input = source.index("del shifted_x")
        attention_call = source.index("attn_windows = self.attn(")
        release_attention_input = source.index("del x_windows")
        reshape_attention_output = source.index(
            "attn_windows = attn_windows.view("
        )

        self.assertEqual(len(partition_calls), 2)
        self.assertEqual(len(release_partition_inputs), 2)
        self.assertLess(roll_call, release_partition_inputs[0])
        self.assertLess(release_partition_inputs[0], partition_calls[0])
        self.assertLess(partition_calls[1], release_partition_inputs[1])
        self.assertLess(max(partition_calls), release_shifted_input)
        self.assertLess(release_shifted_input, attention_call)
        self.assertLess(attention_call, release_attention_input)
        self.assertLess(release_attention_input, reshape_attention_output)

    def test_reverse_inputs_and_temporary_alias_are_released_at_last_use(self):
        source = ast.get_source_segment(
            MODEL.read_text(encoding="utf-8"), load_forward_node()
        )
        reverse_calls = [
            match.start()
            for match in re.finditer(r"shifted_x = window_reverse\(", source)
        ]
        release_reverse_inputs = [
            match.start()
            for match in re.finditer(r"del attn_windows", source)
        ]
        roll_output = source.index("x = torch.roll(", reverse_calls[0])
        direct_output = source.index("x = shifted_x", reverse_calls[1])
        release_reverse_temporary = source.rindex("del shifted_x")
        crop = source.index("x = crop3d(")

        self.assertEqual(len(reverse_calls), 2)
        self.assertEqual(len(release_reverse_inputs), 2)
        self.assertLess(reverse_calls[0], release_reverse_inputs[0])
        self.assertLess(release_reverse_inputs[0], roll_output)
        self.assertLess(reverse_calls[1], release_reverse_inputs[1])
        self.assertLess(release_reverse_inputs[1], direct_output)
        self.assertLess(roll_output, release_reverse_temporary)
        self.assertLess(direct_output, release_reverse_temporary)
        self.assertLess(release_reverse_temporary, crop)


if __name__ == "__main__":
    unittest.main()
