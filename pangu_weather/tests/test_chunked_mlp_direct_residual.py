"""Static guards for the opt-in direct-residual chunked-MLP candidate."""

import ast
import unittest
from pathlib import Path


MODEL = Path(__file__).parents[1] / "pangu_profile_model.py"


def load_forward_source():
    tree = ast.parse(MODEL.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_chunked_mlp_block"
    )
    return ast.get_source_segment(MODEL.read_text(encoding="utf-8"), node)


class ChunkedMlpDirectResidualStaticTests(unittest.TestCase):
    def test_direct_path_is_explicitly_opt_in_and_inference_only(self):
        source = load_forward_source()
        self.assertIn("_pangu_mlp_direct_residual", source)
        self.assertIn("elif direct_residual:", source)
        self.assertIn("if self.training:", source)
        self.assertIn("PANGU_MLP_DIRECT_RESIDUAL is inference-only", source)

    def test_direct_path_accumulates_each_chunk_and_releases_it(self):
        source = load_forward_source()
        direct_path = source.split("elif direct_residual:", 1)[1].split(
            "else:\n        x_mlp", 1
        )[0]
        self.assertIn("x_chunk = x[:, start:end]", direct_path)
        self.assertIn("mlp_chunk = self.mlp(self.norm2(x_chunk))", direct_path)
        self.assertIn("x_chunk.add_(self.drop_path(mlp_chunk))", direct_path)
        self.assertIn("del mlp_chunk", direct_path)
        self.assertIn("del x_chunk", direct_path)
        self.assertNotIn("x_mlp = x.new_empty", direct_path)

    def test_builder_keeps_the_candidate_default_off(self):
        source = MODEL.read_text(encoding="utf-8")
        self.assertIn(
            'direct_mlp_residual = _is_enabled("PANGU_MLP_DIRECT_RESIDUAL")',
            source,
        )
        self.assertIn(
            "direct_residual_accumulation=direct_mlp_residual",
            source,
        )


if __name__ == "__main__":
    unittest.main()
