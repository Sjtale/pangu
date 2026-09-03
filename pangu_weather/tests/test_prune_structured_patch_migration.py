import ast
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


SCRIPT = Path(__file__).parents[1] / "scripts" / "prune_structured.py"


def load_resize_helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    wanted = {"_resize_patch_weight", "_interpolate_earth_bias"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"F": F}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


class PruneStructuredPatchMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.helpers = load_resize_helpers()

    def test_target_model_uses_target_profile_patch_size(self):
        self.assertIn("patch_size=target_patch_size", self.source)
        self.assertIn('"patch_size": target_patch_size', self.source)
        self.assertIn('"depth_blocks": list(target_depths)', self.source)
        self.assertIn('"window_size": [int(value) for value in cfg.window_size]', self.source)

    def test_patch4_kernel_resizes_to_patch8(self):
        value = torch.ones(2, 3, 1, 4, 4)
        resized = self.helpers["_resize_patch_weight"](
            value, (2, 3, 1, 8, 8), preserve_embedding_scale=True
        )
        self.assertEqual(tuple(resized.shape), (2, 3, 1, 8, 8))
        self.assertTrue(torch.allclose(resized, torch.full_like(resized, 0.25)))

    def test_earth_bias_resizes_window_axis(self):
        value = torch.arange(8.0).reshape(1, 8, 1)
        resized = self.helpers["_interpolate_earth_bias"](value, (1, 4, 1))
        self.assertEqual(tuple(resized.shape), (1, 4, 1))

    def test_grid_dependent_buffers_are_regenerated(self):
        self.assertIn('key.endswith(("attn_mask", "earth_position_index"))', self.source)


if __name__ == "__main__":
    unittest.main()
