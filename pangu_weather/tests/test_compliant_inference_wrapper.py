"""Static contract for the organizer inference entry point."""

import ast
import unittest
from pathlib import Path


INFERENCE = Path(__file__).parents[1] / "inference.py"


class InferenceContractTests(unittest.TestCase):
    @staticmethod
    def profile_validator():
        tree = ast.parse(INFERENCE.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PROFILE" for target in node.targets)
            )
            or (isinstance(node, ast.FunctionDef) and node.name == "validate_profile")
        ]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(INFERENCE), "exec"), namespace)
        return namespace["validate_profile"]

    def test_fixed_checkpoint_and_profile(self):
        source = INFERENCE.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('"model_fp16.pth"', source)
        self.assertIn('"name": "pgw_lite_pruned_96"', source)
        self.assertNotIn("compliant_inference_wrapper", source)
        self.assertNotIn("PANGU_FP16_CHECKPOINT", source)
        self.assertIn(
            'profile.get("depth_blocks", PROFILE["depth_blocks"])', source
        )
        self.assertIn('if distillation is not None:', source)

    def test_legacy_scored_profile_defaults_to_fixed_depth(self):
        validate = self.profile_validator()
        validate(
            {
                "name": "pgw_lite_pruned_96",
                "patch_size": [2, 8, 8],
                "embed_dim": 96,
                "num_heads": [3, 6, 6, 3],
                "window_size": [2, 6, 12],
            }
        )

    def test_wrong_profile_remains_rejected(self):
        validate = self.profile_validator()
        with self.assertRaises(ValueError):
            validate(
                {
                    "name": "pgw_lite_pruned_96",
                    "patch_size": [2, 8, 8],
                    "embed_dim": 128,
                    "num_heads": [4, 8, 8, 4],
                    "window_size": [2, 6, 12],
                }
            )

    def test_official_timer_contains_only_forward_and_sync(self):
        source = INFERENCE.read_text(encoding="utf-8")
        loop = source[source.index("for data in tqdm") :]
        start = loop.index("start_time = time.perf_counter()")
        forward = loop.index("out_surface, out_upper_air = model")
        synchronize = loop.index("torch.cuda.synchronize()")
        end = loop.index("end_time = time.perf_counter()")
        self.assertLess(start, forward)
        self.assertLess(forward, synchronize)
        self.assertLess(synchronize, end)
        self.assertLess(loop.index("model_input ="), start)
        self.assertGreater(loop.index("prediction = torch.cat"), end)

    def test_no_output_calibration_or_runtime_backend_switches(self):
        source = INFERENCE.read_text(encoding="utf-8")
        for forbidden in (
            "calibration_coeffs.npy",
            "calibration_affine.npz",
            "physics_mean_targets.npz",
            "onnx",
            "torch.compile",
            "PANGU_DISABLE",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
