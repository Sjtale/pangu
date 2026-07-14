import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
ONESCIENCE_SRC = REPOSITORY.parent / "onescience" / "src"


MODEL_PROBE = r"""
import json
import sys
import types

sys.modules.setdefault("s3fs", types.ModuleType("s3fs"))
timm = types.ModuleType("timm")
timm_layers = types.ModuleType("timm.layers")
timm_models = types.ModuleType("timm.models")
timm_swin = types.ModuleType("timm.models.swin_transformer")
timm_layers.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
timm_swin.SwinTransformerStage = type("SwinTransformerStage", (), {})
sys.modules.setdefault("timm", timm)
sys.modules.setdefault("timm.layers", timm_layers)
sys.modules.setdefault("timm.models", timm_models)
sys.modules.setdefault("timm.models.swin_transformer", timm_swin)

from pangu_weather.pangu_profile_model import (
    SELECTIVE_MLP_96_PARAMETER_COUNT,
    apply_mlp_ratio_blocks,
    build_pangu_model,
    selective_mlp_96_profile,
    validate_mlp_ratio_blocks,
)

profile = selective_mlp_96_profile()
mutated_profile = selective_mlp_96_profile()
mutated_profile["mlp_ratio_blocks"][0][0] = 99

validation_errors = {}
invalid_schedules = {
    "stage_count": profile["mlp_ratio_blocks"][:3],
    "stage_depth": [
        profile["mlp_ratio_blocks"][0],
        profile["mlp_ratio_blocks"][1][:-1],
        profile["mlp_ratio_blocks"][2],
        profile["mlp_ratio_blocks"][3],
    ],
    "negative_ratio": [
        profile["mlp_ratio_blocks"][0],
        [4, -2, 2, 2, 2, 2],
        profile["mlp_ratio_blocks"][2],
        profile["mlp_ratio_blocks"][3],
    ],
    "fractional_hidden": [
        profile["mlp_ratio_blocks"][0],
        [4, 1 / 7, 2, 2, 2, 2],
        profile["mlp_ratio_blocks"][2],
        profile["mlp_ratio_blocks"][3],
    ],
}
for label, schedule in invalid_schedules.items():
    try:
        validate_mlp_ratio_blocks(schedule, profile["depth_blocks"], profile["embed_dim"])
    except (TypeError, ValueError) as exc:
        validation_errors[label] = str(exc)
    else:
        validation_errors[label] = None

model = build_pangu_model(
    img_size=[721, 1440],
    patch_size=profile["patch_size"],
    embed_dim=profile["embed_dim"],
    num_heads=profile["num_heads"],
    window_size=profile["window_size"],
    depth_blocks=profile["depth_blocks"],
    mlp_ratio_blocks=profile["mlp_ratio_blocks"],
    use_swiglu=False,
    use_rmsnorm=False,
    use_gqa=False,
    share_deep_blocks=False,
    chunked_attention=False,
)
state = model.state_dict()
shape_keys = {
    "layer1": "layer1.Fuser.blocks.0.transformer.mlp.fc1.weight",
    "protected_layer2": "layer2.Fuser.blocks.0.transformer.mlp.fc1.weight",
    "reduced_layer2": "layer2.Fuser.blocks.1.transformer.mlp.fc1.weight",
    "reduced_layer3": "layer3.Fuser.blocks.5.transformer.mlp.fc1.weight",
    "layer4": "layer4.Fuser.blocks.1.transformer.mlp.fc1.weight",
}
before_invalid_apply = sum(parameter.numel() for parameter in model.parameters())
try:
    apply_mlp_ratio_blocks(
        model,
        invalid_schedules["stage_depth"],
        profile["embed_dim"],
    )
except ValueError as exc:
    invalid_apply_error = str(exc)
else:
    invalid_apply_error = None

print(json.dumps({
    "profile": profile,
    "fresh_first_ratio": selective_mlp_96_profile()["mlp_ratio_blocks"][0][0],
    "expected_parameter_count": SELECTIVE_MLP_96_PARAMETER_COUNT,
    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    "parameter_count_before_invalid_apply": before_invalid_apply,
    "replaced": model._pangu_mlp_ratio_replaced,
    "applied_schedule": [list(stage) for stage in model._pangu_mlp_ratio_blocks],
    "shapes": {label: list(state[key].shape) for label, key in shape_keys.items()},
    "has_native_fc_keys": all(
        key in state and key.replace(".fc1.", ".fc2.") in state
        for key in shape_keys.values()
    ),
    "has_swiglu_keys": any(".mlp.w1." in key for key in state),
    "validation_errors": validation_errors,
    "invalid_apply_error": invalid_apply_error,
}))
"""


class SelectiveMLP96ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        python_path = [str(REPOSITORY), str(ONESCIENCE_SRC)]
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path)
        env.update(
            {
                "PANGU_CHUNKED_MLP": "0",
                "PANGU_CHUNKED_ATTENTION": "0",
                "PANGU_USE_SWIGLU": "0",
                "PANGU_USE_RMSNORM": "0",
                "PANGU_USE_GQA": "0",
                "PANGU_SHARE_DEEP_BLOCKS": "0",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(MODEL_PROBE)],
            cwd=REPOSITORY,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "SelectiveMLP-96 model probe failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        cls.report = json.loads(completed.stdout.strip().splitlines()[-1])

    def test_profile_is_exact_and_returns_independent_copies(self):
        self.assertEqual(self.report["profile"]["name"], "selective_mlp96")
        self.assertEqual(self.report["profile"]["patch_size"], [2, 8, 8])
        self.assertEqual(self.report["profile"]["embed_dim"], 96)
        self.assertEqual(self.report["profile"]["num_heads"], [3, 6, 6, 3])
        self.assertEqual(self.report["profile"]["depth_blocks"], [2, 6, 6, 2])
        self.assertEqual(
            self.report["profile"]["mlp_ratio_blocks"],
            [
                [4, 4],
                [4, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [4, 4],
            ],
        )
        self.assertEqual(self.report["fresh_first_ratio"], 4)

    def test_validation_fails_closed_before_mutation(self):
        self.assertTrue(all(self.report["validation_errors"].values()))
        self.assertIn("must contain 6 ratios", self.report["invalid_apply_error"])
        self.assertEqual(
            self.report["parameter_count"],
            self.report["parameter_count_before_invalid_apply"],
        )

    def test_native_mlp_keys_and_selective_hidden_shapes_are_preserved(self):
        self.assertTrue(self.report["has_native_fc_keys"])
        self.assertFalse(self.report["has_swiglu_keys"])
        self.assertEqual(self.report["replaced"], 11)
        self.assertEqual(self.report["shapes"]["layer1"], [384, 96])
        self.assertEqual(self.report["shapes"]["protected_layer2"], [768, 192])
        self.assertEqual(self.report["shapes"]["reduced_layer2"], [384, 192])
        self.assertEqual(self.report["shapes"]["reduced_layer3"], [384, 192])
        self.assertEqual(self.report["shapes"]["layer4"], [384, 96])

    def test_exact_parameter_count(self):
        self.assertEqual(self.report["expected_parameter_count"], 14_768_265)
        self.assertEqual(self.report["parameter_count"], 14_768_265)


if __name__ == "__main__":
    unittest.main()
