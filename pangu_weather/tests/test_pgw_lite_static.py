"""Static checks for PGW-Lite profile, quantization, and inference wiring.

These tests avoid importing PyTorch or OneScience so they can run locally on
machines without the competition runtime.
"""

import ast
import unittest
from pathlib import Path


PANGU_WEATHER = Path(__file__).parents[1]
REPO = PANGU_WEATHER.parents[0]
CONFIG = PANGU_WEATHER / "conf" / "config.yaml"
INFERENCE = PANGU_WEATHER / "inference.py"
QUANTIZE = PANGU_WEATHER / "scripts" / "quantize_linear.py"
DISTILL = PANGU_WEATHER / "distill_train.py"
PROFILE_MODEL = PANGU_WEATHER / "pangu_profile_model.py"


class PGWLiteStaticTests(unittest.TestCase):
    def test_config_declares_isolated_pgw_lite_profile_and_checkpoints(self):
        config = CONFIG.read_text(encoding="utf-8")
        self.assertIn("student_profiles:", config)
        self.assertIn("full_192:", config)
        self.assertIn("student_160:", config)
        self.assertIn("pgw_lite_patch8:", config)
        self.assertIn("patch_size: [2, 8, 8]", config)
        self.assertIn("pgw_lite_pruned_96_patch16_full:", config)
        self.assertIn("pgw_lite_pruned_96_patch16_mid:", config)
        self.assertIn("patch_size: [2, 16, 16]", config)
        self.assertIn("depth_blocks: [1, 4, 4, 1]", config)
        self.assertIn('pgw_lite_distilled_checkpoint: "model_pgw_lite_fp16.pth"', config)
        self.assertIn('pgw_lite_quantized_checkpoint: "model_pgw_lite_quantized.pth"', config)
        self.assertIn("distill_teacher_weight: 0.5", config)
        self.assertIn("distill_hint_weight: 0.2", config)
        self.assertIn('distill_hint_layers: ["layer1", "layer2"]', config)

    def test_submission_helper_patches_pgw_lite_recovery(self):
        source = PROFILE_MODEL.read_text(encoding="utf-8")
        self.assertIn("from onescience.models.pangu import Pangu", source)
        self.assertIn("from onescience.modules import OneRecovery", source)
        self.assertIn("if patch_size != [2, 4, 4]:", source)
        self.assertIn("patch_size=tuple(patch_size[1:])", source)
        self.assertIn("patch_size=tuple(patch_size)", source)
        recovery_region = source.split("model.patchrecovery2d = OneRecovery", 1)[1]
        self.assertNotIn("patch_size=(4, 4),", recovery_region)
        self.assertNotIn("patch_size=(2, 4, 4),", recovery_region)

    def test_inference_supports_profile_metadata_and_per_channel_dequant(self):
        source = INFERENCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("_infer_profile_from_state", source)
        self.assertIn("_dequantize_state_dict", source)
        self.assertIn("PANGU_USE_PGW_LITE", source)
        self.assertIn("pgw_lite_quantized_checkpoint", source)
        self.assertIn("scale.view(-1, 1)", source)
        self.assertIn("build_pangu_model", source)
        self.assertIn("model_profile['patch_size']", source)
        self.assertIn("PANGU_RECOMPUTE_SKIP", source)
        self.assertIn("recompute_skip=", source)

    def test_profile_model_can_enable_skip_recompute_forward(self):
        source = PROFILE_MODEL.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("def _forward_recompute_skip", source)
        self.assertIn("def enable_skip_recompute", source)
        self.assertIn("model.forward = types.MethodType", source)
        self.assertIn("recompute_skip=False", source)
        self.assertIn("skip_sequence = self.layer1(skip_sequence)", source)

    def test_quantizer_outputs_per_channel_versioned_metadata(self):
        source = QUANTIZE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("PANGU_QUANTIZE_PROFILE", source)
        self.assertIn("build_pangu_model", source)
        self.assertIn("quantize_per_output_channel", source)
        self.assertIn('"scheme": "per_channel_int8"', source)
        self.assertIn('"model_profile": profile', source)
        self.assertIn("pgw_lite_quantized_checkpoint", source)

    def test_distillation_can_select_pgw_lite_profile(self):
        source = DISTILL.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("PANGU_STUDENT_PROFILE", source)
        self.assertIn("build_pangu_model", source)
        self.assertIn("pgw_lite_distilled_checkpoint", source)
        self.assertIn("load_compatible_state", source)
        self.assertIn('"model_profile": student_profile', source)
        self.assertIn("PANGU_DISTILL_INIT_CHECKPOINT", source)
        self.assertIn("resolve_checkpoint_arg", source)


if __name__ == "__main__":
    unittest.main()
