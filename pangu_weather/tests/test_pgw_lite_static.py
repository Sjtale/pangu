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
        self.assertIn("_load_dequantized_state_dict_incremental", source)
        self.assertIn("_dequantize_tensor_for_model", source)
        self.assertIn('torch.load(path, map_location="cpu")', source)
        self.assertIn("PANGU_INCREMENTAL_STATE_LOAD", source)
        self.assertIn("class RuntimeQuantLinear", source)
        self.assertIn("_replace_quantized_linear_modules", source)
        self.assertIn("_load_runtime_quant_state_dict", source)
        self.assertIn("PANGU_RUNTIME_QUANT_LINEAR", source)
        self.assertIn("share_deep_blocks", source)
        self.assertIn("PANGU_USE_PGW_LITE", source)
        self.assertIn("pgw_lite_quantized_checkpoint", source)
        self.assertIn("scale.view(-1, 1)", source)
        self.assertIn("build_pangu_model", source)
        self.assertIn("model_profile['patch_size']", source)
        self.assertIn("PANGU_RECOMPUTE_SKIP", source)
        self.assertIn("recompute_skip=", source)
        self.assertIn("PANGU_LAYERWISE_INFERENCE", source)
        self.assertIn("PANGU_LAYERWISE_EMPTY_CACHE", source)
        self.assertIn("PANGU_DIRECT_RECOVERY", source)
        self.assertIn("PANGU_DIRECT_RECOVERY_WIDTH_CHUNK", source)
        self.assertIn("layerwise_inference=", source)
        self.assertIn("PANGU_CHUNKED_ATTENTION", source)
        self.assertIn("attention_chunk_size=", source)
        self.assertIn("PANGU_CHECKPOINT", source)
        self.assertIn("PANGU_AUTO_SCAN_CHECKPOINT", source)
        self.assertIn('os.environ.setdefault("PANGU_AUTO_SCAN_CHECKPOINT", "0")', source)
        self.assertIn('os.environ.setdefault("PANGU_DISABLE_CUDA_GRAPH", "0")', source)
        self.assertIn("_scan_checkpoint_path", source)
        self.assertIn("_detect_architecture_from_state", source)
        self.assertIn("_infer_gqa_group_size", source)
        self.assertIn("_layer_index_from_key", source)
        self.assertIn("kv_group_size=gqa_group_size", source)
        self.assertIn("share_deep_blocks=share_deep_blocks", source)

    def test_profile_model_can_enable_skip_recompute_forward(self):
        source = PROFILE_MODEL.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("def _forward_recompute_skip", source)
        self.assertIn("def enable_skip_recompute", source)
        self.assertIn("model.forward = types.MethodType", source)
        self.assertIn("recompute_skip=False", source)
        self.assertIn("skip_sequence = self.layer1(skip_sequence)", source)

    def test_profile_model_can_enable_layerwise_forward(self):
        source = PROFILE_MODEL.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("def _forward_layerwise", source)
        self.assertIn("def _run_fuser_layerwise", source)
        self.assertIn("def enable_layerwise_inference", source)
        self.assertIn("def enable_memory_efficient_forward", source)
        self.assertIn("def enable_chunked_attention", source)
        self.assertIn("def _forward_chunked_earth_attention_3d", source)
        self.assertIn("def enable_deep_block_sharing", source)
        self.assertIn("def _direct_patch_recovery", source)
        self.assertIn("def _direct_patch_unembed_chunk", source)
        self.assertIn("def _recover_surface", source)
        self.assertIn("PANGU_DIRECT_RECOVERY", source)
        self.assertIn("PANGU_SHARE_DEEP_BLOCKS", source)
        self.assertIn("layer2_to_layer3", source)
        self.assertIn("layerwise_inference=False", source)
        self.assertIn("layerwise_empty_cache=False", source)
        self.assertIn("model._layerwise_empty_cache", source)
        self.assertIn("chunked_attention=None", source)
        self.assertIn("PANGU_CHUNKED_ATTENTION", source)
        self.assertIn("Pangu.forward = _forward_memory_efficient", source)
        self.assertNotIn("EarthAttention3D.forward =", source)

    def test_gqa_uses_original_attention_mask_layout(self):
        source = PROFILE_MODEL.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("class GQAEarthAttention3D", source)
        self.assertIn("mask.unsqueeze(1).unsqueeze(0)", source)
        self.assertNotIn("scaled_dot_product_attention", source)

    def test_quantizer_outputs_per_channel_versioned_metadata(self):
        source = QUANTIZE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("PANGU_QUANTIZE_PROFILE", source)
        self.assertIn("build_pangu_model", source)
        self.assertIn("quantize_per_output_channel", source)
        self.assertIn('"scheme": "per_channel_int8"', source)
        self.assertIn('"model_profile": profile', source)
        self.assertIn("merge_source_profile_metadata", source)
        self.assertIn('clean_key.startswith("layer3.")', source)
        self.assertIn('"share_deep_blocks": profile.get("share_deep_blocks")', source)
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
        self.assertIn("dequantize_linear_weight_state", source)
        self.assertIn("average_layer2_layer3_for_sharing", source)
        self.assertIn("PANGU_SHARE_DEEP_BLOCKS", source)
        self.assertIn('share_deep_blocks=profile.get("share_deep_blocks", False)', source)


if __name__ == "__main__":
    unittest.main()
