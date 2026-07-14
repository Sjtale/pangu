import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path

import torch


PANGU = Path(__file__).resolve().parents[1]
SCRIPT = PANGU / "scripts" / "audit_selective_mlp96_checkpoint.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_selective_mlp96_checkpoint", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_parameter_state():
    state = OrderedDict(
        (key, torch.zeros(shape, dtype=torch.float16))
        for key, shape in MODULE.expected_mlp_shapes().items()
    )
    current = sum(tensor.numel() for tensor in state.values())
    state["backbone.weight"] = torch.zeros(
        MODULE.EXPECTED_PARAMETER_COUNT - current,
        dtype=torch.float16,
    )
    return state


def make_initialization(state):
    neuron_indices = {
        f"{stage}.Fuser.blocks.{block}.transformer.mlp": list(range(384))
        for stage, block in MODULE.TARGET_MLP_BLOCKS
    }
    key_count = len(state)
    return {
        "method": MODULE.INITIALIZATION_METHOD,
        "profile_name": MODULE.PROFILE_NAME,
        "human_label": "SelectiveMLP-96",
        "source": "full_depth_ratio4_pruned96",
        "teacher": "official_full192",
        "source_sha256": "a" * 64,
        "teacher_sha256": "b" * 64,
        "init_sha256": "c" * 64,
        "mlp_ratio_blocks": copy.deepcopy(MODULE.MLP_RATIO_BLOCKS),
        "neuron_indices": neuron_indices,
        "activation_calibration": {
            "dataset": "official_train",
            "input_indices": list(range(32)),
            "input_count": 32,
            "tokens_per_input": 4096,
            "tokens_per_mlp": 131072,
            "seed": 20260713,
            "importance": MODULE.EXPECTED_IMPORTANCE_FORMULA,
        },
        "state_tensor_keys": key_count,
        "covered_state_tensor_keys": key_count,
        "parameter_state_keys": key_count,
        "covered_parameter_state_keys": key_count,
        "strict_coverage": True,
        "random_initialized_parameters": 0,
    }


def make_payload():
    state = make_parameter_state()
    return {
        "model_state_dict": state,
        "model_profile": {
            "name": MODULE.PROFILE_NAME,
            **copy.deepcopy(MODULE.PROFILE_SPEC),
        },
        "initialization": make_initialization(state),
    }


class SelectiveMLP96CheckpointAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temporary.name) / "selective_mlp96_parameters.pth"
        cls.payload = make_payload()
        torch.save(cls.payload, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.payload = None
        cls.temporary.cleanup()

    def test_valid_parameter_only_checkpoint_reports_exact_size_and_count(self):
        report = MODULE.audit_checkpoint(self.checkpoint, model=None)
        self.assertTrue(report["passed"])
        self.assertEqual(report["parameter_count"], 14_768_265)
        self.assertEqual(report["verified_mlp_tensor_shapes"], 64)
        self.assertEqual(report["dtype_tensor_counts"], {"torch.float16": 65})
        self.assertAlmostEqual(
            report["logical_mib"],
            14_768_265 * 2 / MODULE.MIB,
            places=6,
        )
        self.assertLessEqual(report["file_mib"], 29.1)
        self.assertTrue(report["file_size_gate_passed"])
        self.assertFalse(report["model_key_check"]["performed"])
        self.assertRegex(report["checkpoint_sha256"], r"^[0-9a-f]{64}$")

    def test_profile_schedule_dtype_and_parameter_only_fail_closed(self):
        wrong_profile = dict(self.payload)
        wrong_profile["model_profile"] = copy.deepcopy(
            self.payload["model_profile"]
        )
        wrong_profile["model_profile"]["mlp_ratio_blocks"][1][1] = 4
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            MODULE.audit_payload(
                wrong_profile,
                file_size_bytes=self.checkpoint.stat().st_size,
            )

        wrong_dtype = dict(self.payload)
        wrong_dtype["model_state_dict"] = OrderedDict(
            self.payload["model_state_dict"]
        )
        bias_key = "layer1.fuser.blocks.0.transformer.mlp.fc1.bias"
        wrong_dtype["model_state_dict"][bias_key] = torch.zeros(384)
        with self.assertRaisesRegex(ValueError, "dense CPU FP16"):
            MODULE.audit_payload(
                wrong_dtype,
                file_size_bytes=self.checkpoint.stat().st_size,
            )

        with_buffer = dict(self.payload)
        with_buffer["model_state_dict"] = OrderedDict(
            self.payload["model_state_dict"]
        )
        with_buffer["model_state_dict"][
            "layer1.fuser.blocks.0.transformer.attn.earth_position_index"
        ] = torch.zeros(1, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "parameter-only"):
            MODULE.audit_payload(
                with_buffer,
                file_size_bytes=self.checkpoint.stat().st_size,
            )

        wrong_shape = dict(self.payload)
        wrong_shape["model_state_dict"] = OrderedDict(
            self.payload["model_state_dict"]
        )
        wrong_shape["model_state_dict"][bias_key] = torch.zeros(
            383, dtype=torch.float16
        )
        with self.assertRaisesRegex(ValueError, "state-shape mismatch"):
            MODULE.audit_payload(
                wrong_shape,
                file_size_bytes=self.checkpoint.stat().st_size,
            )

    def test_quantization_provenance_random_and_coverage_are_rejected(self):
        quantized = dict(self.payload)
        quantized["quantization"] = {"format": "int8"}
        with self.assertRaisesRegex(ValueError, "Quantization metadata"):
            MODULE.audit_payload(
                quantized,
                file_size_bytes=self.checkpoint.stat().st_size,
            )

        for field, value, message in (
            ("source_sha256", "broken", "invalid source_sha256"),
            ("random_initialized_parameters", 1, "random_initialized_parameters=0"),
            ("covered_parameter_state_keys", 0, "coverage mismatch"),
        ):
            with self.subTest(field=field):
                metadata = copy.deepcopy(self.payload["initialization"])
                metadata[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.validate_initialization_metadata(metadata)

        old_sampling = copy.deepcopy(self.payload["initialization"])
        old_sampling["activation_calibration"]["tokens_per_input"] = 128
        old_sampling["activation_calibration"]["tokens_per_mlp"] = 4096
        with self.assertRaisesRegex(ValueError, "activation calibration mismatch"):
            MODULE.validate_initialization_metadata(old_sampling)

        wrong_importance = copy.deepcopy(self.payload["initialization"])
        wrong_importance["activation_calibration"]["importance"] = "magnitude-only"
        with self.assertRaisesRegex(ValueError, "activation calibration mismatch"):
            MODULE.validate_initialization_metadata(wrong_importance)

    def test_built_model_check_rejects_missing_unexpected_and_wrong_shapes(self):
        model = torch.nn.Linear(3, 2)
        state = OrderedDict(
            weight=torch.zeros(2, 3, dtype=torch.float16),
            bias=torch.zeros(2, dtype=torch.float16),
        )
        result = MODULE.validate_model_parameter_keys(state, model)
        self.assertTrue(result["performed"])
        self.assertEqual(result["expected_trainable_keys"], 2)

        with self.assertRaisesRegex(ValueError, "missing=.*bias"):
            MODULE.validate_model_parameter_keys(
                OrderedDict(weight=state["weight"]), model
            )
        with self.assertRaisesRegex(ValueError, "unexpected=.*extra.weight"):
            MODULE.validate_model_parameter_keys(
                OrderedDict(state, **{"extra.weight": torch.zeros(1)}), model
            )
        with self.assertRaisesRegex(ValueError, "shape_mismatch=.*weight"):
            MODULE.validate_model_parameter_keys(
                OrderedDict(
                    weight=torch.zeros(3, 2, dtype=torch.float16),
                    bias=state["bias"],
                ),
                model,
            )

    def test_file_size_gate_and_server_cli(self):
        with self.assertRaisesRegex(ValueError, "file-size gate failed"):
            MODULE.audit_payload(
                self.payload,
                file_size_bytes=math.floor(29.1 * MODULE.MIB) + 1,
            )

        report_path = Path(self.temporary.name) / "audit.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = MODULE.main(
                [
                    str(self.checkpoint),
                    "--skip-model-check",
                    "--json-out",
                    str(report_path),
                ]
            )
        self.assertEqual(returncode, 0)
        emitted = json.loads(stdout.getvalue())
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(emitted["passed"])
        self.assertEqual(emitted, persisted)


if __name__ == "__main__":
    unittest.main()
