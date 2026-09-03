"""Focused tests for deterministic checkpoint gzip and package decoding."""

import contextlib
import gzip
import hashlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPRESS = load_script("compress_checkpoint_gzip")
AUDIT = load_script("audit_submission_package")


def compliant_checkpoint():
    import torch

    return {
        "model_state_dict": {
            **{f"linear{index}.weight": torch.ones(1, 1, dtype=torch.int8) for index in range(62)},
            **{f"linear{index}.weight_scale": torch.ones(1, 1, dtype=torch.float16) for index in range(62)},
        },
        "model_profile": {"name": "pgw_lite_pruned_96"},
        "distillation": {
            "teacher_source": "organizer_pangu_full_model",
            "ground_truth_weight": 0.5,
            "teacher_weight": 0.5,
            "all_69_channels": True,
            "predict_residual": False,
        },
        "quantization": {"fp16_keep_count": 5, "quantized_keys_count": 62},
        "alias_compaction": {"alias_pair_count": 224},
    }


class CheckpointGzipAuditTests(unittest.TestCase):
    def test_tensor_auditor_accepts_only_canonical_index_elision_manifest(self):
        import torch

        state = OrderedDict(
            (
                f"layer{index}.attn.earth_position_bias_table",
                torch.ones(1, dtype=torch.float16),
            )
            for index in range(16)
        )
        upper = "layer.Fuser.earth_position_index"
        lower = "layer.fuser.earth_position_index"
        generated = {
            key: {
                "shape": [2, 2],
                "dtype": "torch.int64",
                "sha256": "a" * 64,
                "bytes": 32,
            }
            for key in (upper, lower)
        }
        checkpoint = {
            "model_state_dict": state,
            "model_profile": {"name": "pgw_lite_pruned_96"},
            "storage_optimization": {
                "schema_version": 1,
                "deterministic_buffer_elision": {
                    "method": "constructor-earth-position-index-v1",
                    "profile": "pgw_lite_pruned_96",
                    "source_tensor_count": 17,
                    "output_tensor_count": 16,
                    "output_dtype_bytes": {"torch.float16": 32},
                    "removed_tensor_count": 1,
                    "removed_logical_bytes": 32,
                    "removed_checkpoint_keys": [upper],
                    "expected_runtime_missing_keys": [upper, lower],
                    "generated_indices": generated,
                },
            },
        }
        report = AUDIT.audit_checkpoint_tensors(checkpoint)
        self.assertTrue(report["storage_optimization_validated"])
        self.assertEqual(report["earth_position_bias_table_count"], 16)

        checkpoint["model_profile"]["name"] = "different_profile"
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            AUDIT.audit_checkpoint_tensors(checkpoint)
        checkpoint["model_profile"]["name"] = "pgw_lite_pruned_96"

        checkpoint["storage_optimization"]["deterministic_buffer_elision"][
            "method"
        ] = "unverified"
        with self.assertRaisesRegex(ValueError, "Unsupported.*method"):
            AUDIT.audit_checkpoint_tensors(checkpoint)

    def test_compressor_is_deterministic_and_omits_filename_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint.pth"
            first = root / "first.pth"
            second = root / "second.pth"
            payload = (b"deterministic-checkpoint\0" * 4096) + bytes(range(256))
            source.write_bytes(payload)

            with contextlib.redirect_stdout(io.StringIO()):
                COMPRESS.compress_checkpoint(source, first)
                COMPRESS.compress_checkpoint(source, second)

            first_bytes = first.read_bytes()
            self.assertEqual(first_bytes, second.read_bytes())
            self.assertEqual(first_bytes[4:8], b"\0\0\0\0")
            self.assertEqual(first_bytes[3] & 0x08, 0)
            self.assertEqual(gzip.decompress(first_bytes), payload)
            with self.assertRaisesRegex(ValueError, "requires gzip level 9"):
                COMPRESS.compress_checkpoint(source, root / "level8.pth", level=8)

    def test_decoder_accepts_raw_and_one_layer_but_rejects_bad_or_nested_gzip(self):
        payload = b"raw-checkpoint-payload"
        compressed = gzip.compress(payload, mtime=0)
        self.assertIs(AUDIT._decode_checkpoint_bytes(payload), payload)
        self.assertEqual(AUDIT._decode_checkpoint_bytes(compressed), payload)
        with self.assertRaisesRegex(ValueError, "Corrupt gzip"):
            AUDIT._decode_checkpoint_bytes(AUDIT.GZIP_MAGIC + b"truncated")
        with self.assertRaisesRegex(ValueError, "Nested gzip"):
            AUDIT._decode_checkpoint_bytes(gzip.compress(compressed, mtime=0))

    def test_external_model_report_binds_raw_and_decoded_hashes(self):
        checkpoint = compliant_checkpoint()
        decoded = b"synthetic-torch-checkpoint"
        encoded = gzip.compress(decoded, mtime=0)
        fake_torch = types.SimpleNamespace()

        def fake_load(stream, **_kwargs):
            self.assertEqual(stream.read(), decoded)
            return checkpoint

        fake_torch.load = fake_load
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"torch": fake_torch}
        ):
            root = Path(directory)
            package = root / "submission.zip"
            with zipfile.ZipFile(package, "w") as archive:
                for name in sorted(AUDIT.REQUIRED_DIRECTORIES):
                    archive.writestr(name, b"")
                for name in sorted(AUDIT.REQUIRED_PATHS):
                    archive.writestr(name, name)
            model = root / "model.zip"
            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("model_fp16.pth", encoded)

            report = AUDIT.audit_zip(package, model)

        self.assertEqual(report["model_member_encoding"], "gzip")
        self.assertEqual(report["model_member_raw_bytes"], len(encoded))
        self.assertEqual(report["model_member_decoded_bytes"], len(decoded))
        self.assertEqual(
            report["model_member_raw_sha256"], hashlib.sha256(encoded).hexdigest()
        )
        self.assertEqual(
            report["model_member_decoded_sha256"],
            hashlib.sha256(decoded).hexdigest(),
        )
        self.assertTrue(
            report["model_tensor_audit"]["quantization_metadata_matches_actual"]
        )


if __name__ == "__main__":
    unittest.main()
