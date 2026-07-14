"""Tests for deterministic earth-position-index checkpoint elision."""

import importlib.util
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


SCRIPT = Path(__file__).parents[1] / "scripts" / "elide_deterministic_indices.py"
SPEC = importlib.util.spec_from_file_location("elide_deterministic_indices", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        self.register_buffer(
            "earth_position_index", torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        )


class AliasWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuser = TinyAttention()
        self.Fuser = self.fuser


class TinyPruned96(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = AliasWrapper()


def exact_profile():
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in MODULE.EXPECTED_PROFILE.items()
    }


def alias_compacted_state(model):
    state = OrderedDict()
    for key, value in model.state_dict().items():
        if ".fuser." in key:
            continue
        state[key] = value.clone().half() if torch.is_floating_point(value) else value.clone()
    return state


def write_checkpoint(path, model, index_delta=0):
    state = alias_compacted_state(model)
    index_key = "layer.Fuser.earth_position_index"
    if index_delta:
        state[index_key] = state[index_key].clone()
        state[index_key][0, 0] += index_delta
    checkpoint = {
        "model_profile": exact_profile(),
        "model_state_dict": state,
        "distillation": {"teacher_source": "organizer_pangu_full_model"},
        "storage_optimization": {"existing_marker": "preserve-me"},
    }
    torch.save(checkpoint, path)
    return checkpoint


class DeterministicIndexElisionTests(unittest.TestCase):
    def test_elides_verified_indices_and_preserves_everything_else(self):
        model = TinyPruned96()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            original = write_checkpoint(source, model)

            with mock.patch.object(MODULE, "build_runtime_model", return_value=model):
                manifest = MODULE.elide_checkpoint(source, output)

            candidate = MODULE.load_checkpoint(output)
            state = candidate["model_state_dict"]
            retained_key = "layer.Fuser.weight"
            removed_key = "layer.Fuser.earth_position_index"
            self.assertEqual(list(state), [retained_key])
            self.assertTrue(torch.equal(state[retained_key], original["model_state_dict"][retained_key]))
            self.assertEqual(state[retained_key].dtype, torch.float16)
            self.assertEqual(candidate["distillation"], original["distillation"])
            self.assertEqual(
                candidate["storage_optimization"]["existing_marker"], "preserve-me"
            )
            self.assertEqual(candidate["storage_optimization"]["schema_version"], 1)

            stored = candidate["storage_optimization"][MODULE.OPTIMIZATION_KEY]
            self.assertEqual(stored, manifest)
            self.assertEqual(manifest["removed_checkpoint_keys"], [removed_key])
            self.assertEqual(manifest["source_tensor_count"], 2)
            self.assertEqual(manifest["output_tensor_count"], 1)
            self.assertEqual(
                manifest["expected_runtime_missing_keys"],
                [
                    "layer.Fuser.earth_position_index",
                    "layer.fuser.earth_position_index",
                ],
            )
            self.assertEqual(manifest["method"], MODULE.ELISION_METHOD)
            self.assertEqual(manifest["removed_logical_bytes"], 32)
            self.assertEqual(len(manifest["generated_indices"]), 2)
            for key, item in manifest["generated_indices"].items():
                self.assertEqual(item["dtype"], "torch.int64")
                self.assertEqual(item["shape"], [2, 2])
                self.assertEqual(item["bytes"], 32)
                self.assertEqual(
                    item["sha256"], MODULE.tensor_sha256(model.state_dict()[key])
                )

    def test_rejects_index_value_mismatch_without_writing(self):
        model = TinyPruned96()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            write_checkpoint(source, model, index_delta=1)

            with mock.patch.object(MODULE, "build_runtime_model", return_value=model):
                with self.assertRaisesRegex(ValueError, "Index value mismatch"):
                    MODULE.elide_checkpoint(source, output)
            self.assertFalse(output.exists())

    def test_refuses_to_overwrite_existing_output(self):
        model = TinyPruned96()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            write_checkpoint(source, model)
            output.write_bytes(b"keep-existing")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                MODULE.elide_checkpoint(source, output)
            self.assertEqual(output.read_bytes(), b"keep-existing")

    def test_audit_only_validates_without_writing(self):
        model = TinyPruned96()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            write_checkpoint(source, model)

            with mock.patch.object(MODULE, "build_runtime_model", return_value=model):
                manifest = MODULE.elide_checkpoint(source, output, audit_only=True)
            self.assertEqual(manifest["output_tensor_count"], 1)
            self.assertFalse(output.exists())

    def test_rejects_non_exact_profile(self):
        model = TinyPruned96()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            checkpoint = write_checkpoint(source, model)
            checkpoint["model_profile"]["embed_dim"] = 95
            torch.save(checkpoint, source)

            with self.assertRaisesRegex(ValueError, "not exact pgw_lite_pruned_96"):
                MODULE.elide_checkpoint(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
