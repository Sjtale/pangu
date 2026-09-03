#!/usr/bin/env python3
"""Audit the compliant code package and optional external model archive."""

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


REQUIRED_PATHS = {
    "pangu_weather/README.md",
    "pangu_weather/蒸馏与推理说明.md",
    "pangu_weather/侍奉部_说明文档.pdf",
    "pangu_weather/compliant_inference_wrapper.py",
    "pangu_weather/conf/config.yaml",
    "pangu_weather/data/download_model_url.txt",
    "pangu_weather/hip_earth_attention_tiled.py",
    "pangu_weather/hip_kernels/earth_attention_tiled_fwd.hip",
    "pangu_weather/hip_runtime_controls.py",
    "pangu_weather/inference.py",
    "pangu_weather/p2_tiled_attention.py",
    "pangu_weather/pangu_profile_model.py",
    "pangu_weather/result.py",
    "pangu_weather/score_training_utils.py",
    "pangu_weather/selective_mlp96.py",
    "pangu_weather/train.py",
    "pangu_weather/distill_train.py",
    "pangu_weather/scripts/analyze_quant_sensitivity.py",
    "pangu_weather/scripts/compact_fuser_alias_checkpoint.py",
    "pangu_weather/scripts/convert_fp16.py",
    "pangu_weather/scripts/compress_checkpoint_gzip.py",
    "pangu_weather/scripts/audit_submission_package.py",
    "pangu_weather/scripts/prune_structured.py",
    "pangu_weather/scripts/quantize_mixed_precision.py",
}
ALLOWED_PATHS = set(REQUIRED_PATHS)
REQUIRED_DIRECTORIES = {
    "pangu_weather/result/output/",
}

FORBIDDEN_SOURCE_MARKERS = {
    "PANGU_GLOBAL_MEAN_CORRECTION",
    "_direct_patch_recovery_scored_only",
    "calibration_affine.npz",
    "calibration_coeffs.npy",
    "physics_mean_targets.npz",
}
FORBIDDEN_MODEL_MEMBERS = {
    "calibration_affine.npz",
    "calibration_coeffs.npy",
    "physics_mean_targets.npz",
}
GZIP_MAGIC = b"\x1f\x8b"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _decode_checkpoint_bytes(raw_bytes):
    """Decode at most one gzip layer from an external checkpoint member."""

    if not isinstance(raw_bytes, bytes):
        raise TypeError("Checkpoint member payload must be bytes")
    if not raw_bytes.startswith(GZIP_MAGIC):
        return raw_bytes
    try:
        decoded = gzip.decompress(raw_bytes)
    except (EOFError, OSError) as error:
        raise ValueError(f"Corrupt gzip checkpoint member: {error}") from error
    if decoded.startswith(GZIP_MAGIC):
        raise ValueError("Nested gzip checkpoint members are forbidden")
    return decoded


def audit_checkpoint_metadata(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a metadata dictionary")
    profile = checkpoint.get("model_profile") or {}
    if not isinstance(profile, dict) or not profile.get("name"):
        raise ValueError("Checkpoint is missing model_profile.name")
    distillation = checkpoint.get("distillation")
    if distillation is None:
        return {
            "model_profile": profile["name"],
            "distillation_metadata_present": False,
        }
    if not isinstance(distillation, dict):
        raise ValueError("Checkpoint distillation metadata must be a dictionary")
    required = {
        "teacher_source",
        "ground_truth_weight",
        "teacher_weight",
        "all_69_channels",
        "predict_residual",
    }
    missing = sorted(required - set(distillation))
    if missing:
        raise ValueError(f"Checkpoint is missing compliance metadata: {missing}")
    if distillation["teacher_source"] != "organizer_pangu_full_model":
        raise ValueError("Student teacher_source must be organizer_pangu_full_model")
    if distillation["all_69_channels"] is not True:
        raise ValueError("Checkpoint must train and predict all 69 channels")
    if distillation["predict_residual"] is not False:
        raise ValueError("Residual-target student checkpoints are forbidden")
    hard = float(distillation["ground_truth_weight"])
    teacher = float(distillation["teacher_weight"])
    if min(hard, teacher) < 0:
        raise ValueError("Distillation weights must be non-negative")
    if abs(hard - 0.5) > 1.0e-12 or abs(teacher - 0.5) > 1.0e-12:
        raise ValueError("Fixed pruned_96 distillation weights must be 0.5/0.5")
    if "hint_weight" in distillation or "hint_layers" in distillation:
        raise ValueError("Fixed pruned_96 distillation must not use hint loss")
    return {
        "model_profile": profile["name"],
        "distillation_metadata_present": True,
        "teacher_source": distillation["teacher_source"],
        "ground_truth_weight": hard,
        "teacher_weight": teacher,
        "all_69_channels": True,
        "predict_residual": False,
    }


def _tensor_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def _elision_manifest(checkpoint):
    storage = checkpoint.get("storage_optimization")
    if not isinstance(storage, Mapping):
        return None, None
    if "deterministic_index_elision" in storage:
        raise ValueError("Legacy deterministic_index_elision metadata is forbidden")
    key = "deterministic_buffer_elision"
    manifest = storage.get(key)
    if manifest is None:
        return None, None
    if storage.get("schema_version") != 1:
        raise ValueError("Unsupported storage_optimization schema_version")
    if not isinstance(manifest, Mapping):
        raise ValueError(f"storage_optimization.{key} must be a mapping")
    return key, manifest


def _validate_elision_manifest(
    manifest_key,
    manifest,
    state,
    tensor_report,
    checkpoint_profile,
):
    if manifest_key != "deterministic_buffer_elision":
        raise ValueError("Unsupported deterministic buffer-elision manifest")
    if manifest.get("method") != "constructor-earth-position-index-v1":
        raise ValueError("Unsupported deterministic buffer-elision method")
    if manifest.get("profile") != "pgw_lite_pruned_96":
        raise ValueError("Buffer elision requires exact pgw_lite_pruned_96")
    if checkpoint_profile != manifest.get("profile"):
        raise ValueError(
            "Buffer-elision profile differs from checkpoint model_profile.name"
        )
    removed = manifest.get("removed_checkpoint_keys")
    expected_missing = manifest.get("expected_runtime_missing_keys")
    if not isinstance(removed, list) or not removed:
        raise ValueError("Elision manifest requires removed_checkpoint_keys")
    if not isinstance(expected_missing, list) or not expected_missing:
        raise ValueError("Elision manifest requires expected_runtime_missing_keys")
    if len(removed) != len(set(removed)):
        raise ValueError("Elision removed_checkpoint_keys contains duplicates")
    if len(expected_missing) != len(set(expected_missing)):
        raise ValueError("Elision expected_runtime_missing_keys contains duplicates")

    removed_set = set(removed)
    expected_missing_set = set(expected_missing)
    if any(not key.endswith("earth_position_index") for key in removed_set):
        raise ValueError("Elision removed_checkpoint_keys contains a non-index key")
    if not removed_set.issubset(expected_missing_set):
        raise ValueError("Elision removed keys are absent from expected missing keys")
    if removed_set & set(state):
        raise ValueError("Elided index key remains in model_state_dict")

    generated = manifest.get("generated_indices")
    if not isinstance(generated, Mapping):
        raise ValueError("Buffer-elision manifest requires generated_indices")
    generated_set = set(generated)
    if generated_set != expected_missing_set:
        raise ValueError("Elision generated index manifest set mismatch")
    if any(not key.endswith("earth_position_index") for key in generated_set):
        raise ValueError("Elision generated manifest contains a non-index key")

    removed_bytes = 0
    for key, record in generated.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid generated-index record: {key}")
        shape = record.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(f"Invalid generated-index shape: {key}")
        if record.get("dtype") != "torch.int64":
            raise ValueError(f"Invalid generated-index dtype: {key}")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid generated-index SHA256: {key}")
        expected_bytes = 8
        for dimension in shape:
            expected_bytes *= dimension
        if record.get("bytes") != expected_bytes:
            raise ValueError(f"Invalid generated-index byte count: {key}")
        if key in removed_set:
            removed_bytes += expected_bytes

    if manifest.get("removed_tensor_count") != len(removed_set):
        raise ValueError("Elision removed tensor count mismatch")
    if manifest.get("removed_logical_bytes") != removed_bytes:
        raise ValueError("Elision removed logical byte count mismatch")
    if manifest.get("output_tensor_count") != tensor_report["tensor_count"]:
        raise ValueError("Elision output tensor count mismatch")
    if manifest.get("output_dtype_bytes") != tensor_report["dtype_bytes"]:
        raise ValueError("Elision output dtype-byte map mismatch")
    source_count = manifest.get("source_tensor_count")
    if source_count != tensor_report["tensor_count"] + len(removed_set):
        raise ValueError("Elision source tensor count mismatch")

    if tensor_report["earth_position_index_count"] != 0:
        raise ValueError("Index-elided checkpoint still contains earth_position_index")
    if tensor_report["int8_tensor_count"] != 0 or tensor_report["scale_tensor_count"] != 0:
        raise ValueError("Index-elided checkpoint must not contain INT8 or scale tensors")
    if tensor_report["earth_position_bias_table_count"] != 16:
        raise ValueError("Index-elided checkpoint must retain 16 earth-position bias tables")


def audit_checkpoint_tensors(checkpoint):
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint model_state_dict must be a mapping")

    dtype_counts = Counter()
    dtype_bytes = Counter()
    unique_storages = {}
    int8_keys = []
    scale_keys = []
    index_keys = []
    bias_keys = []
    logical_bytes = 0
    for key, tensor in state.items():
        if not hasattr(tensor, "untyped_storage"):
            raise ValueError(f"model_state_dict contains a non-tensor value: {key}")
        dtype = str(tensor.dtype)
        logical = _tensor_bytes(tensor)
        logical_bytes += logical
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += logical
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), int(storage._cdata))
        unique_storages.setdefault(storage_key, int(storage.nbytes()))
        if dtype == "torch.int8":
            int8_keys.append(key)
        if key.endswith("_scale"):
            scale_keys.append(key)
        if key.endswith("earth_position_index"):
            index_keys.append(key)
        if key.endswith("earth_position_bias_table"):
            bias_keys.append(key)

    quantization = checkpoint.get("quantization")
    metadata_quantized_count = (
        quantization.get("quantized_keys_count")
        if isinstance(quantization, Mapping)
        else None
    )
    consistency_issues = []
    if metadata_quantized_count is not None:
        try:
            metadata_quantized_count = int(metadata_quantized_count)
        except (TypeError, ValueError) as error:
            raise ValueError("quantization.quantized_keys_count must be an integer") from error
        if metadata_quantized_count != len(int8_keys):
            consistency_issues.append(
                "quantized_keys_count does not match actual INT8 tensor count"
            )
        if metadata_quantized_count != len(scale_keys):
            consistency_issues.append(
                "quantized_keys_count does not match actual scale tensor count"
            )
    elif int8_keys or scale_keys:
        consistency_issues.append("actual INT8/scale tensors lack quantization metadata")

    report = {
        "tensor_count": len(state),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "logical_bytes": logical_bytes,
        "unique_storage_count": len(unique_storages),
        "unique_storage_bytes": sum(unique_storages.values()),
        "int8_tensor_count": len(int8_keys),
        "int8_tensor_bytes": sum(_tensor_bytes(state[key]) for key in int8_keys),
        "int8_keys": sorted(int8_keys),
        "scale_tensor_count": len(scale_keys),
        "scale_tensor_bytes": sum(_tensor_bytes(state[key]) for key in scale_keys),
        "scale_keys": sorted(scale_keys),
        "earth_position_index_count": len(index_keys),
        "earth_position_index_bytes": sum(
            _tensor_bytes(state[key]) for key in index_keys
        ),
        "earth_position_index_keys": sorted(index_keys),
        "earth_position_bias_table_count": len(bias_keys),
        "earth_position_bias_table_bytes": sum(
            _tensor_bytes(state[key]) for key in bias_keys
        ),
        "earth_position_bias_table_keys": sorted(bias_keys),
        "quantization_metadata_present": isinstance(quantization, Mapping),
        "quantization_metadata_quantized_keys_count": metadata_quantized_count,
        "quantization_metadata_matches_actual": not consistency_issues,
        "quantization_consistency_issues": consistency_issues,
    }
    manifest_key, manifest = _elision_manifest(checkpoint)
    report["storage_optimization_manifest"] = manifest_key
    if manifest is not None:
        profile = checkpoint.get("model_profile")
        checkpoint_profile = profile.get("name") if isinstance(profile, Mapping) else None
        _validate_elision_manifest(
            manifest_key,
            manifest,
            state,
            report,
            checkpoint_profile,
        )
        report["storage_optimization_validated"] = True
    else:
        report["storage_optimization_validated"] = False
    return report


def audit_zip(path, model_path=None):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        directories = [item for item in archive.infolist() if item.is_dir()]
        path_list = [item.filename for item in files]
        directory_list = [item.filename for item in directories]
        paths = set(path_list)
        directory_paths = set(directory_list)
        missing = sorted(REQUIRED_PATHS - paths)
        missing_directories = sorted(REQUIRED_DIRECTORIES - directory_paths)
        unexpected = sorted(paths - ALLOWED_PATHS)
        duplicates = sorted(name for name in paths if path_list.count(name) > 1)
        directory_duplicates = sorted(
            name for name in directory_paths if directory_list.count(name) > 1
        )
        if missing or missing_directories or unexpected or duplicates or directory_duplicates:
            raise ValueError(
                "Submission package mismatch: "
                f"missing={missing}, missing_directories={missing_directories}, "
                f"unexpected={unexpected}, duplicates={duplicates}, "
                f"directory_duplicates={directory_duplicates}"
            )

        violations = {}
        for item in files:
            if not item.filename.endswith((".py", ".yaml", ".sh")):
                continue
            if item.filename == "pangu_weather/scripts/audit_submission_package.py":
                continue
            source = archive.read(item).decode("utf-8", errors="replace")
            found = sorted(marker for marker in FORBIDDEN_SOURCE_MARKERS if marker in source)
            if found:
                violations[item.filename] = found
        if violations:
            raise ValueError(f"Forbidden compliance paths found: {violations}")

        report = {
            "package": str(path.resolve()),
            "package_bytes": path.stat().st_size,
            "package_sha256": sha256_file(path),
            "uncompressed_code_bytes": sum(item.file_size for item in files),
            "files": sorted(item.filename for item in files),
            "directories": sorted(item.filename for item in directories),
        }
    if model_path is not None:
        model_path = Path(model_path)
        with zipfile.ZipFile(model_path) as model_archive:
            model_entries = [item for item in model_archive.infolist() if not item.is_dir()]
            model_files = [item.filename for item in model_entries]
        if model_files != ["model_fp16.pth"]:
            raise ValueError(
                "External model ZIP must contain exactly model_fp16.pth at its root: "
                f"files={model_files}"
            )
        forbidden_members = sorted(
            name for name in model_files if Path(name).name in FORBIDDEN_MODEL_MEMBERS
        )
        if forbidden_members:
            raise ValueError(f"Forbidden model artifacts: {forbidden_members}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for checkpoint provenance audit") from exc
        with zipfile.ZipFile(model_path) as model_archive:
            raw_checkpoint_bytes = model_archive.read("model_fp16.pth")
            decoded_checkpoint_bytes = _decode_checkpoint_bytes(raw_checkpoint_bytes)
            checkpoint = torch.load(
                io.BytesIO(decoded_checkpoint_bytes),
                map_location="cpu",
                weights_only=False,
            )
        provenance = audit_checkpoint_metadata(checkpoint)
        tensor_audit = audit_checkpoint_tensors(checkpoint)
        quantization = checkpoint.get("quantization")
        if not isinstance(quantization, Mapping):
            raise ValueError("Final checkpoint is missing quantization metadata")
        if int(quantization.get("fp16_keep_count", -1)) != 67:
            raise ValueError("Final checkpoint must retain all 67 FP16 Linear weights")
        if int(quantization.get("quantized_keys_count", -1)) != 0:
            raise ValueError("Final checkpoint must declare zero quantized weights")
        if not tensor_audit["quantization_metadata_matches_actual"]:
            raise ValueError(
                "Final checkpoint quantization metadata differs from actual tensors: "
                f"{tensor_audit['quantization_consistency_issues']}"
            )
        alias_compaction = checkpoint.get("alias_compaction")
        if not isinstance(alias_compaction, Mapping):
            raise ValueError("Final checkpoint is missing alias_compaction metadata")
        if int(alias_compaction.get("alias_pair_count", -1)) != 224:
            raise ValueError("Final checkpoint must compact exactly 224 alias pairs")
        report["model"] = str(model_path.resolve())
        report["model_bytes"] = model_path.stat().st_size
        report["model_sha256"] = sha256_file(model_path)
        report["model_member_encoding"] = (
            "gzip" if raw_checkpoint_bytes.startswith(GZIP_MAGIC) else "raw"
        )
        report["model_member_raw_bytes"] = len(raw_checkpoint_bytes)
        report["model_member_raw_sha256"] = _sha256_bytes(raw_checkpoint_bytes)
        report["model_member_decoded_bytes"] = len(decoded_checkpoint_bytes)
        report["model_member_decoded_sha256"] = _sha256_bytes(
            decoded_checkpoint_bytes
        )
        report["scored_bytes_compressed_view"] = (
            report["package_bytes"] + report["model_bytes"]
        )
        report["scored_bytes_expanded_code_view"] = (
            report["uncompressed_code_bytes"] + report["model_bytes"]
        )
        report["model_members"] = model_files
        report["model_provenance"] = provenance
        report["model_tensor_audit"] = tensor_audit
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--model")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_zip(args.package, args.model)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite audit report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
