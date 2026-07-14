#!/usr/bin/env python3
"""Audit the compliant code package and optional external model archive."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path


REQUIRED_PATHS = {
    "pangu_weather/README.md",
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
    "pangu_weather/scripts/compact_fuser_alias_checkpoint.py",
    "pangu_weather/scripts/convert_fp16.py",
    "pangu_weather/scripts/audit_submission_package.py",
    "pangu_weather/scripts/prune_structured.py",
    "pangu_weather/scripts/quantize_mixed_precision.py",
}
ALLOWED_PATHS = set(REQUIRED_PATHS)

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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint_metadata(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a metadata dictionary")
    profile = checkpoint.get("model_profile") or {}
    distillation = checkpoint.get("distillation") or {}
    if not isinstance(profile, dict) or not profile.get("name"):
        raise ValueError("Checkpoint is missing model_profile.name")
    if not isinstance(distillation, dict):
        raise ValueError("Checkpoint distillation metadata must be a dictionary")
    required = {
        "teacher_source",
        "ground_truth_weight",
        "teacher_weight",
        "hint_weight",
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
    hint = float(distillation["hint_weight"])
    if min(hard, teacher, hint) < 0:
        raise ValueError("Distillation weights must be non-negative")
    if teacher + hint <= hard:
        raise ValueError(
            "Full-model teacher constraints must dominate ground-truth supervision"
        )
    return {
        "model_profile": profile["name"],
        "teacher_source": distillation["teacher_source"],
        "ground_truth_weight": hard,
        "teacher_weight": teacher,
        "hint_weight": hint,
        "all_69_channels": True,
        "predict_residual": False,
    }


def audit_zip(path, model_path=None):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        path_list = [item.filename for item in files]
        paths = set(path_list)
        missing = sorted(REQUIRED_PATHS - paths)
        unexpected = sorted(paths - ALLOWED_PATHS)
        duplicates = sorted(name for name in paths if path_list.count(name) > 1)
        if missing or unexpected or duplicates:
            raise ValueError(
                "Submission package mismatch: "
                f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
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
            checkpoint = torch.load(
                io.BytesIO(model_archive.read("model_fp16.pth")),
                map_location="cpu",
                weights_only=False,
            )
        provenance = audit_checkpoint_metadata(checkpoint)
        report["model"] = str(model_path.resolve())
        report["model_bytes"] = model_path.stat().st_size
        report["model_sha256"] = sha256_file(model_path)
        report["scored_bytes_compressed_view"] = (
            report["package_bytes"] + report["model_bytes"]
        )
        report["scored_bytes_expanded_code_view"] = (
            report["uncompressed_code_bytes"] + report["model_bytes"]
        )
        report["model_members"] = model_files
        report["model_provenance"] = provenance
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
