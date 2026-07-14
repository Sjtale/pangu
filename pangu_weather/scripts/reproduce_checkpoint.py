#!/usr/bin/env python3
"""Reproduce the submitted checkpoint from the organizer Pangu checkpoint."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "data" / "checkpoints"
TEACHER = ROOT / "pangu_backups" / "model_bak.pth"
TEACHER_FP16 = CHECKPOINTS / "model_teacher_fp16.pth"
PRUNED = CHECKPOINTS / "model_pgw_lite_pruned_96.pth"
TRAINED = CHECKPOINTS / "model_pgw_lite_pruned_96_train.pth"
DISTILLED = CHECKPOINTS / "model_pgw_lite_pruned_96_fp16.pth"
CONVERTED = CHECKPOINTS / "model_pgw_lite_pruned_96_converted_fp16.pth"
SENSITIVITY = ROOT / "data" / "quant_sensitivity.json"
QUANTIZED = CHECKPOINTS / "model_pgw_lite_pruned_96_quantized.pth"
FINAL = CHECKPOINTS / "model_fp16.pth"
MANIFEST = CHECKPOINTS / "reproduction_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    if not TEACHER.is_file():
        raise FileNotFoundError(f"Organizer checkpoint not found: {TEACHER}")
    outputs = (
        TEACHER_FP16,
        PRUNED,
        TRAINED,
        DISTILLED,
        CONVERTED,
        SENSITIVITY,
        QUANTIZED,
        FINAL,
        MANIFEST,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to reuse reproduction outputs; remove or archive them first: "
            + ", ".join(existing)
        )
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    run(
        "scripts/convert_fp16.py",
        "--source",
        str(TEACHER),
        "--output",
        str(TEACHER_FP16),
    )
    run(
        "scripts/prune_structured.py",
        "--source",
        str(TEACHER_FP16),
        "--output",
        str(PRUNED),
    )

    run("distill_train.py")
    if not DISTILLED.is_file():
        raise FileNotFoundError(f"Distillation did not create {DISTILLED}")

    run(
        "scripts/convert_fp16.py",
        "--source",
        str(DISTILLED),
        "--output",
        str(CONVERTED),
    )
    run("scripts/analyze_quant_sensitivity.py", "--checkpoint", str(CONVERTED))
    run(
        "scripts/quantize_mixed_precision.py",
        "--checkpoint",
        str(CONVERTED),
        "--output",
        str(QUANTIZED),
    )
    run(
        "scripts/compact_fuser_alias_checkpoint.py",
        "--source",
        str(QUANTIZED),
        "--output",
        str(FINAL),
    )

    import torch

    checkpoint = torch.load(FINAL, map_location="cpu", weights_only=False)
    quantization = checkpoint.get("quantization", {})
    compaction = checkpoint.get("alias_compaction", {})
    if quantization.get("fp16_keep_count") != 5:
        raise ValueError("Final checkpoint must retain exactly five FP16 Linear weights")
    if quantization.get("quantized_keys_count") != 62:
        raise ValueError("Final checkpoint must contain exactly 62 quantized weights")
    if compaction.get("alias_pair_count") != 224:
        raise ValueError("Final checkpoint must compact exactly 224 OneFuser aliases")

    artifacts = (
        TEACHER,
        TEACHER_FP16,
        PRUNED,
        TRAINED,
        DISTILLED,
        CONVERTED,
        SENSITIVITY,
        QUANTIZED,
        FINAL,
    )
    payload = {
        "pipeline": [
            "organizer_teacher",
            "teacher_fp16",
            "structured_pruning",
            "all_69_distillation",
            "fp16_conversion",
            "int4_sensitivity",
            "top5_mixed_precision",
            "onefuser_alias_compaction",
        ],
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
        "model_profile": checkpoint.get("model_profile"),
        "distillation": checkpoint.get("distillation"),
        "quantization": quantization,
        "alias_compaction": compaction,
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Reproduced checkpoint: {FINAL}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
