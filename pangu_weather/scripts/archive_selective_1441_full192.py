#!/usr/bin/env python3
"""Archive the rejected selective 1-4-4-1 student and write its manifest."""

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROFILE = "pangu_selective_1441_full192"
DEFAULT_PREFIX = "selective_1441_full192_recovery"
ARCHIVE_NAME = "pangu_selective_1441_full192_rejected_20260713"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prefix(prefix):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix):
        raise ValueError(f"Unsafe checkpoint prefix: {prefix}")


def artifact_paths(checkpoint_dir, logs_dir, prefix):
    checkpoint_names = [
        f"{PROFILE}_init_train.pth",
        f"{PROFILE}_init_fp16.pth",
        f"{prefix}_latest.pth",
        f"{prefix}_train.pth",
        f"{prefix}_fp16.pth",
        f"{prefix}_fp16_compact.pth",
    ]
    log_names = [
        "selective_1441_full192_init.jsonl",
        "selective_1441_full192_trained_compact.jsonl",
    ]
    artifacts = [
        (path, Path("checkpoints") / path.name)
        for name in checkpoint_names
        if (path := checkpoint_dir / name).is_file()
    ]
    artifacts.extend(
        (path, Path("logs") / path.name)
        for name in log_names
        if (path := logs_dir / name).is_file()
    )
    training_logs = sorted(
        logs_dir.glob("distill_train_*.log"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if training_logs:
        path = training_logs[0]
        artifacts.append((path, Path("logs") / path.name))
    return artifacts


def archive(args):
    validate_prefix(args.prefix)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    if archive_dir.exists():
        raise FileExistsError(f"Archive already exists: {archive_dir}")

    artifacts = artifact_paths(checkpoint_dir, logs_dir, args.prefix)
    if not artifacts:
        raise FileNotFoundError("No selective 1-4-4-1 artifacts found to archive")

    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = archive_dir.with_name(archive_dir.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"Stale archive staging directory exists: {staging}")
    staging.mkdir()

    records = []
    try:
        for source, relative in artifacts:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha256_file(source)
            shutil.copy2(source, destination)
            copied_hash = sha256_file(destination)
            if copied_hash != source_hash:
                raise RuntimeError(f"Archive copy SHA256 mismatch: {source}")
            records.append(
                {
                    "source": str(source),
                    "archived_path": str(relative),
                    "bytes": source.stat().st_size,
                    "sha256": source_hash,
                }
            )

        manifest = {
            "schema_version": 1,
            "status": "rejected_archived",
            "profile": PROFILE,
            "checkpoint_prefix": args.prefix,
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "rejection": {
                "weighted_validation_loss": 0.2003,
                "reason": (
                    "patch4-to-patch8 token collapse plus width192-to-width96 "
                    "reduction did not preserve full-model forecast quality"
                ),
                "teacher_used": False,
                "random_initialized_parameters": 0,
            },
            "structure": {
                "patch_size": [2, 8, 8],
                "embed_dim": 96,
                "num_heads": [3, 6, 6, 3],
                "depth_blocks": [1, 4, 4, 1],
                "window_size": [2, 6, 12],
                "mlp_ratio": 4,
            },
            "artifacts": records,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, archive_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    for source, _relative in artifacts:
        source.unlink()

    print(f"Archived rejected profile: {PROFILE}")
    print(f"Archive directory: {archive_dir}")
    print(f"Artifacts: {len(records)}")
    print(f"Manifest: {archive_dir / 'manifest.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--checkpoint-dir", default="data/checkpoints")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--archive-dir",
        default=f"data/checkpoints/archive/{ARCHIVE_NAME}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    archive(parse_args())
