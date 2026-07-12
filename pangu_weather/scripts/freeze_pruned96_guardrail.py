#!/usr/bin/env python3
"""Freeze the verified pruned_96 submission artifacts with a hash manifest."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


VERIFIED_SCORE = {"total": 89.6297, "u": 36.0095, "v": 17.4002, "w": 36.2200}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_guardrail(output_dir, artifacts):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite guardrail: {output_dir}")
    resolved = []
    for label, source in artifacts:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing {label}: {source}")
        resolved.append((label, source))

    output_dir.mkdir(parents=True)
    manifest = {
        "profile": {
            "name": "pgw_lite_pruned_96",
            "patch_size": [2, 8, 8],
            "embed_dim": 96,
            "depth_blocks": [2, 6, 6, 2],
            "output_channels": 69,
        },
        "platform_score": dict(VERIFIED_SCORE),
        "artifacts": [],
    }
    try:
        for label, source in resolved:
            destination = output_dir / source.name
            if destination.exists():
                raise FileExistsError(f"Duplicate guardrail filename: {source.name}")
            shutil.copy2(source, destination)
            manifest["artifacts"].append(
                {
                    "label": label,
                    "filename": destination.name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        manifest_path = output_dir / "guardrail_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_dir)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-zip", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", default="guardrails/pruned96_89_6297")
    args = parser.parse_args()
    manifest = freeze_guardrail(
        args.output_dir,
        [
            ("submission_zip", args.submission_zip),
            ("checkpoint", args.checkpoint),
            ("calibration", args.calibration),
        ],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
