#!/usr/bin/env python3
"""Freeze the verified pruned_96 submission artifacts with a hash manifest."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


VERIFIED_SCORE = {
    "total": 90.7763,
    "lightweight": 36.6011,
    "inference_time": 17.9950,
    "prediction": 36.1803,
    "metric_mapping": {
        "U": "lightweight",
        "V": "inference_time",
        "W": "prediction",
    },
}


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
    parser.add_argument("--code-zip", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inference", default="inference.py")
    parser.add_argument("--result", default="result.py")
    parser.add_argument(
        "--static-audit",
        default="logs/pruned96_static_90_7763.json",
    )
    parser.add_argument("--output-dir", default="guardrails/pruned96_90_7763")
    args = parser.parse_args()
    manifest = freeze_guardrail(
        args.output_dir,
        [
            ("code_zip", args.code_zip),
            ("checkpoint", args.checkpoint),
            ("inference", args.inference),
            ("result", args.result),
            ("static_audit", args.static_audit),
        ],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
