#!/usr/bin/env python3
"""Audit the minimal pruned_96 code package and optional external model file."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


REQUIRED_BASENAMES = {
    "calibration_utils.py",
    "config.yaml",
    "download_model_url.txt",
    "hip_runtime_controls.py",
    "inference.py",
    "pangu_profile_model.py",
    "result.py",
}
ALLOWED_BASENAMES = set(REQUIRED_BASENAMES)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_zip(path, model_path=None):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
    basename_list = [PurePosixPath(item.filename).name for item in files]
    basenames = set(basename_list)
    missing = sorted(REQUIRED_BASENAMES - basenames)
    unexpected = sorted(basenames - ALLOWED_BASENAMES)
    duplicates = sorted(
        basename for basename in basenames if basename_list.count(basename) > 1
    )
    if missing or unexpected or duplicates:
        raise ValueError(
            "Submission package mismatch: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )
    report = {
        "package": str(path.resolve()),
        "package_bytes": path.stat().st_size,
        "package_sha256": sha256_file(path),
        "uncompressed_code_bytes": sum(item.file_size for item in files),
        "files": sorted(item.filename for item in files),
    }
    if model_path is not None:
        model_path = Path(model_path)
        report["model"] = str(model_path.resolve())
        report["model_bytes"] = model_path.stat().st_size
        report["model_sha256"] = sha256_file(model_path)
        report["scored_bytes_compressed_view"] = (
            report["package_bytes"] + report["model_bytes"]
        )
        report["scored_bytes_expanded_code_view"] = (
            report["uncompressed_code_bytes"] + report["model_bytes"]
        )
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
