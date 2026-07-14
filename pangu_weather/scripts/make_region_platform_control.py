#!/usr/bin/env python3
"""Derive a matched Region-off platform control from a scored Region-on ZIP."""

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile


INFERENCE_MEMBER = "pangu_weather/inference.py"
REGION_ON = b'os.environ.setdefault("PANGU_P2_REGION_RELEASE", "1")'
REGION_OFF = b'os.environ.setdefault("PANGU_P2_REGION_RELEASE", "0")'
TIMER_START = "#----------------------AI4S(\u65f6\u95f4\u5ea6\u91cf\u4e0d\u53ef\u66f4\u6539)---------------------------"
TIMER_END = "#---------------------------------------------------------------------"


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timer_sha256(payload):
    source = payload.decode("utf-8")
    loop = source.index("for batch_index, data in enumerate")
    start = source.index(TIMER_START, loop)
    end = source.index(TIMER_END, start) + len(TIMER_END)
    return _sha256_bytes(source[start:end].encode("utf-8"))


def _write_json_temporary(path, report):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _publish_no_clobber(temporary, destination, label):
    """Atomically publish a same-filesystem temporary file without overwriting."""
    temporary = Path(temporary)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite {label}: {destination}"
        ) from exc


def make_region_off_control(source, output, expected_source_sha256, report_path=None):
    source = Path(source).resolve()
    output = Path(output).resolve()
    report_path = Path(report_path).resolve() if report_path is not None else None
    if source == output:
        raise ValueError("Source and output ZIP paths must differ")
    if report_path is not None and report_path in {source, output}:
        raise ValueError("Source ZIP, output ZIP, and report paths must differ")
    if not source.is_file():
        raise FileNotFoundError(source)
    if report_path is not None and report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")

    source_bytes = source.read_bytes()
    observed_source_sha256 = _sha256_bytes(source_bytes)
    if observed_source_sha256 != expected_source_sha256:
        raise ValueError(
            "Source ZIP SHA256 mismatch: "
            f"expected={expected_source_sha256}, observed={observed_source_sha256}"
        )

    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("Source ZIP contains duplicate member names")
        if INFERENCE_MEMBER not in names:
            raise ValueError(f"Source ZIP is missing {INFERENCE_MEMBER}")
        payloads = {info.filename: archive.read(info.filename) for info in infos}
        archive_comment = archive.comment

    inference_on = payloads[INFERENCE_MEMBER]
    if inference_on.count(REGION_ON) != 1 or REGION_OFF in inference_on:
        raise ValueError(
            "Source inference must contain exactly one Region-on default and "
            "no Region-off default"
        )
    inference_off = inference_on.replace(REGION_ON, REGION_OFF)
    timer_sha256 = _timer_sha256(inference_on)
    if _timer_sha256(inference_off) != timer_sha256:
        raise ValueError("Official timer block changed while deriving control")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary_output = Path(temporary_name)
    temporary_report = None
    try:
        with zipfile.ZipFile(temporary_output, "w") as control:
            control.comment = archive_comment
            for source_info in infos:
                info = copy.copy(source_info)
                payload = payloads[info.filename]
                if info.filename == INFERENCE_MEMBER:
                    payload = inference_off
                control.writestr(info, payload)
        temporary_output.chmod(0o644)

        with zipfile.ZipFile(temporary_output) as control:
            control_infos = control.infolist()
            control_names = [info.filename for info in control_infos]
            control_payloads = {
                info.filename: control.read(info.filename) for info in control_infos
            }
            control_comment = control.comment
        if control_names != names:
            raise ValueError("Control ZIP member order or names changed")
        if control_comment != archive_comment:
            raise ValueError("Control ZIP archive comment changed")
        changed_members = [
            name for name in names if control_payloads[name] != payloads[name]
        ]
        if changed_members != [INFERENCE_MEMBER]:
            raise ValueError(
                f"Control ZIP changed unexpected members: {changed_members}"
            )
        if control_payloads[INFERENCE_MEMBER] != inference_off:
            raise ValueError("Control inference payload is not the exact Region-off edit")

        report = {
            "record_type": "region_platform_control_package",
            "source_package": str(source),
            "source_package_bytes": len(source_bytes),
            "source_package_sha256": observed_source_sha256,
            "output_package": str(output),
            "output_package_bytes": temporary_output.stat().st_size,
            "output_package_sha256": _sha256_path(temporary_output),
            "changed_members": changed_members,
            "inference_member": INFERENCE_MEMBER,
            "source_inference_sha256": _sha256_bytes(inference_on),
            "output_inference_sha256": _sha256_bytes(inference_off),
            "official_timer_block_sha256": timer_sha256,
            "source_region_release_default": 1,
            "output_region_release_default": 0,
        }
        if report_path is not None:
            temporary_report = _write_json_temporary(report_path, report)
        _publish_no_clobber(temporary_output, output, "output")
        if temporary_report is not None:
            try:
                _publish_no_clobber(temporary_report, report_path, "report")
            except Exception as exc:
                raise RuntimeError(
                    "Control ZIP was published and left intact, but its report "
                    f"could not be published: output={output}, report={report_path}"
                ) from exc
        return report
    finally:
        temporary_output.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = make_region_off_control(
        args.source,
        args.output,
        args.expected_source_sha256,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
