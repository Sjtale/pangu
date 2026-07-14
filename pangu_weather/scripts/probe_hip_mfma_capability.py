#!/usr/bin/env python3
"""Inventory HIP MFMA/WMMA evidence without installing or changing software."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HIPCC = "/opt/dtk/bin/hipcc"
HEADER_ROOTS = (Path("/opt/dtk/lib/clang"), Path("/opt/dtk/include"))
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".inc", ".cuh", ".def"}
TOOL_NAMES = (
    "hipcc",
    "hipconfig",
    "rocminfo",
    "hipprof",
    "rocprof",
    "rocprofv2",
    "llvm-objdump",
    "roc-objdump",
    "readelf",
)
TARGET_FLAG_CANDIDATES = ("--offload-arch", "--amdgpu-target")

MAX_COMMAND_OUTPUT_CHARS = 64 * 1024
MAX_HEADER_FILES = 4000
MAX_HEADER_FILE_BYTES = 1024 * 1024
MAX_HEADER_TOTAL_BYTES = 64 * 1024 * 1024
MAX_HEADER_MATCHES = 80
MAX_IDENTIFIERS = 160
MAX_SCAN_ERRORS = 12

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ARCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}$")
MFMA_WMMA_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:mfma|wmma)[A-Za-z0-9_]*\b",
    re.IGNORECASE,
)

NOT_RECOGNIZED_MARKER = "PANGU_CANDIDATE_BUILTIN_NOT_RECOGNIZED"
HAS_BUILTIN_UNAVAILABLE_MARKER = "PANGU_HAS_BUILTIN_OPERATOR_UNAVAILABLE"
INTERPRETATION = (
    "A successful __has_builtin check establishes only front-end recognition of "
    "the exact user-supplied identifier for the exact user-supplied target. It "
    "does not confirm an intrinsic signature, a compilable intrinsic call, emitted "
    "MFMA/WMMA ISA, runtime support, numerical correctness, or performance."
)


def _clip_output(value):
    value = value or ""
    if not isinstance(value, str):
        value = value.decode("utf-8", errors="replace")
    return value[:MAX_COMMAND_OUTPUT_CHARS], len(value) > MAX_COMMAND_OUTPUT_CHARS


def _run_command(argv, timeout_seconds, cwd=None):
    command = [str(item) for item in argv]
    base = {
        "argv": command,
        "timeout_seconds": timeout_seconds,
    }
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout, stdout_truncated = _clip_output(error.stdout)
        stderr, stderr_truncated = _clip_output(error.stderr)
        return (
            {
                **base,
                "status": "timeout",
                "returncode": None,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "error": f"command exceeded {timeout_seconds} seconds",
            },
            f"{error.stdout or ''}\n{error.stderr or ''}",
        )
    except (OSError, ValueError) as error:
        return (
            {
                **base,
                "status": "unavailable",
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
                "error": f"{type(error).__name__}: {error}",
            },
            "",
        )

    stdout, stdout_truncated = _clip_output(completed.stdout)
    stderr, stderr_truncated = _clip_output(completed.stderr)
    return (
        {
            **base,
            "status": "ok" if completed.returncode == 0 else "nonzero_exit",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        },
        f"{completed.stdout}\n{completed.stderr}",
    )


def _resolve_executable(requested):
    expanded = os.path.expanduser(requested)
    if os.sep in expanded:
        path = Path(expanded)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.absolute())
        return None
    return shutil.which(expanded)


def _collect_hipcc(requested):
    resolved = _resolve_executable(requested)
    if resolved is None:
        not_run = {
            "status": "not_run",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "reason": "hipcc executable was not found or is not executable",
        }
        return {
            "requested": requested,
            "resolved": None,
            "version": not_run,
            "help": dict(not_run),
            "advertised_target_flags": [],
        }

    version, _ = _run_command([resolved, "--version"], timeout_seconds=20)
    help_result, raw_help = _run_command([resolved, "--help"], timeout_seconds=20)
    advertised = [
        flag for flag in TARGET_FLAG_CANDIDATES if flag in raw_help
    ]
    return {
        "requested": requested,
        "resolved": resolved,
        "version": version,
        "help": help_result,
        "advertised_target_flags": advertised,
    }


def _collect_torch():
    report = {
        "status": "unconfirmed",
        "version": None,
        "hip_version": None,
        "cuda_version": None,
        "cuda_available": None,
        "device_count": None,
        "current_device": None,
        "devices": [],
    }
    try:
        import torch
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report

    report["version"] = getattr(torch, "__version__", None)
    torch_version = getattr(torch, "version", None)
    report["hip_version"] = getattr(torch_version, "hip", None)
    report["cuda_version"] = getattr(torch_version, "cuda", None)
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        report["error"] = "torch.cuda namespace is unavailable"
        return report

    try:
        report["cuda_available"] = bool(cuda.is_available())
        report["device_count"] = int(cuda.device_count())
        if report["cuda_available"] and report["device_count"]:
            report["current_device"] = int(cuda.current_device())
        for index in range(report["device_count"]):
            properties = cuda.get_device_properties(index)
            device = {"index": index, "repr": str(properties)}
            for source_name, output_name in (
                ("name", "name"),
                ("gcnArchName", "gcn_arch_name"),
                ("major", "major"),
                ("minor", "minor"),
                ("total_memory", "total_memory_bytes"),
                ("multi_processor_count", "multi_processor_count"),
                ("warp_size", "warp_size"),
            ):
                value = getattr(properties, source_name, None)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    device[output_name] = value
                else:
                    device[output_name] = str(value)
            report["devices"].append(device)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report

    report["status"] = "collected"
    return report


def _collect_command_v():
    return {name: shutil.which(name) for name in TOOL_NAMES}


def _scan_headers():
    matches = []
    identifiers = set()
    errors = []
    existing_roots = []
    missing_roots = []
    files_scanned = 0
    files_skipped_large = 0
    bytes_scanned = 0
    truncated = False

    def record_walk_error(error):
        if len(errors) < MAX_SCAN_ERRORS:
            errors.append(f"{type(error).__name__}: {error}")

    stop = False
    for root in HEADER_ROOTS:
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        existing_roots.append(str(root))
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, onerror=record_walk_error, followlinks=False
        ):
            directory_names.sort()
            file_names.sort()
            for file_name in file_names:
                path = Path(directory) / file_name
                if path.suffix.lower() not in HEADER_SUFFIXES:
                    continue
                if files_scanned >= MAX_HEADER_FILES:
                    truncated = True
                    stop = True
                    break
                try:
                    file_bytes = path.stat().st_size
                except OSError as error:
                    if len(errors) < MAX_SCAN_ERRORS:
                        errors.append(f"{type(error).__name__}: {path}: {error}")
                    continue
                files_scanned += 1
                if file_bytes > MAX_HEADER_FILE_BYTES:
                    files_skipped_large += 1
                    continue
                if bytes_scanned + file_bytes > MAX_HEADER_TOTAL_BYTES:
                    truncated = True
                    stop = True
                    break
                bytes_scanned += file_bytes
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as stream:
                        for line_number, line in enumerate(stream, start=1):
                            found = sorted(set(MFMA_WMMA_IDENTIFIER_RE.findall(line)))
                            if not found:
                                continue
                            for identifier in found:
                                if len(identifiers) < MAX_IDENTIFIERS:
                                    identifiers.add(identifier)
                            matches.append(
                                {
                                    "path": str(path),
                                    "line": line_number,
                                    "identifiers": found,
                                    "text": line.strip()[:400],
                                }
                            )
                            if len(matches) >= MAX_HEADER_MATCHES:
                                truncated = True
                                stop = True
                                break
                except OSError as error:
                    if len(errors) < MAX_SCAN_ERRORS:
                        errors.append(f"{type(error).__name__}: {path}: {error}")
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    if not existing_roots:
        status = "unconfirmed"
    elif truncated or errors or missing_roots:
        status = "limited"
    else:
        status = "complete"
    return {
        "status": status,
        "requested_roots": [str(root) for root in HEADER_ROOTS],
        "existing_roots": existing_roots,
        "missing_roots": missing_roots,
        "limits": {
            "max_header_files": MAX_HEADER_FILES,
            "max_file_bytes": MAX_HEADER_FILE_BYTES,
            "max_total_bytes": MAX_HEADER_TOTAL_BYTES,
            "max_match_lines": MAX_HEADER_MATCHES,
            "max_unique_identifiers": MAX_IDENTIFIERS,
        },
        "files_scanned": files_scanned,
        "files_skipped_large": files_skipped_large,
        "bytes_scanned": bytes_scanned,
        "truncated": truncated,
        "identifiers": sorted(identifiers),
        "matches": matches,
        "errors": errors,
        "auto_selected_candidate": None,
        "interpretation": (
            "Names found in headers are candidates only; this probe never selects "
            "or compiles one unless --candidate-builtin is supplied explicitly."
        ),
    }


def _validate_extra_flags(extra_flags):
    for flag in extra_flags:
        if not flag or "\x00" in flag or "\n" in flag or "\r" in flag:
            return "each --extra-flag must be one non-empty argv token"
        if flag.startswith("@"):
            return "response-file syntax is not accepted in --extra-flag"
        if flag == "-o" or flag.startswith("-o") or flag.startswith("--output"):
            return "output-path flags are reserved for the temporary probe output"
        if flag.startswith(("-MF", "-MJ", "--serialize-diagnostics")):
            return "diagnostic/dependency output paths are not accepted"
        if any(flag.startswith(candidate) for candidate in TARGET_FLAG_CANDIDATES):
            return "target flags must be supplied only through --arch"
    return None


def _probe_source(candidate_builtin):
    return f"""\
#include <hip/hip_runtime.h>

#ifndef __has_builtin
#error \"{HAS_BUILTIN_UNAVAILABLE_MARKER}\"
#endif

#if defined(__HIP_DEVICE_COMPILE__)
#if !__has_builtin({candidate_builtin})
#error \"{NOT_RECOGNIZED_MARKER}\"
#endif
#endif

extern \"C\" __global__ void pangu_has_builtin_frontend_probe() {{
    return;
}}
"""


def _frontend_probe(args, hipcc):
    base = {
        "status": "unconfirmed",
        "executed": False,
        "candidate_builtin": args.candidate_builtin,
        "arch": args.arch,
        "extra_flags": list(args.extra_flag),
        "target_flag": None,
        "frontend_builtin_recognized": None,
        "builtin_signature_confirmed": False,
        "intrinsic_call_compiled": False,
        "isa_emission_confirmed": False,
        "interpretation": INTERPRETATION,
    }
    missing = []
    if not args.candidate_builtin:
        missing.append("--candidate-builtin")
    if not args.arch:
        missing.append("--arch")
    if missing:
        base["reason"] = "explicit arguments required: " + ", ".join(missing)
        return base
    if not IDENTIFIER_RE.fullmatch(args.candidate_builtin):
        base["reason"] = "--candidate-builtin must be one C/C++ identifier"
        return base
    if not ARCH_RE.fullmatch(args.arch):
        base["reason"] = "--arch contains unsupported characters or length"
        return base
    extra_flag_error = _validate_extra_flags(args.extra_flag)
    if extra_flag_error:
        base["reason"] = extra_flag_error
        return base
    if not hipcc["resolved"]:
        base["reason"] = "hipcc is unavailable"
        return base
    if not hipcc["advertised_target_flags"]:
        base["reason"] = (
            "hipcc --help did not advertise a supported target flag; refusing "
            "to guess how --arch should be passed"
        )
        return base

    target_flag = hipcc["advertised_target_flags"][0]
    base["target_flag"] = target_flag
    source = _probe_source(args.candidate_builtin)
    base["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with tempfile.TemporaryDirectory(prefix="pangu_hip_mfma_probe_") as directory:
        temporary_root = Path(directory)
        source_path = temporary_root / "has_builtin_probe.hip"
        object_path = temporary_root / "has_builtin_probe.o"
        source_path.write_text(source, encoding="utf-8")
        command = [
            hipcc["resolved"],
            *args.extra_flag,
            f"{target_flag}={args.arch}",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ]
        command_result, raw_output = _run_command(
            command, timeout_seconds=60, cwd=directory
        )
        command_result["argv"] = [
            item.replace(directory, "<temporary-directory>")
            for item in command_result["argv"]
        ]
        object_created = object_path.is_file()

    base["executed"] = True
    base["compile"] = command_result
    base["object_created"] = object_created
    if command_result["returncode"] == 0 and object_created:
        base["status"] = "frontend_recognized"
        base["frontend_builtin_recognized"] = True
    elif NOT_RECOGNIZED_MARKER in raw_output:
        base["status"] = "frontend_not_recognized"
        base["frontend_builtin_recognized"] = False
    elif HAS_BUILTIN_UNAVAILABLE_MARKER in raw_output:
        base["reason"] = "the compiler does not expose __has_builtin for this probe"
    else:
        base["reason"] = (
            "the front-end compile did not complete cleanly; inspect compile output"
        )
    return base


def _default_output_path():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    logs = Path(__file__).resolve().parents[1] / "logs"
    return logs / f"hip_mfma_capability_{timestamp}_{os.getpid()}.json"


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hipcc",
        default=DEFAULT_HIPCC,
        help=f"hipcc path or executable name (default: {DEFAULT_HIPCC})",
    )
    parser.add_argument(
        "--arch",
        help="exact target architecture supplied by the user; never inferred",
    )
    parser.add_argument(
        "--candidate-builtin",
        help="exact builtin identifier supplied by the user; never inferred",
    )
    parser.add_argument(
        "--extra-flag",
        action="append",
        default=[],
        help="one additional hipcc argv token; repeat as needed",
    )
    parser.add_argument(
        "--output",
        help="new JSON path; defaults to a unique overwrite-protected logs file",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path = (
        Path(args.output).expanduser() if args.output else _default_output_path()
    )
    hipcc = _collect_hipcc(args.hipcc)
    frontend_probe = _frontend_probe(args, hipcc)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": frontend_probe["status"],
        "output_path": str(output_path.resolve()),
        "scope": {
            "network_used": False,
            "packages_installed": False,
            "environment_modified": False,
            "temporary_files_managed_by_tempfile": True,
        },
        "inputs": {
            "hipcc": args.hipcc,
            "arch": args.arch,
            "candidate_builtin": args.candidate_builtin,
            "extra_flags": list(args.extra_flag),
            "auto_selected_arch": None,
            "auto_selected_builtin": None,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "hipcc": hipcc,
        "torch": _collect_torch(),
        "command_v": _collect_command_v(),
        "header_scan": _scan_headers(),
        "frontend_compile_probe": frontend_probe,
        "conclusion": {
            "frontend_builtin_recognized": frontend_probe[
                "frontend_builtin_recognized"
            ],
            "builtin_signature_confirmed": False,
            "intrinsic_call_compiled": False,
            "isa_emission_confirmed": False,
            "interpretation": INTERPRETATION,
        },
    }
    try:
        _write_report(output_path, report)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite existing report: {output_path}")


if __name__ == "__main__":
    main()
