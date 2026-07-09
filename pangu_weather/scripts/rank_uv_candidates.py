#!/usr/bin/env python3
"""Rank U/V sweep candidates from local JSONL and optional platform CSV."""

import argparse
import csv
import json
from pathlib import Path


PLATFORM_FLOATS = ("u", "v", "w", "total", "package_size_mb", "max_vram_mb", "time_record_ms")


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_platform_csv(path):
    if not path:
        return {}
    platform = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("label")
            if not label:
                continue
            parsed = {key: _float_or_none(row.get(key)) for key in PLATFORM_FLOATS}
            parsed["notes"] = row.get("notes", "")
            platform[label] = parsed
    return platform


def merge_platform(rows, platform):
    merged = []
    for row in rows:
        item = dict(row)
        platform_row = platform.get(item.get("label"), {})
        for key, value in platform_row.items():
            item[f"platform_{key}"] = value
        merged.append(item)
    return merged


def _metric(row, key, default=float("inf")):
    value = row.get(key)
    return default if value is None else float(value)


def pareto_flags(rows):
    flags = {}
    ok_rows = [row for row in rows if row.get("returncode") == 0]
    for row in ok_rows:
        dominated = False
        row_metrics = (
            _metric(row, "max_vram_mb"),
            _metric(row, "latency_avg_ms"),
            _metric(row, "output_max_abs"),
        )
        for other in ok_rows:
            if other is row:
                continue
            other_metrics = (
                _metric(other, "max_vram_mb"),
                _metric(other, "latency_avg_ms"),
                _metric(other, "output_max_abs"),
            )
            if all(o <= r for o, r in zip(other_metrics, row_metrics)) and any(
                o < r for o, r in zip(other_metrics, row_metrics)
            ):
                dominated = True
                break
        flags[row["label"]] = not dominated
    return flags


def sort_key(row):
    platform_total = row.get("platform_total")
    if platform_total is not None:
        return (0, -float(platform_total), _metric(row, "max_vram_mb"), _metric(row, "latency_avg_ms"))
    if row.get("returncode") != 0:
        return (2, row.get("label", ""))
    return (
        1,
        _metric(row, "max_vram_mb"),
        _metric(row, "latency_avg_ms"),
        _metric(row, "output_max_abs"),
        row.get("label", ""),
    )


def format_row(row, is_pareto):
    return {
        "label": row.get("label"),
        "kind": row.get("kind"),
        "pareto": "yes" if is_pareto else "no",
        "platform_total": row.get("platform_total"),
        "platform_u": row.get("platform_u"),
        "platform_v": row.get("platform_v"),
        "platform_w": row.get("platform_w"),
        "max_vram_mb": row.get("max_vram_mb"),
        "reserved_mb": row.get("reserved_mb"),
        "latency_avg_ms": row.get("latency_avg_ms"),
        "output_max_abs": row.get("output_max_abs"),
        "returncode": row.get("returncode"),
    }


def print_markdown(rows, limit):
    headers = [
        "rank",
        "label",
        "pareto",
        "platform_total",
        "U",
        "V",
        "W",
        "max_vram_mb",
        "latency_avg_ms",
        "output_max_abs",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for index, row in enumerate(rows[:limit], start=1):
        values = [
            index,
            row.get("label"),
            row.get("pareto"),
            row.get("platform_total"),
            row.get("platform_u"),
            row.get("platform_v"),
            row.get("platform_w"),
            row.get("max_vram_mb"),
            row.get("latency_avg_ms"),
            row.get("output_max_abs"),
        ]
        print("| " + " | ".join("" if value is None else str(value) for value in values) + " |")


def print_csv(rows, limit):
    if not rows:
        return
    headers = list(rows[0].keys())
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows[:limit])


def main():
    parser = argparse.ArgumentParser(description="Rank U/V runtime sweep candidates.")
    parser.add_argument("jsonl")
    parser.add_argument("--platform-csv", default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = parser.parse_args()

    rows = merge_platform(load_jsonl(args.jsonl), load_platform_csv(args.platform_csv))
    flags = pareto_flags(rows)
    ranked = [format_row(row, flags.get(row.get("label"), False)) for row in sorted(rows, key=sort_key)]
    if args.format == "csv":
        print_csv(ranked, args.top)
    else:
        print_markdown(ranked, args.top)


if __name__ == "__main__":
    main()
