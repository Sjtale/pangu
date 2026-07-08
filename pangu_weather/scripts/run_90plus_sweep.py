import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def make_log_file(path):
    if path is None:
        path = os.path.join("logs", f"90plus_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def run(cmd, *, env=None, dry_run=False):
    printable = " ".join(cmd)
    if env:
        prefix = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
        printable = f"{prefix} {printable}"
    print(f"\n$ {printable}")
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    process = subprocess.Popen(
        cmd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    print(f"\n[exit_code] {process.returncode}")
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def reset_output_dir(path, *, dry_run=False):
    print(f"\n# reset {path}")
    if dry_run:
        return
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def require_files(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Required source files are missing. Sync the latest pangu_weather code first:\n"
            f"  - {joined}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run local candidate generation for the 89.0614 -> 90+ route."
    )
    parser.add_argument(
        "--checkpoint",
        default="./data/checkpoints/model_pgw_lite_pruned_96_fp16.pth",
        help=(
            "Best FP16 distilled source checkpoint passed to quantize_mixed_precision.py. "
            "Do not use data/checkpoints/model_fp16.pth here; that is the quantized "
            "submission/runtime checkpoint."
        ),
    )
    parser.add_argument("--keep-counts", nargs="+", type=int, default=[4, 3])
    parser.add_argument("--mlp-chunks", nargs="+", type=int, default=[32768, 16384, 8192])
    parser.add_argument("--attn-chunks", nargs="+", type=int, default=[3])
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Fast screening mode: limit inference.py to the first N batches. 0 means full set.",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Reuse the current data/checkpoints/model_fp16.pth and only sweep runtime chunks.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Single file for the full sweep log. Defaults to logs/90plus_sweep_<timestamp>.log.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log_path = make_log_file(args.log_file)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"📝 Sweep log: {log_path}")

    require_files(
        [
            "inference.py",
            "pangu_profile_model.py",
            "calibration_utils.py",
            "scripts/calibrate_outputs.py",
            "scripts/apply_calibration_to_outputs.py",
            "scripts/run_calibration_pipeline.py",
            "scripts/quantize_mixed_precision.py",
        ]
    )

    if not args.dry_run and not args.skip_quantize and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Source checkpoint not found: {args.checkpoint}. "
            "Pass --checkpoint or use --skip-quantize."
        )

    keep_counts = args.keep_counts if not args.skip_quantize else [None]
    for keep_count in keep_counts:
        keep_label = f"keep{keep_count}" if keep_count is not None else "reuse"
        if keep_count is not None:
            run(
                [
                    sys.executable,
                    "scripts/quantize_mixed_precision.py",
                    "--keep-count",
                    str(keep_count),
                    "--checkpoint",
                    args.checkpoint,
                ],
                dry_run=args.dry_run,
            )

        runtime_index = 0
        for chunk_size in args.mlp_chunks:
            for attn_chunk_size in args.attn_chunks:
                candidate_label = (
                    f"{keep_label}_mlp{chunk_size}_attn{attn_chunk_size}"
                )
                print(f"\n# Candidate: {candidate_label}")
                env = {
                    "PANGU_MLP_CHUNK_SIZE": str(chunk_size),
                    "PANGU_CHUNKED_MLP": "1",
                    "PANGU_CHUNKED_ATTENTION": "1",
                    "PANGU_ATTN_CHUNK_SIZE": str(attn_chunk_size),
                    "PANGU_DISABLE_CUDA_GRAPH": "1",
                    "PANGU_LAYERWISE_INFERENCE": "1",
                }
                if args.max_batches > 0:
                    env["PANGU_MAX_INFERENCE_BATCHES"] = str(args.max_batches)
                if runtime_index == 0:
                    cmd = [sys.executable, "scripts/run_calibration_pipeline.py"]
                    if args.max_batches > 0:
                        cmd.extend(["--max-batches", str(args.max_batches)])
                    run(
                        cmd,
                        env=env,
                        dry_run=args.dry_run,
                    )
                else:
                    print(
                        "\n# Reusing calibration files from the first runtime candidate "
                        "for this keep-count; "
                        "runtime chunking is expected to be numerically equivalent."
                    )
                    reset_output_dir("./result/output", dry_run=args.dry_run)
                    run(
                        [
                            sys.executable,
                            "inference.py",
                            "--model_path",
                            "data/checkpoints/model_fp16.pth",
                        ],
                        env=env,
                        dry_run=args.dry_run,
                    )
                    run(
                        [sys.executable, "result.py"],
                        env={"PANGU_RESULT_OUTPUT_DIR": "./result/output"},
                        dry_run=args.dry_run,
                    )
                print(
                    "Record platform U/V/W after packaging this candidate before "
                    "moving to the next combination."
                )
                runtime_index += 1

    print(f"\n📝 Full sweep log saved to: {log_path}")


if __name__ == "__main__":
    main()
