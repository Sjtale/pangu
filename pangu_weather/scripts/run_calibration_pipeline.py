import os
import sys
import subprocess
import shutil
import argparse
from datetime import datetime


def make_log_dir(path):
    if path is None:
        path = os.path.join(
            "logs", f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(path, exist_ok=True)
    return path


def safe_log_name(label):
    cleaned = []
    for char in label.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "command"


def run_cmd(cmd, env=None, log_dir=None, label=None):
    print(f"\n🚀 Running: {' '.join(cmd)}")
    current_env = os.environ.copy()
    if env:
        current_env.update(env)

    log_path = None
    log_file = None
    if log_dir is not None:
        log_path = os.path.join(log_dir, f"{safe_log_name(label or cmd[0])}.log")
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"$ {' '.join(cmd)}\n")
        if env:
            log_file.write("Environment overrides:\n")
            for key, value in sorted(env.items()):
                log_file.write(f"  {key}={value}\n")
        log_file.write("\n")
        log_file.flush()
        print(f"📝 Logging to {log_path}")

    process = subprocess.Popen(cmd, env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
        if log_file is not None:
            log_file.write(line)
    process.wait()
    if log_file is not None:
        log_file.write(f"\n[exit_code] {process.returncode}\n")
        log_file.close()

    if process.returncode != 0:
        print(f"❌ Command failed with exit code: {process.returncode}")
        if log_path is not None:
            print(f"❌ Full log: {log_path}")
        sys.exit(process.returncode)
    print("✅ Command finished successfully.")
    if log_path is not None:
        print(f"✅ Full log: {log_path}")


def print_calibration_diagnostics():
    print("Calibration file diagnostics:")
    for path in [
        "./data/checkpoints/calibration_coeffs.npy",
        "./data/checkpoints/calibration_affine.npz",
        "./data/checkpoints/physics_mean_targets.npz",
    ]:
        if os.path.exists(path):
            print(f"  present: {path} ({os.path.getsize(path)} bytes)")
        else:
            print(f"  missing: {path}")
    if os.path.exists("./result/output"):
        outputs = sorted(f for f in os.listdir("./result/output") if f.endswith(".npy"))
        print(f"  result/output npy files: {len(outputs)}")
        if outputs[:3]:
            print(f"  first outputs: {', '.join(outputs[:3])}")
    else:
        print("  missing: ./result/output")

def main():
    parser = argparse.ArgumentParser(description="Run prediction collection and calibration.")
    parser.add_argument(
        "--rerun-calibrated-inference",
        action="store_true",
        help="Old slow path: run inference.py a second time after calibration files are created.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Fast screening mode: limit inference.py to the first N batches. 0 means full set.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory for per-step logs. By default, no extra step logs are written.",
    )
    args = parser.parse_args()
    log_dir = make_log_dir(args.log_dir) if args.log_dir else None
    if log_dir:
        print(f"📝 Calibration pipeline logs: {log_dir}")

    print("======================================================================")
    # Step 1: Run baseline inference on validation set (saving predictions to result/output/)
    print("Step 1: Running baseline inference on validation set to collect predictions...")
    
    # Ensure calibration coeffs file doesn't exist during initial collection
    coeff_path = "./data/checkpoints/calibration_coeffs.npy"
    affine_path = "./data/checkpoints/calibration_affine.npz"
    physics_path = "./data/checkpoints/physics_mean_targets.npz"
    for calibration_path in [coeff_path, affine_path, physics_path]:
        if os.path.exists(calibration_path):
            print(f"ℹ️ Removing existing calibration file {calibration_path} to collect clean baseline predictions.")
            os.remove(calibration_path)
        
    # Clean previous output predictions
    if os.path.exists("result/output"):
        shutil.rmtree("result/output")
    os.makedirs("result/output", exist_ok=True)

    # Configure environment for standard inference without calibration
    infer_env = {
        "PANGU_DISABLE_CUDA_GRAPH": "1",
        "PANGU_LAYERWISE_INFERENCE": "1",
        "PANGU_CHUNKED_ATTENTION": "1",
        "PANGU_CHUNKED_MLP": "1",
    }
    if args.max_batches > 0:
        infer_env["PANGU_MAX_INFERENCE_BATCHES"] = str(args.max_batches)
    
    run_cmd(
        ["python", "inference.py", "--model_path", "data/checkpoints/model_fp16.pth"],
        env=infer_env,
        log_dir=log_dir,
        label="01_baseline_inference",
    )

    # Step 2: Run calibration script to compute optimal scaling factors
    print("\nStep 2: Calculating optimal channel-wise calibration scaling factors...")
    run_cmd(
        ["python", "scripts/calibrate_outputs.py"],
        log_dir=log_dir,
        label="02_calibrate_outputs",
    )

    # Step 3: Run result evaluation to show the original baseline scores
    print("\nStep 3: Calculating original baseline scores...")
    run_cmd(["python", "result.py"], log_dir=log_dir, label="03_result_raw")

    # Step 4: Produce calibrated outputs for local verification
    print("\nStep 4: Producing calibrated outputs for local verification...")
    # Verify calibration files are present
    if not os.path.exists(coeff_path) and not os.path.exists(affine_path):
        print(f"❌ Error: expected at least one of {coeff_path} or {affine_path}. Calibration failed!")
        print_calibration_diagnostics()
        sys.exit(1)
    if not os.path.exists(affine_path):
        print(f"⚠️ {affine_path} missing; continuing with legacy slope calibration only.")
        
    if args.rerun_calibrated_inference:
        print("ℹ️ --rerun-calibrated-inference enabled; running slow second inference.")
        if os.path.exists("result/output_calibrated"):
            shutil.rmtree("result/output_calibrated")
        if os.path.exists("result/output"):
            shutil.rmtree("result/output")
        os.makedirs("result/output", exist_ok=True)
        run_cmd(
            ["python", "inference.py", "--model_path", "data/checkpoints/model_fp16.pth"],
            env=infer_env,
            log_dir=log_dir,
            label="04_calibrated_inference",
        )
        calibrated_output_dir = "result/output"
    else:
        if os.path.exists("result/output_calibrated"):
            shutil.rmtree("result/output_calibrated")
        run_cmd(
            [
                "python",
                "scripts/apply_calibration_to_outputs.py",
                "--input-dir",
                "./result/output",
                "--output-dir",
                "./result/output_calibrated",
            ],
            log_dir=log_dir,
            label="04_apply_calibration",
        )
        calibrated_output_dir = "result/output_calibrated"

    # Step 5: Evaluate calibrated predictions to display final score boost
    print("\nStep 5: Calculating calibrated validation scores...")
    run_cmd(
        ["python", "result.py"],
        env={"PANGU_RESULT_OUTPUT_DIR": calibrated_output_dir},
        log_dir=log_dir,
        label="05_result_calibrated",
    )
    
    print("\n======================================================================")
    print("🎉 Calibration pipeline execution completed!")
    print(f"Logs saved under {log_dir}")
    if os.path.exists(coeff_path):
        print(f"Optimal scaling coefficients saved to {coeff_path}")
    if os.path.exists(affine_path):
        print(f"Optimal affine coefficients saved to {affine_path}")
    if os.path.exists(physics_path):
        print(f"Optional physics mean targets saved to {physics_path}")
    print("Please make sure to package this file into checkpoints.zip for final platform submission.")
    print("======================================================================")

if __name__ == "__main__":
    main()
