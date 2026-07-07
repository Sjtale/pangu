import os
import sys
import subprocess
import shutil

def run_cmd(cmd, env=None):
    print(f"\n🚀 Running: {' '.join(cmd)}")
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    
    # Run process and stream output
    process = subprocess.Popen(cmd, env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    if process.returncode != 0:
        print(f"❌ Command failed with exit code: {process.returncode}")
        sys.exit(process.returncode)
    print("✅ Command finished successfully.")

def main():
    print("======================================================================")
    # Step 1: Run baseline inference on validation set (saving predictions to result/output/)
    print("Step 1: Running baseline inference on validation set to collect predictions...")
    
    # Ensure calibration coeffs file doesn't exist during initial collection
    coeff_path = "./data/checkpoints/calibration_coeffs.npy"
    if os.path.exists(coeff_path):
        print(f"ℹ️ Removing existing calibration file {coeff_path} to collect clean baseline predictions.")
        os.remove(coeff_path)
        
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
    
    run_cmd(["python", "inference.py", "--model_path", "data/checkpoints/model_fp16.pth"], env=infer_env)

    # Step 2: Run calibration script to compute optimal scaling factors
    print("\nStep 2: Calculating optimal channel-wise calibration scaling factors...")
    run_cmd(["python", "scripts/calibrate_outputs.py"])

    # Step 3: Run result evaluation to show the original baseline scores
    print("\nStep 3: Calculating original baseline scores...")
    run_cmd(["python", "result.py"])

    # Step 4: Re-run inference with calibration enabled to produce calibrated outputs
    print("\nStep 4: Running inference with calibration enabled to verify score improvement...")
    # Verify calibration file is present
    if not os.path.exists(coeff_path):
        print(f"❌ Error: {coeff_path} not found. Calibration failed!")
        sys.exit(1)
        
    # Clean output predictions for calibrated run
    if os.path.exists("result/output"):
        shutil.rmtree("result/output")
    os.makedirs("result/output", exist_ok=True)
        
    run_cmd(["python", "inference.py", "--model_path", "data/checkpoints/model_fp16.pth"], env=infer_env)

    # Step 5: Evaluate calibrated predictions to display final score boost
    print("\nStep 5: Calculating calibrated validation scores...")
    run_cmd(["python", "result.py"])
    
    print("\n======================================================================")
    print("🎉 Calibration pipeline execution completed!")
    print(f"Optimal scaling coefficients saved to {coeff_path}")
    print("Please make sure to package this file into checkpoints.zip for final platform submission.")
    print("======================================================================")

if __name__ == "__main__":
    main()
