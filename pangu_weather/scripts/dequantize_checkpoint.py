import os
import sys
import torch

# Add pangu_weather to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill_train import dequantize_linear_weight_state

def main():
    path = "./data/checkpoints/model_fp16.pth"
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")
        return

    print(f"Loading quantized checkpoint from: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Dequantize directly to float16
    dequantized_state = dequantize_linear_weight_state(state_dict, target_dtype=torch.float16)

    new_checkpoint = {
        "model_state_dict": dequantized_state,
        "model_profile": checkpoint.get("model_profile")
    }

    out_path = "./data/checkpoints/model_dequantized_fp16.pth"
    torch.save(new_checkpoint, out_path)
    print(f"Successfully saved mathematically dequantized FP16 checkpoint to: {out_path}")
    print(f"Original size: {os.path.getsize(path) / (1024**2):.2f} MB")
    print(f"Dequantized size: {os.path.getsize(out_path) / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
