import os
import sys
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Average weights of multiple PyTorch checkpoints (Model Soup/SWA)")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of checkpoint paths to average")
    parser.add_argument("--output", type=str, default="./data/checkpoints/model_soup.pth", help="Output checkpoint path")
    args = parser.parse_args()

    checkpoint_paths = args.inputs
    output_path = args.output

    if not checkpoint_paths:
        print("❌ No checkpoints provided for averaging.")
        sys.exit(1)

    print(f"Averaging {len(checkpoint_paths)} checkpoints...")
    
    # Load first checkpoint to initialize state dict and metadata
    first_path = checkpoint_paths[0]
    if not os.path.exists(first_path):
        print(f"❌ Checkpoint not found: {first_path}")
        sys.exit(1)

    first_ckpt = torch.load(first_path, map_location="cpu")
    
    is_state_dict_wrapped = "model_state_dict" in first_ckpt
    if is_state_dict_wrapped:
        averaged_state = {k: v.clone().float() for k, v in first_ckpt["model_state_dict"].items()}
        metadata = first_ckpt.copy()
        del metadata["model_state_dict"]
    else:
        averaged_state = {k: v.clone().float() for k, v in first_ckpt.items()}
        metadata = {}

    for path in checkpoint_paths[1:]:
        if not os.path.exists(path):
            print(f"❌ Checkpoint not found: {path}, skipping.")
            continue
        print(f"  Adding {path}...")
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        for k, v in state.items():
            if k in averaged_state:
                averaged_state[k] += v.float()
            else:
                print(f"⚠️ Warning: key {k} not found in first checkpoint, skipping.")

    num_ckpts = len(checkpoint_paths)
    for k in averaged_state:
        averaged_state[k] /= num_ckpts
        # Cast back to original dtype
        orig_dtype = first_ckpt.get("model_state_dict", first_ckpt)[k].dtype
        averaged_state[k] = averaged_state[k].to(orig_dtype)

    if is_state_dict_wrapped:
        output_ckpt = metadata
        output_ckpt["model_state_dict"] = averaged_state
    else:
        output_ckpt = averaged_state

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"🎉 Averaged checkpoint saved successfully to {output_path}!")

if __name__ == "__main__":
    main()
