#!/usr/bin/env python3
"""Create an untrained FP16 screening student from the official Pangu model."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

PANGU_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANGU_DIR))

from distill_train import get_student_profile, load_compatible_state, make_model
from onescience.utils.YParams import YParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = PANGU_DIR / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    cfg = YParams(str(PANGU_DIR / "conf/config.yaml"), "model")
    cfg_data = YParams(str(PANGU_DIR / "conf/config.yaml"), "datapipe")
    os.environ["PANGU_STUDENT_PROFILE"] = args.profile
    profile = get_student_profile(cfg)
    profile["window_size"] = [int(value) for value in cfg.window_size]

    teacher = PANGU_DIR / cfg.official_checkpoint_dir / "model_bak.pth"
    local_teacher = PANGU_DIR / cfg.checkpoint_dir / "model_bak.pth"
    teacher = local_teacher if local_teacher.exists() else teacher
    if not teacher.exists():
        raise FileNotFoundError(f"Official teacher checkpoint not found: {teacher}")

    model = make_model(cfg_data, profile, use_upgrades=False)
    logger = logging.getLogger("prepare_official_uv_student")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_compatible_state(model, str(teacher), logger)
    state = {
        key: value.detach().half().cpu() if torch.is_floating_point(value) else value.cpu()
        for key, value in model.state_dict().items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "model_profile": profile,
            "distillation": {
                "teacher": "official_full_192",
                "student_profile": args.profile,
                "trained_epochs": 0,
            },
        },
        output,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"profile={args.profile}")
    print(f"parameters={parameters}")
    print(f"checkpoint_bytes={output.stat().st_size}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
