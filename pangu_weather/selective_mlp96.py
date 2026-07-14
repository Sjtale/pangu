"""Fixed profile and training rules for the SelectiveMLP-96 student."""

from __future__ import annotations

import re
from collections import Counter
from contextlib import contextmanager

import torch
import torch.nn.functional as F


PROFILE_NAME = "selective_mlp96"
MLP_RATIO_BLOCKS = [
    [4, 4],
    [4, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2],
    [4, 4],
]
PROFILE_SPEC = {
    "patch_size": [2, 8, 8],
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "depth_blocks": [2, 6, 6, 2],
    "window_size": [2, 6, 12],
    "mlp_ratio_blocks": MLP_RATIO_BLOCKS,
}
SOURCE_PROFILE = {
    "name": "pruned_96_source",
    "patch_size": [2, 8, 8],
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "depth_blocks": [2, 6, 6, 2],
    "window_size": [2, 6, 12],
}
TARGET_MLP_BLOCKS = (
    *(("layer2", block) for block in range(1, 6)),
    *(("layer3", block) for block in range(6)),
)
TARGET_MLP_PREFIXES = tuple(
    f"{stage}.Fuser.blocks.{block}.transformer.mlp"
    for stage, block in TARGET_MLP_BLOCKS
)
EXPECTED_PARAMETER_COUNT = 14_768_265
INITIALIZATION_METHOD = "pruned96_activation_aware_mlp_pair_selection"
SCORED_UPPER_INDICES = (2, 3, 5, 15, 16, 18, 28, 29, 31, 44, 57)

STAGE_PROTOCOLS = {
    "source_recovery": {
        "total_epochs": 1,
        "steps_per_epoch": 512,
        "warmup_steps": 64,
        "base_lr": 2.0e-5,
        "min_lr_ratio": 0.1,
    },
    "full_teacher": {
        "total_epochs": 3,
        "steps_per_epoch": 1024,
        "warmup_steps": 128,
        "base_lr": 5.0e-6,
        "min_lr_ratio": 0.1,
    },
}


def validate_profile(profile):
    """Fail closed unless every architecture field matches SelectiveMLP-96."""

    if profile.get("name") != PROFILE_NAME:
        raise ValueError(f"Expected profile {PROFILE_NAME!r}")
    if profile.get("share_deep_blocks"):
        raise ValueError("SelectiveMLP-96 forbids shared deep blocks")
    actual = {key: profile.get(key) for key in PROFILE_SPEC}
    if actual != PROFILE_SPEC:
        raise ValueError(
            f"SelectiveMLP-96 profile mismatch: actual={actual}, expected={PROFILE_SPEC}"
        )


def validate_initialization_metadata(metadata):
    """Validate the audited, zero-random initialization lineage."""

    if not isinstance(metadata, dict):
        raise ValueError("SelectiveMLP-96 checkpoint has no initialization metadata")
    expected = {
        "method": INITIALIZATION_METHOD,
        "profile_name": PROFILE_NAME,
        "source": "full_depth_ratio4_pruned96",
        "teacher": "official_full192",
        "random_initialized_parameters": 0,
        "mlp_ratio_blocks": MLP_RATIO_BLOCKS,
        "strict_coverage": True,
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"SelectiveMLP-96 initialization mismatch: actual={actual}, expected={expected}"
        )
    for key in ("source_sha256", "teacher_sha256", "init_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))):
            raise ValueError(f"SelectiveMLP-96 initialization has invalid {key}")
    if metadata.get("covered_parameter_state_keys") != metadata.get(
        "parameter_state_keys"
    ):
        raise ValueError("SelectiveMLP-96 initialization coverage is incomplete")
    if metadata.get("covered_state_tensor_keys") != metadata.get(
        "state_tensor_keys"
    ):
        raise ValueError("SelectiveMLP-96 state-tensor coverage is incomplete")
    selections = metadata.get("neuron_indices")
    if not isinstance(selections, dict) or set(selections) != set(
        TARGET_MLP_PREFIXES
    ):
        raise ValueError("SelectiveMLP-96 neuron selection metadata is incomplete")
    for key in TARGET_MLP_PREFIXES:
        indices = selections.get(key)
        if (
            not isinstance(indices, list)
            or len(indices) != 384
            or indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] >= 768
        ):
            raise ValueError(f"Invalid paired-neuron indices for {key}")


def canonical_parameter_key(key):
    """Collapse OneScience wrapper aliases that share one parameter storage."""

    normalized = str(key)
    for upper, lower in (
        ("Fuser", "fuser"),
        ("Sampler", "sampler"),
        ("Reconvery", "recovery"),
    ):
        if normalized.startswith(upper + "."):
            normalized = lower + normalized[len(upper) :]
        normalized = normalized.replace(f".{upper}.", f".{lower}.")
    return normalized


def validate_inference_state_load(model, state_dict, missing_keys, unexpected_keys):
    """Require one and only one source tensor for every trainable parameter."""

    parameter_keys = set(dict(model.named_parameters()).keys())
    canonical_parameters = {canonical_parameter_key(key) for key in parameter_keys}
    source_keys = set(state_dict)
    canonical_sources = [canonical_parameter_key(key) for key in source_keys]
    duplicate_sources = sorted(
        key for key, count in Counter(canonical_sources).items() if count > 1
    )
    missing_parameters = sorted(canonical_parameters - set(canonical_sources))
    unexpected = sorted(unexpected_keys)
    if missing_parameters or unexpected or duplicate_sources:
        raise RuntimeError(
            "SelectiveMLP-96 strict parameter load failed: "
            f"missing_parameters={missing_parameters}, unexpected={unexpected}, "
            f"duplicate_aliases={duplicate_sources}"
        )

    allowed_buffer_suffixes = ("earth_position_index", "attn_mask")
    invalid_missing_buffers = sorted(
        key
        for key in missing_keys
        if canonical_parameter_key(key) not in canonical_parameters
        and not key.endswith(allowed_buffer_suffixes)
    )
    if invalid_missing_buffers:
        raise RuntimeError(
            "SelectiveMLP-96 checkpoint omits non-regenerable buffers: "
            f"{invalid_missing_buffers}"
        )
    non_parameter_source = sorted(
        key
        for key in source_keys
        if canonical_parameter_key(key) not in canonical_parameters
    )
    if non_parameter_source:
        raise RuntimeError(
            "SelectiveMLP-96 inference checkpoint must be parameter-only: "
            f"{non_parameter_source}"
        )


def validate_stage_protocol(
    stage,
    *,
    total_epochs,
    steps_per_epoch,
    warmup_steps,
    base_lr,
    min_lr_ratio,
    checkpoint_interval,
    gradient_accumulation,
    ground_truth_weight,
    teacher_weight,
    hint_weight,
    hint_layers,
    score_aligned,
    score_project_quantized,
):
    """Lock the teacher-only two-stage recipe."""

    if stage not in STAGE_PROTOCOLS:
        raise ValueError(f"Unknown SelectiveMLP-96 stage: {stage!r}")
    actual = {
        "total_epochs": int(total_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_steps": int(warmup_steps),
        "base_lr": float(base_lr),
        "min_lr_ratio": float(min_lr_ratio),
    }
    if actual != STAGE_PROTOCOLS[stage]:
        raise ValueError(
            f"SelectiveMLP-96 {stage} protocol mismatch: "
            f"actual={actual}, expected={STAGE_PROTOCOLS[stage]}"
        )
    forbidden_or_fixed = {
        "checkpoint_interval": int(checkpoint_interval),
        "gradient_accumulation": int(gradient_accumulation),
        "ground_truth_weight": float(ground_truth_weight),
        "teacher_weight": float(teacher_weight),
        "hint_weight": float(hint_weight),
        "hint_layers": list(hint_layers),
        "score_aligned": bool(score_aligned),
        "score_project_quantized": bool(score_project_quantized),
    }
    expected = {
        "checkpoint_interval": 256,
        "gradient_accumulation": 1,
        "ground_truth_weight": 0.0,
        "teacher_weight": 1.0,
        "hint_weight": 0.0,
        "hint_layers": [],
        "score_aligned": False,
        "score_project_quantized": False,
    }
    if forbidden_or_fixed != expected:
        raise ValueError(
            "SelectiveMLP-96 must use teacher-only all-69 distillation: "
            f"actual={forbidden_or_fixed}, expected={expected}"
        )


def split_scored_channels(surface, upper_air):
    indices = torch.as_tensor(SCORED_UPPER_INDICES, device=upper_air.device)
    scored = torch.cat((surface, upper_air.index_select(1, indices)), dim=1)
    unscored_indices = torch.as_tensor(
        [
            index
            for index in range(upper_air.shape[1])
            if index not in SCORED_UPPER_INDICES
        ],
        device=upper_air.device,
    )
    return scored, upper_air.index_select(1, unscored_indices)


def teacher_only_loss(student, teacher):
    """Use only full-model/source-model outputs; labels never enter this loss."""

    student_scored, student_unscored = split_scored_channels(*student)
    teacher_scored, teacher_unscored = split_scored_channels(*teacher)
    scored = F.l1_loss(student_scored, teacher_scored.detach())
    unscored = F.l1_loss(student_unscored, teacher_unscored.detach())
    return 0.70 * scored + 0.30 * unscored, scored, unscored


def configure_trainable_parameters(model, stage):
    """Freeze only for source recovery; later stages train the entire student."""

    model.requires_grad_(stage != "source_recovery")
    if stage == "source_recovery":
        prefixes = tuple(
            f"{stage_name}.fuser.blocks.{block}.transformer.mlp."
            for stage_name, block in TARGET_MLP_BLOCKS
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise ValueError(f"No trainable parameters for SelectiveMLP-96 stage {stage}")
    return names


@contextmanager
def source_recovery_gradient_boundary(model):
    """Give reentrant layer2 checkpointing a leaf input after the frozen prefix."""

    base = model.module if hasattr(model, "module") else model
    downsample = getattr(base, "downsample", None)
    if not isinstance(downsample, torch.nn.Module):
        raise TypeError("SelectiveMLP-96 source recovery requires model.downsample")

    def detach_frozen_prefix(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("SelectiveMLP-96 downsample output must be a tensor")
        return output.detach().requires_grad_(True)

    handle = downsample.register_forward_hook(detach_frozen_prefix)
    try:
        yield
    finally:
        handle.remove()
