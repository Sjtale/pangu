"""Distill the organizer Pangu model into the fixed pruned_96 student.

This entry point intentionally supports one model only.  It consumes the
structured ``model_pgw_lite_pruned_96.pth`` initialization, trains the full
69-channel student, and writes ``model_pgw_lite_pruned_96_fp16.pth``.  The
submitted ``model_fp16_alias_compact.pth`` is produced from that checkpoint by
the documented conversion, mixed-precision, and alias-compaction steps.

Run from ``pangu_weather``::

    python distill_train.py
"""

import logging
import math
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from apex import optimizers

from onescience.datapipes.climate import ERA5Datapipe
from onescience.memory.checkpoint import replace_function
from onescience.utils.YParams import YParams
from pangu_profile_model import build_pangu_model


STUDENT_PROFILE = {
    "name": "pgw_lite_pruned_96",
    "patch_size": [2, 8, 8],
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "depth_blocks": [2, 6, 6, 2],
    "window_size": [2, 6, 12],
}
GROUND_TRUTH_WEIGHT = 0.3
TEACHER_WEIGHT = 0.5
HINT_WEIGHT = 0.2
HINT_LAYERS = ("layer1", "layer2")

LATEST_CHECKPOINT = "model_pgw_lite_pruned_96_latest.pth"
TRAIN_CHECKPOINT = "model_pgw_lite_pruned_96_train.pth"
FP16_CHECKPOINT = "model_pgw_lite_pruned_96_fp16.pth"
PRUNED_INITIALIZATION = "model_pgw_lite_pruned_96.pth"


def forecast_loss(surface, upper_air, target_surface, target_upper_air):
    """All-channel L1 used by the verified pruned_96 training protocol."""

    return F.l1_loss(upper_air, target_upper_air) + 0.25 * F.l1_loss(
        surface, target_surface
    )


def distillation_loss(student, target, teacher, hint_loss):
    hard_loss = forecast_loss(*student, *target)
    teacher_loss = forecast_loss(*student, *teacher)
    total = (
        GROUND_TRUTH_WEIGHT * hard_loss
        + TEACHER_WEIGHT * teacher_loss
        + HINT_WEIGHT * hint_loss
    )
    return total, hard_loss, teacher_loss


def checkpoint_path(cfg, name):
    return os.path.join(cfg.checkpoint_dir, name)


def get_student_profile(_cfg=None):
    """Return a copy so callers cannot mutate the fixed architecture."""

    return dict(STUDENT_PROFILE)


def get_teacher_profile(cfg):
    return {
        "name": "organizer_full_192",
        "patch_size": [int(value) for value in cfg.patch_size],
        "embed_dim": int(cfg.embed_dim),
        "num_heads": [int(value) for value in cfg.num_heads],
        "window_size": [int(value) for value in cfg.window_size],
    }


def load_state(model, path, strict=True):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=strict)
    return checkpoint


def dequantize_linear_weight_state(source_state, target_dtype=torch.float32):
    """Restore per-output-channel INT8 Linear tensors for offline inspection."""

    restored = OrderedDict()
    for key, value in source_state.items():
        clean_key = key.replace("module.", "")
        if clean_key.endswith("_scale"):
            continue
        scale = source_state.get(key + "_scale")
        if value.dtype == torch.int8 and isinstance(scale, torch.Tensor):
            shape = [value.shape[0]] + [1] * (value.dim() - 1)
            restored[clean_key] = (
                value.float() * scale.float().view(*shape)
            ).to(target_dtype)
        elif torch.is_floating_point(value):
            restored[clean_key] = value.to(target_dtype)
        else:
            restored[clean_key] = value
    return restored


def load_compatible_state(model, path, logger):
    """Load matching tensors from the organizer model into pruned_96.

    Normal training starts from the exact structured-pruning checkpoint and
    therefore uses strict loading.  This helper remains available for the
    pruning/audit utility that constructs that initialization.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    logger.info("Loaded %d/%d shape-compatible tensors from %s", len(compatible), len(target), path)
    return checkpoint


def make_model(cfg_data, profile, use_upgrades=False):
    """Build the fixed native Pangu architecture without experimental variants."""

    del use_upgrades
    return build_pangu_model(
        img_size=cfg_data.dataset.img_size,
        patch_size=profile["patch_size"],
        embed_dim=profile["embed_dim"],
        num_heads=profile["num_heads"],
        window_size=profile["window_size"],
        depth_blocks=profile.get("depth_blocks"),
        use_swiglu=False,
        use_rmsnorm=False,
        use_gqa=False,
        share_deep_blocks=False,
    )


def prepare_batch(data, surface_mask, device):
    invar, outvar = data[:2]
    invar_surface = invar[:, :4].to(device, dtype=torch.float32)
    invar_upper_air = invar[:, 4:].to(device, dtype=torch.float32)
    model_input = torch.cat([invar_surface, surface_mask, invar_upper_air], dim=1)
    target_surface = outvar[:, :4].to(device, dtype=torch.float32)
    target_upper_air = outvar[:, 4:].to(device, dtype=torch.float32)
    return model_input, target_surface, target_upper_air


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def feature_grids(img_size, patch_size):
    pressure = 1 + ceil_div(13, patch_size[0])
    layer1 = (
        pressure,
        ceil_div(img_size[0], patch_size[1]),
        ceil_div(img_size[1], patch_size[2]),
    )
    return {
        "layer1": layer1,
        "layer2": (pressure, ceil_div(layer1[1], 2), ceil_div(layer1[2], 2)),
    }


class FeatureCapture:
    def __init__(self, model, layers):
        self.features = {}
        self.handles = [
            getattr(model, layer).register_forward_hook(self._hook(layer))
            for layer in layers
        ]

    def _hook(self, name):
        def capture(_module, _inputs, output):
            self.features[name] = output

        return capture

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def tokens_to_grid(tokens, grid):
    if tokens.dim() != 3:
        raise ValueError(f"Expected [B, N, C] feature tensor, got {tuple(tokens.shape)}")
    batch, token_count, channels = tokens.shape
    expected = math.prod(grid)
    if token_count != expected:
        raise ValueError(
            f"Feature token count mismatch: got {token_count}, expected {expected}"
        )
    return tokens.transpose(1, 2).reshape(batch, channels, *grid)


def resize_channels(feature, channels):
    if feature.shape[1] == channels:
        return feature
    batch, source_channels, pressure, height, width = feature.shape
    series = feature.permute(0, 2, 3, 4, 1).reshape(-1, 1, source_channels)
    resized = F.interpolate(series.float(), size=channels, mode="linear", align_corners=False)
    return resized.reshape(batch, pressure, height, width, channels).permute(0, 4, 1, 2, 3)


def normalize_feature(feature):
    dims = tuple(range(2, feature.dim()))
    mean = feature.mean(dim=dims, keepdim=True)
    std = feature.std(dim=dims, keepdim=True, unbiased=False)
    return (feature - mean) / (std + 1.0e-4)


def feature_hint_loss(student_capture, teacher_capture, student_grids, teacher_grids):
    losses = []
    for layer in HINT_LAYERS:
        student = tokens_to_grid(student_capture.features[layer].float(), student_grids[layer])
        teacher = tokens_to_grid(teacher_capture.features[layer].float(), teacher_grids[layer])
        teacher = F.adaptive_avg_pool3d(teacher, student.shape[2:])
        teacher = resize_channels(teacher, student.shape[1])
        losses.append(
            F.l1_loss(normalize_feature(student), normalize_feature(teacher.detach()))
        )
    return torch.stack(losses).mean()


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        self._update_lr()

    def _update_lr(self):
        if self.current_step < self.warmup_steps:
            factor = self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state):
        self.current_step = int(state.get("current_step", self.current_step))
        self._update_lr()


def parameter_groups(model, base_lr, decay=0.95):
    groups = {name: [] for name in ("head", "layer4", "layer3", "layer2", "layer1")}
    for name, parameter in model.named_parameters():
        if "patchembed" in name or "patchrecovery" in name or "earth_position_bias_table" in name:
            group = "head"
        elif "layer4" in name or "upsample" in name:
            group = "layer4"
        elif "layer3" in name:
            group = "layer3"
        elif "layer2" in name or "downsample" in name:
            group = "layer2"
        else:
            group = "layer1"
        groups[group].append(parameter)
    rates = {
        "head": base_lr * 10.0,
        "layer4": base_lr,
        "layer3": base_lr * decay,
        "layer2": base_lr * decay**2,
        "layer1": base_lr * decay**3,
    }
    return [
        {"params": parameters, "lr": rates[name]}
        for name, parameters in groups.items()
        if parameters
    ]


def save_student(
    model,
    optimizer,
    scheduler,
    epoch,
    best_valid_loss,
    best_loss_epoch,
    cfg,
    train_name,
    export_fp16=False,
):
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
        "model_profile": dict(STUDENT_PROFILE),
    }
    train_path = checkpoint_path(cfg, train_name)
    torch.save(training_state, train_path + ".tmp")
    os.replace(train_path + ".tmp", train_path)
    if not export_fp16:
        return

    inference_state = {
        "model_state_dict": OrderedDict(
            (
                key,
                value.detach().half().cpu()
                if torch.is_floating_point(value)
                else value.detach().cpu(),
            )
            for key, value in model.state_dict().items()
        ),
        "model_profile": dict(STUDENT_PROFILE),
        "distillation": {
            "teacher_source": "organizer_pangu_full_model",
            "teacher_embed_dim": 192,
            "student_profile": STUDENT_PROFILE["name"],
            "student_embed_dim": STUDENT_PROFILE["embed_dim"],
            "ground_truth_weight": GROUND_TRUTH_WEIGHT,
            "teacher_weight": TEACHER_WEIGHT,
            "hint_weight": HINT_WEIGHT,
            "hint_layers": list(HINT_LAYERS),
            "all_69_channels": True,
            "predict_residual": False,
        },
    }
    export_path = checkpoint_path(cfg, FP16_CHECKPOINT)
    torch.save(inference_state, export_path + ".tmp")
    os.replace(export_path + ".tmp", export_path)


def build_validation_loader(valid_loader, stride):
    if stride <= 1:
        return valid_loader
    subset = torch.utils.data.Subset(
        valid_loader.dataset, range(0, len(valid_loader.dataset), stride)
    )
    return torch.utils.data.DataLoader(
        subset,
        batch_size=valid_loader.batch_size,
        shuffle=False,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=valid_loader.pin_memory,
        drop_last=valid_loader.drop_last,
    )


def main():
    handlers = [logging.StreamHandler(sys.stdout)]
    os.makedirs("logs", exist_ok=True)
    handlers.append(
        logging.FileHandler(
            os.path.join("logs", f"distill_train_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            encoding="utf-8",
        )
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    config_path = os.path.join(os.getcwd(), "conf/config.yaml")
    cfg = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    device = torch.device("cuda:0")

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    train_loader, _ = datapipe.train_dataloader()
    valid_loader, _ = datapipe.val_dataloader()
    valid_loader = build_validation_loader(valid_loader, int(getattr(cfg, "val_stride", 10)))

    static_dir = cfg_data.dataset.static_dir
    land_mask = torch.from_numpy(np.load(os.path.join(static_dir, "land_mask.npy")))
    soil_type = torch.from_numpy(np.load(os.path.join(static_dir, "soil_type.npy")))
    topography = torch.from_numpy(np.load(os.path.join(static_dir, "topography.npy")))
    topography = (topography - topography.mean()) / (
        topography.std(unbiased=False) + 1.0e-6
    )
    surface_mask = torch.stack([land_mask, soil_type, topography]).to(
        device, dtype=torch.float32
    )
    surface_mask = surface_mask.unsqueeze(0).repeat(
        cfg_data.dataloader.batch_size, 1, 1, 1
    )

    teacher_profile = get_teacher_profile(cfg)
    teacher = make_model(cfg_data, teacher_profile).half()
    local_teacher = checkpoint_path(cfg, "model_bak.pth")
    backup_teacher = os.path.join(cfg.official_checkpoint_dir, "model_bak.pth")
    teacher_path = local_teacher if os.path.isfile(local_teacher) else backup_teacher
    load_state(teacher, teacher_path, strict=True)
    teacher.to(device).eval().requires_grad_(False)

    student = make_model(cfg_data, STUDENT_PROFILE)
    latest_path = checkpoint_path(cfg, LATEST_CHECKPOINT)
    best_path = checkpoint_path(cfg, TRAIN_CHECKPOINT)
    pruned_path = checkpoint_path(cfg, PRUNED_INITIALIZATION)
    resume = os.path.isfile(latest_path) or os.path.isfile(best_path)
    initial_path = latest_path if os.path.isfile(latest_path) else best_path
    if not resume:
        initial_path = pruned_path
        if not os.path.isfile(initial_path):
            raise FileNotFoundError(
                f"Missing fixed pruned_96 initialization: {initial_path}. "
                "Run scripts/prune_structured.py with target profile pgw_lite_pruned_96 first."
            )
    student_checkpoint = load_state(student, initial_path, strict=True)
    student.to(device)

    base_lr = float(cfg.distill_learning_rate)
    optimizer = optimizers.FusedAdam(
        parameter_groups(student, base_lr),
        betas=(0.9, 0.999),
        weight_decay=3.0e-6,
    )
    steps_per_epoch = min(int(cfg.distill_steps_per_epoch), len(train_loader))
    total_epochs = int(cfg.distill_max_epoch)
    scheduler = WarmupCosineSchedule(
        optimizer,
        int(getattr(cfg, "distill_warmup_steps", 256)),
        total_epochs * steps_per_epoch,
    )

    best_valid_loss = float("inf")
    best_loss_epoch = -1
    start_epoch = 0
    if resume:
        optimizer.load_state_dict(student_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(student_checkpoint["scheduler_state_dict"])
        best_valid_loss = float(student_checkpoint["best_valid_loss"])
        best_loss_epoch = int(student_checkpoint["best_loss_epoch"])
        start_epoch = int(student_checkpoint["epoch"]) + 1
    del student_checkpoint

    teacher_capture = FeatureCapture(teacher, HINT_LAYERS)
    student_capture = FeatureCapture(student, HINT_LAYERS)
    teacher_grids = feature_grids(cfg_data.dataset.img_size, teacher_profile["patch_size"])
    student_grids = feature_grids(cfg_data.dataset.img_size, STUDENT_PROFILE["patch_size"])
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    logger.info(
        "pruned_96 distillation: teacher=%s init=%s weights=(%.1f, %.1f, %.1f)",
        teacher_path,
        initial_path,
        GROUND_TRUTH_WEIGHT,
        TEACHER_WEIGHT,
        HINT_WEIGHT,
    )

    try:
        for epoch in range(start_epoch, total_epochs):
            student.train()
            totals = {"loss": 0.0, "hard": 0.0, "teacher": 0.0, "hint": 0.0}
            started = time.time()
            for step, data in enumerate(train_loader, start=1):
                if step > steps_per_epoch:
                    break
                model_input, target_surface, target_upper_air = prepare_batch(
                    data, surface_mask, device
                )
                teacher_capture.clear()
                student_capture.clear()
                with torch.no_grad():
                    teacher_surface, teacher_upper_air = teacher(model_input.half())
                    teacher_surface = teacher_surface.float()
                    teacher_upper_air = teacher_upper_air.float().reshape(target_upper_air.shape)
                with replace_function(student, ["layer2", "layer3"], False):
                    student_surface, student_upper_air = student(model_input)
                student_upper_air = student_upper_air.reshape(target_upper_air.shape)
                hint_loss = feature_hint_loss(
                    student_capture, teacher_capture, student_grids, teacher_grids
                )
                loss, hard_loss, teacher_loss = distillation_loss(
                    (student_surface, student_upper_air),
                    (target_surface, target_upper_air),
                    (teacher_surface, teacher_upper_air),
                    hint_loss,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                totals["loss"] += loss.item()
                totals["hard"] += hard_loss.item()
                totals["teacher"] += teacher_loss.item()
                totals["hint"] += hint_loss.item()
                logger.info(
                    "Train %d-%d/%d %.2fs/step total=%.4f hard=%.4f teacher=%.4f hint=%.4f",
                    epoch,
                    step,
                    steps_per_epoch,
                    (time.time() - started) / step,
                    totals["loss"] / step,
                    totals["hard"] / step,
                    totals["teacher"] / step,
                    totals["hint"] / step,
                )

            student.eval()
            valid_loss = 0.0
            valid_steps = 0
            with torch.no_grad():
                for data in valid_loader:
                    model_input, target_surface, target_upper_air = prepare_batch(
                        data, surface_mask, device
                    )
                    output_surface, output_upper_air = student(model_input)
                    output_upper_air = output_upper_air.reshape(target_upper_air.shape)
                    valid_loss += forecast_loss(
                        output_surface,
                        output_upper_air,
                        target_surface,
                        target_upper_air,
                    ).item()
                    valid_steps += 1
            valid_loss /= max(1, valid_steps)
            improved = valid_loss < best_valid_loss
            if improved:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch
            save_student(
                student,
                optimizer,
                scheduler,
                epoch,
                best_valid_loss,
                best_loss_epoch,
                cfg,
                LATEST_CHECKPOINT,
            )
            if improved:
                save_student(
                    student,
                    optimizer,
                    scheduler,
                    epoch,
                    best_valid_loss,
                    best_loss_epoch,
                    cfg,
                    TRAIN_CHECKPOINT,
                    export_fp16=True,
                )
            logger.info(
                "Epoch %d validation=%.4f best=%.4f at epoch %d",
                epoch,
                valid_loss,
                best_valid_loss,
                best_loss_epoch,
            )
            if epoch - best_loss_epoch > int(cfg.patience):
                break
    finally:
        teacher_capture.close()
        student_capture.close()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
