"""Distill the official Pangu model into a lightweight student profile.

Run from ``pangu_weather`` after generating the structured-pruning checkpoint,
or set ``PANGU_STUDENT_PROFILE=pgw_lite_patch8`` to train the PGW-Lite
patch-size student:

    python scripts/prune_structured.py
    python distill_train.py

The teacher is used only during training. The exported FP16 student can be
selected in inference with ``PANGU_USE_DISTILLED=1``.
"""

import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from apex import optimizers
from torch.nn.parallel import DistributedDataParallel

from onescience.datapipes.climate import ERA5Datapipe
from onescience.memory.checkpoint import replace_function
from onescience.utils.YParams import YParams
from pangu_profile_model import build_pangu_model


def weighted_l1(prediction, target, weights, level_weight=1.0):
    return level_weight * (
        F.l1_loss(prediction, target, reduction="none") * weights
    ).mean()


def forecast_loss(surface, upper_air, target_surface, target_upper_air, weights):
    surface_weights, pressure_weights = weights
    return weighted_l1(
        surface, target_surface, surface_weights, level_weight=0.25
    ) + weighted_l1(upper_air, target_upper_air, pressure_weights)


def distillation_loss(student, target, teacher, weights, ground_truth_weight):
    hard_loss = forecast_loss(*student, *target, weights)
    teacher_loss = forecast_loss(*student, *teacher, weights)
    total = ground_truth_weight * hard_loss + (1.0 - ground_truth_weight) * teacher_loss
    return total, hard_loss, teacher_loss


def checkpoint_path(cfg, name):
    return os.path.join(cfg.checkpoint_dir, name)


def cfg_list(value):
    return [int(v) for v in value]


def get_model_profile(cfg, profile_name):
    profiles = getattr(cfg, "student_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown student profile: {profile_name}")
    profile = profiles[profile_name]
    res = {
        "name": profile_name,
        "patch_size": cfg_list(profile.patch_size),
        "embed_dim": int(profile.embed_dim),
        "num_heads": cfg_list(profile.num_heads),
    }
    if hasattr(profile, "depth_blocks"):
        res["depth_blocks"] = cfg_list(profile.depth_blocks)
    return res


def get_default_profile(cfg):
    return {
        "name": "full_192",
        "patch_size": cfg_list(cfg.patch_size),
        "embed_dim": int(cfg.embed_dim),
        "num_heads": cfg_list(cfg.num_heads),
    }


def get_student_profile(cfg):
    profile_name = os.environ.get(
        "PANGU_STUDENT_PROFILE", getattr(cfg, "default_student_profile", "student_160")
    )
    return get_model_profile(cfg, profile_name)


def get_profile_checkpoint_names(cfg, profile):
    name = profile["name"]
    if name == "student_160":
        return {
            "latest": "model_distilled_latest.pth",
            "train": cfg.distilled_train_checkpoint,
            "inference": cfg.distilled_checkpoint,
        }
    if name == "pgw_lite_patch8":
        return {
            "latest": cfg.pgw_lite_distilled_latest_checkpoint,
            "train": cfg.pgw_lite_distilled_train_checkpoint,
            "inference": cfg.pgw_lite_distilled_checkpoint,
        }
    return {
        "latest": f"model_{name}_latest.pth",
        "train": f"model_{name}_train.pth",
        "inference": f"model_{name}_fp16.pth",
    }


def load_state(model, path, strict=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.pop("model_state_dict"), strict=strict)
    return checkpoint


def load_compatible_state(model, path, logger):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    compatible = {}
    warm_started_count = 0
    interpolated_count = 0

    for key, target_val in target_state.items():
        if key not in source_state:
            continue

        source_val = source_state[key]
        if tuple(source_val.shape) == tuple(target_val.shape):
            compatible[key] = source_val
            warm_started_count += 1
        elif ("patchembed" in key or "patchrecovery" in key) and key.endswith(".proj.weight"):
            # Conv3d / ConvTranspose3d weight interpolation
            s_shape = source_val.shape
            t_shape = target_val.shape
            reshaped = source_val.view(-1, 1, s_shape[-2], s_shape[-1])
            interpolated = F.interpolate(
                reshaped,
                size=(t_shape[-2], t_shape[-1]),
                mode='bicubic',
                align_corners=False
            ).view(t_shape)

            # Apply scale factor if it is standard convolution (patchembed) to preserve activation scale
            if "patchembed" in key:
                scale = (s_shape[-2] * s_shape[-1]) / (t_shape[-2] * t_shape[-1])
                compatible[key] = interpolated * scale
            else:
                compatible[key] = interpolated

            interpolated_count += 1
        elif "earth_position_bias_table" in key:
            # Earth-Specific Position Bias table interpolation
            n_tokens, src_wins, num_heads = source_val.shape
            tgt_wins = target_val.shape[1]
            if src_wins % 4 == 0 and tgt_wins % 4 == 0:
                src_lat_wins = src_wins // 4
                tgt_lat_wins = tgt_wins // 4
                reshaped = source_val.view(n_tokens, 4, src_lat_wins, num_heads)
                reshaped = reshaped.permute(0, 3, 1, 2).reshape(-1, 1, src_lat_wins)
                interpolated = F.interpolate(
                    reshaped,
                    size=tgt_lat_wins,
                    mode='linear',
                    align_corners=False
                )
                interpolated = interpolated.view(n_tokens, num_heads, 4, tgt_lat_wins)
                interpolated = interpolated.permute(0, 2, 3, 1).reshape(n_tokens, -1, num_heads)
            else:
                reshaped = source_val.permute(0, 2, 1).reshape(-1, 1, src_wins)
                interpolated = F.interpolate(
                    reshaped,
                    size=tgt_wins,
                    mode='linear',
                    align_corners=False
                )
                interpolated = interpolated.view(n_tokens, num_heads, tgt_wins).permute(0, 2, 1)
            compatible[key] = interpolated
            interpolated_count += 1

    model.load_state_dict(compatible, strict=False)
    logger.info(
        "Warm-started %d/%d tensors (including %d interpolated) from %s",
        warm_started_count + interpolated_count,
        len(target_state),
        interpolated_count,
        path,
    )
    return checkpoint



def save_student(
    model,
    optimizer,
    scheduler,
    epoch,
    best_valid_loss,
    best_loss_epoch,
    cfg,
    student_profile,
    train_checkpoint_name,
    inference_checkpoint_name=None,
):
    model_to_save = model.module if hasattr(model, "module") else model
    train_state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
        "model_profile": student_profile,
    }
    torch.save(train_state, checkpoint_path(cfg, train_checkpoint_name))

    if inference_checkpoint_name is None:
        return

    inference_state = {
        "model_state_dict": {
            key: value.detach().half().cpu()
            if torch.is_floating_point(value)
            else value.detach().cpu()
            for key, value in model_to_save.state_dict().items()
        },
        "distillation": {
            "teacher_embed_dim": int(cfg.embed_dim),
            "student_profile": student_profile["name"],
            "student_embed_dim": int(student_profile["embed_dim"]),
            "ground_truth_weight": float(cfg.distill_ground_truth_weight),
        },
        "model_profile": student_profile,
    }
    torch.save(inference_state, checkpoint_path(cfg, inference_checkpoint_name))


def prepare_batch(data, surface_mask, device):
    invar, outvar = data[:2]
    invar_surface = invar[:, :4].to(device, dtype=torch.float32)
    invar_upper_air = invar[:, 4:].to(device, dtype=torch.float32)
    model_input = torch.cat([invar_surface, surface_mask, invar_upper_air], dim=1)
    target_surface = outvar[:, :4].to(device, dtype=torch.float32)
    target_upper_air = outvar[:, 4:].to(device, dtype=torch.float32)
    return model_input, target_surface, target_upper_air


def make_model(cfg_data, profile):
    return build_pangu_model(
        img_size=cfg_data.dataset.img_size,
        patch_size=profile["patch_size"],
        embed_dim=profile["embed_dim"],
        num_heads=profile["num_heads"],
        window_size=profile["window_size"],
        depth_blocks=profile.get("depth_blocks", None),
    )


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # Store the initial learning rate for each parameter group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        self._update_lr()

    def _update_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            factor = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr_ratio': self.min_lr_ratio,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict.get('current_step', self.current_step)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.total_steps = state_dict.get('total_steps', self.total_steps)
        self.min_lr_ratio = state_dict.get('min_lr_ratio', self.min_lr_ratio)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self._update_lr()


def get_llrd_param_groups(model, base_lr, decay=0.95):
    head_params = []
    layer4_params = []
    layer3_params = []
    layer2_params = []
    layer1_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine parameter group based on layer depth
        if "patchembed" in name or "patchrecovery" in name or "earth_position_bias_table" in name:
            head_params.append(param)
        elif "layer4" in name or "upsample" in name:
            layer4_params.append(param)
        elif "layer3" in name:
            layer3_params.append(param)
        elif "layer2" in name or "downsample" in name:
            layer2_params.append(param)
        elif "layer1" in name:
            layer1_params.append(param)
        else:
            layer1_params.append(param)

    param_groups = [
        {"params": head_params, "lr": base_lr * 10.0},
        {"params": layer4_params, "lr": base_lr},
        {"params": layer3_params, "lr": base_lr * decay},
        {"params": layer2_params, "lr": base_lr * (decay ** 2)},
        {"params": layer1_params, "lr": base_lr * (decay ** 3)},
    ]

    # Remove empty groups to avoid optimizer warnings
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
    return param_groups


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    handlers = [logging.StreamHandler(sys.stdout)]
    if local_rank == 0:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"distill_train_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    current_path = os.getcwd()
    config_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    teacher_profile = get_default_profile(cfg)
    teacher_profile["window_size"] = cfg_list(cfg.window_size)
    student_profile = get_student_profile(cfg)
    student_profile["window_size"] = cfg_list(cfg.window_size)
    checkpoint_names = get_profile_checkpoint_names(cfg, student_profile)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    world_rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{local_rank}")

    datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized())
    train_loader, train_sampler = datapipe.train_dataloader()
    valid_loader, valid_sampler = datapipe.val_dataloader()
    val_stride = int(getattr(cfg, "val_stride", 10))
    if val_stride > 1:
        from torch.utils.data import Subset
        valid_indices = list(range(0, len(valid_loader.dataset), val_stride))
        valid_subset = Subset(valid_loader.dataset, valid_indices)
        if dist.is_initialized():
            from torch.utils.data.distributed import DistributedSampler
            valid_sampler = DistributedSampler(valid_subset, shuffle=False)
        else:
            valid_sampler = None
        valid_loader = torch.utils.data.DataLoader(
            valid_subset,
            batch_size=valid_loader.batch_size,
            shuffle=False,
            num_workers=valid_loader.num_workers,
            pin_memory=valid_loader.pin_memory,
            sampler=valid_sampler,
            drop_last=valid_loader.drop_last,
        )

    surface_weights = torch.as_tensor(
        cfg_data.dataset.weights[:4], device=device, dtype=torch.float32
    ).view(1, -1, 1, 1)
    pressure_weights = torch.as_tensor(
        cfg_data.dataset.weights[4:], device=device, dtype=torch.float32
    ).view(1, -1, 1, 1)
    weights = (surface_weights, pressure_weights)

    static_dir = cfg_data.dataset.static_dir
    land_mask = torch.from_numpy(np.load(os.path.join(static_dir, "land_mask.npy")))
    soil_type = torch.from_numpy(np.load(os.path.join(static_dir, "soil_type.npy")))
    topography = torch.from_numpy(np.load(os.path.join(static_dir, "topography.npy")))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography]).to(
        device, dtype=torch.float32
    )
    surface_mask = surface_mask.unsqueeze(0).repeat(
        cfg_data.dataloader.batch_size, 1, 1, 1
    )

    teacher = make_model(cfg_data, teacher_profile).half()
    local_teacher = checkpoint_path(cfg, "model_bak.pth")
    backup_teacher = os.path.join(cfg.official_checkpoint_dir, "model_bak.pth")
    teacher_path = local_teacher if os.path.exists(local_teacher) else backup_teacher
    load_state(teacher, teacher_path)
    teacher.to(device).eval()
    teacher.requires_grad_(False)

    student = make_model(cfg_data, student_profile)
    latest_train_checkpoint = checkpoint_names["latest"]
    latest_train_path = checkpoint_path(cfg, latest_train_checkpoint)
    distilled_train_path = checkpoint_path(cfg, checkpoint_names["train"])
    resume = os.path.exists(latest_train_path) or os.path.exists(distilled_train_path)

    name = student_profile["name"]
    pruned_start_path = checkpoint_path(cfg, f"model_{name}.pth")

    if os.path.exists(latest_train_path):
        initial_student_path = latest_train_path
    elif os.path.exists(distilled_train_path):
        initial_student_path = distilled_train_path
    elif os.path.exists(pruned_start_path):
        initial_student_path = pruned_start_path
        logger.info("Warm-starting training from pruned checkpoint: %s", pruned_start_path)
    elif name == "student_160" and os.path.exists(checkpoint_path(cfg, cfg.pruned_train_checkpoint)):
        initial_student_path = checkpoint_path(cfg, cfg.pruned_train_checkpoint)
    else:
        legacy_pruned = checkpoint_path(cfg, cfg.pruned_checkpoint)
        initial_student_path = legacy_pruned if name == "student_160" else teacher_path

    if os.path.exists(initial_student_path) and (
        initial_student_path != teacher_path or student_profile["name"] == "full_192"
    ):
        student_checkpoint = load_compatible_state(student, initial_student_path, logger)
    else:
        initial_student_path = teacher_path
        student_checkpoint = load_compatible_state(student, teacher_path, logger)
    student.to(device)

    # Group parameters with Layer-wise Learning Rate Decay (LLRD)
    base_lr = float(cfg.distill_learning_rate)
    param_groups = get_llrd_param_groups(student, base_lr, decay=0.95)
    optimizer = optimizers.FusedAdam(param_groups, betas=(0.9, 0.999), weight_decay=3e-6)

    # Initialize WarmupCosineSchedule scheduler
    steps_per_epoch = min(int(cfg.distill_steps_per_epoch), len(train_loader))
    total_epochs = int(cfg.distill_max_epoch)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(getattr(cfg, "distill_warmup_steps", 256))

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.01
    )

    best_valid_loss = float("inf")
    best_loss_epoch = 0
    start_epoch = 0
    if resume:
        try:
            optimizer.load_state_dict(student_checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(
                "Could not load optimizer state dict due to param_groups mismatch: %s. "
                "Continuing with newly initialized optimizer states.",
                str(e)
            )
        best_valid_loss = student_checkpoint["best_valid_loss"]
        best_loss_epoch = student_checkpoint["best_loss_epoch"]
        start_epoch = student_checkpoint.get("epoch", -1) + 1

        try:
            scheduler_state = student_checkpoint["scheduler_state_dict"]
            scheduler.load_state_dict(scheduler_state)
            if "current_step" not in scheduler_state:
                scheduler.current_step = start_epoch * steps_per_epoch
                scheduler._update_lr()
                logger.info(
                    "Old scheduler state dict format detected. Resetting current_step = %d based on start_epoch = %d",
                    scheduler.current_step,
                    start_epoch
                )
        except Exception as e:
            logger.warning("Could not load scheduler state dict: %s. Defaulting to calculated current_step.", str(e))
            scheduler.current_step = start_epoch * steps_per_epoch
            scheduler._update_lr()

    del student_checkpoint

    if dist.is_initialized():
        student = DistributedDataParallel(
            student, device_ids=[local_rank], output_device=local_rank
        )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ground_truth_weight = float(cfg.distill_ground_truth_weight)
    if not 0.0 <= ground_truth_weight <= 1.0:
        raise ValueError("distill_ground_truth_weight must be in [0, 1]")
    logger.info(
        "Distillation starts: teacher=%s, student_profile=%s, init=%s, "
        "data=%s, years=%s, "
        "ground_truth_weight=%.2f",
        teacher_path,
        student_profile["name"],
        initial_student_path,
        cfg_data.dataset.data_dir,
        cfg_data.dataset.train_ratio,
        ground_truth_weight,
    )

    steps_per_epoch = min(int(cfg.distill_steps_per_epoch), len(train_loader))
    for epoch in range(start_epoch, int(cfg.distill_max_epoch)):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

        student.train()
        train_total = train_hard = train_teacher = 0.0
        start_time = time.time()
        for step, data in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            model_input, target_surface, target_upper_air = prepare_batch(
                data, surface_mask, device
            )
            with torch.no_grad():
                teacher_surface, teacher_upper_air = teacher(model_input.half())
                teacher_surface = teacher_surface.float()
                teacher_upper_air = teacher_upper_air.float().reshape(target_upper_air.shape)

            with replace_function(student, ["layer2", "layer3"], dist.is_initialized()):
                student_surface, student_upper_air = student(model_input)
            student_upper_air = student_upper_air.reshape(target_upper_air.shape)
            loss, hard_loss, teacher_loss = distillation_loss(
                (student_surface, student_upper_air),
                (target_surface, target_upper_air),
                (teacher_surface, teacher_upper_air),
                weights,
                ground_truth_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_total += loss.item()
            train_hard += hard_loss.item()
            train_teacher += teacher_loss.item()
            if world_rank == 0:
                lrs = [group["lr"] for group in optimizer.param_groups]
                lr_str = ", ".join([f"{lr:.2e}" for lr in lrs[:3]])
                logger.info(
                    "Train %d-%d/%d [%.1fs/step] total=%.4f hard=%.4f teacher=%.4f | lrs: %s",
                    epoch,
                    step,
                    steps_per_epoch,
                    (time.time() - start_time) / step,
                    train_total / step,
                    train_hard / step,
                    train_teacher / step,
                    lr_str,
                )

        student.eval()
        valid_loss = 0.0
        val_count = 0
        start_val_time = time.time()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                val_count += 1
                model_input, target_surface, target_upper_air = prepare_batch(
                    data, surface_mask, device
                )
                output_surface, output_upper_air = student(model_input)
                output_upper_air = output_upper_air.reshape(target_upper_air.shape)
                loss = forecast_loss(
                    output_surface,
                    output_upper_air,
                    target_surface,
                    target_upper_air,
                    weights,
                )
                valid_loss += loss.item()
                if world_rank == 0 and val_count % 10 == 0:
                    logger.info(
                        "Valid step %d/%d loss=%.4f [%.2fs/step]",
                        val_count,
                        len(valid_loader),
                        valid_loss / val_count,
                        (time.time() - start_val_time) / val_count,
                    )

        if dist.is_initialized():
            loss_tensor = torch.tensor([valid_loss, float(val_count)], device=device)
            dist.all_reduce(loss_tensor)
            valid_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1.0)
        else:
            valid_loss /= max(val_count, 1)

        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
        if world_rank == 0:
            save_student(
                student,
                optimizer,
                scheduler,
                epoch,
                best_valid_loss,
                best_loss_epoch,
                cfg,
                student_profile,
                latest_train_checkpoint,
            )
        if is_best:
            if world_rank == 0:
                save_student(
                    student,
                    optimizer,
                    scheduler,
                    epoch,
                    best_valid_loss,
                    best_loss_epoch,
                    cfg,
                    student_profile,
                    checkpoint_names["train"],
                    checkpoint_names["inference"],
                )
        if world_rank == 0:
            logger.info(
                "Epoch %d: validation=%.4f, best=%.4f at epoch %d",
                epoch,
                valid_loss,
                best_valid_loss,
                best_loss_epoch,
            )
        if epoch - best_loss_epoch > cfg.patience:
            break


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
