"""Distill the official Pangu model into the width-pruned student.

Run from ``pangu_weather`` after generating the structured-pruning checkpoint:

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
from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams


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


def load_state(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.pop("model_state_dict"), strict=True)
    return checkpoint


def save_student(
    model,
    optimizer,
    scheduler,
    epoch,
    best_valid_loss,
    best_loss_epoch,
    cfg,
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
            "student_embed_dim": int(cfg.pruned_embed_dim),
            "ground_truth_weight": float(cfg.distill_ground_truth_weight),
        },
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


def make_model(cfg, cfg_data, student):
    return Pangu(
        img_size=cfg_data.dataset.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.pruned_embed_dim if student else cfg.embed_dim,
        num_heads=cfg.pruned_num_heads if student else cfg.num_heads,
        window_size=cfg.window_size,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    current_path = os.getcwd()
    config_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    world_rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{local_rank}")

    datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized())
    train_loader, train_sampler = datapipe.train_dataloader()
    valid_loader, valid_sampler = datapipe.val_dataloader()
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

    teacher = make_model(cfg, cfg_data, student=False).half()
    local_teacher = checkpoint_path(cfg, "model_bak.pth")
    backup_teacher = os.path.join(cfg.official_checkpoint_dir, "model_bak.pth")
    teacher_path = local_teacher if os.path.exists(local_teacher) else backup_teacher
    load_state(teacher, teacher_path)
    teacher.to(device).eval()
    teacher.requires_grad_(False)

    student = make_model(cfg, cfg_data, student=True)
    latest_train_checkpoint = "model_distilled_latest.pth"
    latest_train_path = checkpoint_path(cfg, latest_train_checkpoint)
    distilled_train_path = checkpoint_path(cfg, cfg.distilled_train_checkpoint)
    pruned_train_path = checkpoint_path(cfg, cfg.pruned_train_checkpoint)
    pruned_path = checkpoint_path(cfg, cfg.pruned_checkpoint)
    resume = os.path.exists(latest_train_path) or os.path.exists(distilled_train_path)
    if os.path.exists(latest_train_path):
        initial_student_path = latest_train_path
    elif os.path.exists(distilled_train_path):
        initial_student_path = distilled_train_path
    elif os.path.exists(pruned_train_path):
        initial_student_path = pruned_train_path
    else:
        initial_student_path = pruned_path
    student_checkpoint = load_state(student, initial_student_path)
    student.to(device)

    optimizer = optimizers.FusedAdam(
        student.parameters(),
        betas=(0.9, 0.999),
        lr=float(cfg.distill_learning_rate),
        weight_decay=3e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.distill_max_epoch)
    )
    best_valid_loss = float("inf")
    best_loss_epoch = 0
    start_epoch = 0
    if resume:
        optimizer.load_state_dict(student_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(student_checkpoint["scheduler_state_dict"])
        best_valid_loss = student_checkpoint["best_valid_loss"]
        best_loss_epoch = student_checkpoint["best_loss_epoch"]
        start_epoch = student_checkpoint.get("epoch", -1) + 1
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
        "Distillation starts: teacher=%s, student=%s, data=%s, years=%s, "
        "ground_truth_weight=%.2f",
        teacher_path,
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
            train_total += loss.item()
            train_hard += hard_loss.item()
            train_teacher += teacher_loss.item()
            if world_rank == 0:
                logger.info(
                    "Train %d-%d/%d [%.1fs/step] total=%.4f hard=%.4f teacher=%.4f",
                    epoch,
                    step,
                    steps_per_epoch,
                    (time.time() - start_time) / step,
                    train_total / step,
                    train_hard / step,
                    train_teacher / step,
                )

        student.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
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
                if dist.is_initialized():
                    dist.all_reduce(loss)
                    loss /= world_size
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        scheduler.step()
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
                    cfg.distilled_train_checkpoint,
                    cfg.distilled_checkpoint,
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
