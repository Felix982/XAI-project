# training/train_diffusion.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import DDPMScheduler

from data.medmnist import get_dataloaders
from models.conditional_unet import build_class_conditional_unet_from_pretrained

from torch.utils.tensorboard import SummaryWriter


@dataclass
class DiffusionTrainConfig:
    # Data
    data_root: str = "./data"
    image_size: int = 32
    num_channels: int = 3
    num_classes: int = 2
    batch_size: int = 64
    num_workers: int = 2
    
    # delta
    delta: float = 0.0
   
    # Model
    pretrained_repo_id: str = "google/ddpm-cifar10-32"

    # Diffusion
    num_train_timesteps: int = 1000

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 10

    # Logging / saving
    output_dir: str = "./outputs/diffusion"
    best_model_name: str = "diffusion_best.pt"
    last_model_name: str = "diffusion_last.pt"
    history_json_name: str = "history.json"
    history_csv_name: str = "history.csv"
    final_results_name: str = "final_results.json"
    tensorboard_subdir: str = "tb"
    # Early stopping
    early_stopping_patience: int = 5
    min_delta: float = 1e-4

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(num_train_timesteps: int) -> DDPMScheduler:
    # Standard DDPM scheduler for training-time noise addition.
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    cfg: DiffusionTrainConfig,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(cfg),
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    delta:float =0.0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x0 = batch["image"].to(device, non_blocking=True)   # clean image
        y = batch["label"].to(device, non_blocking=True)    # class labels
        bsz = x0.size(0)

        # Sample random noise.
        noise = torch.randn_like(x0) + delta

        # Sample a random timestep for each image.

        timesteps = torch.randint(
            low=0,
            high=scheduler.config.num_train_timesteps,
            size=(bsz,),
            device=device,
            dtype=torch.long,
        )

        # Forward diffusion: q(x_t | x_0).
        x_t = scheduler.add_noise(x0, noise, timesteps)

        optimizer.zero_grad(set_to_none=True)

        # Predict the noise from the noisy image and class label.
        noise_pred = model(x_t, timesteps, class_labels=y).sample

        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bsz
        total_samples += bsz

    return {
        "loss": total_loss / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    scheduler: DDPMScheduler,
    loader,
    device: str,
    delta: float=0.0,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x0 = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        bsz = x0.size(0)

        noise = torch.randn_like(x0)+delta

        timesteps = torch.randint(
            low=0,
            high=scheduler.config.num_train_timesteps,
            size=(bsz,),
            device=device,
            dtype=torch.long,
        )

        x_t = scheduler.add_noise(x0, noise, timesteps)
        noise_pred = model(x_t, timesteps, class_labels=y).sample

        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item() * bsz
        total_samples += bsz

    return {
        "loss": total_loss / max(total_samples, 1),
    }


def train_diffusion(cfg: DiffusionTrainConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    tb_dir = os.path.join(cfg.output_dir, cfg.tensorboard_subdir)
    writer = SummaryWriter(log_dir=tb_dir)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
        root=cfg.data_root,
        num_workers=cfg.num_workers,
        download=True,
    )

    model, load_info = build_class_conditional_unet_from_pretrained(
        repo_id=cfg.pretrained_repo_id,
        num_classes=cfg.num_classes,
        device=cfg.device,
    )

    with open(os.path.join(cfg.output_dir, "pretrained_load_info.json"), "w") as f:
        json.dump(load_info, f, indent=2)

    scheduler = build_scheduler(cfg.num_train_timesteps)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    best_model_path = os.path.join(cfg.output_dir, cfg.best_model_name)
    last_model_path = os.path.join(cfg.output_dir, cfg.last_model_name)

    for epoch in range(1, cfg.max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            scheduler=scheduler,
            loader=train_loader,
            optimizer=optimizer,
            device=cfg.device,
            delta=cfg.delta
        )

        val_metrics = evaluate(
            model=model,
            scheduler=scheduler,
            loader=val_loader,
            device=cfg.device,
            delta=cfg.delta
        )

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        history.append(epoch_stats)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}"
        )

        improved = (best_val_loss - val_metrics["loss"]) > cfg.min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_stats,
                cfg=cfg,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_stats,
            cfg=cfg,
        )

        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("config/delta", cfg.delta, epoch)

        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}"
            )
            break

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "tensorboard_dir": os.path.join(cfg.output_dir, "tb"),
        "history": history,
    }

    with open(os.path.join(cfg.output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    writer.flush()
    writer.close()
    return results