# imports
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.medmnist import get_dataloader
from models.classifier import SmallCNNClassifier

@dataclass
class TrainConfig:
    data_root: str = "./data"
    image_size: int = 32
    batch_size: int = 64
    num_workers: int = 2

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30

    early_stopping_patience: int = 5
    min_delta: float = 1e-4

    base_channels: int = 32
    dropout: float = 0.1

    output_dir: str = "./outputs/classifier"
    best_model_name: str = "classifier_best.pt"
    last_model_name: str = "classifier_last.pt"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"




def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = get_dataloader(
        split="train",
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        root=cfg.data_root,
        shuffle=True,
        num_workers=cfg.num_workers,
        download=True,
    )
    val_loader = get_dataloader(
        split="val",
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        root=cfg.data_root,
        shuffle=False,
        num_workers=cfg.num_workers,
        download=True,
    )
    test_loader = get_dataloader(
        split="test",
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        root=cfg.data_root,
        shuffle=False,
        num_workers=cfg.num_workers,
        download=True,
    )
    return train_loader, val_loader, test_loader


def batch_to_device(batch: Dict[str, torch.Tensor], device: str):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].float().to(device, non_blocking=True)  # BCE wants float
    return images, labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images, labels = batch_to_device(batch, device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        labels_int = labels.long()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels_int).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images, labels = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            labels_int = labels.long()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels_int).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    cfg: TrainConfig,
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


def train_classifier(cfg: TrainConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_loader, val_loader, test_loader = build_loaders(cfg)

    model = SmallCNNClassifier(
        in_channels=1,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
    ).to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
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
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=cfg.device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=cfg.device,
        )

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_stats)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
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

        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}."
            )
            break

    checkpoint = torch.load(best_model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=cfg.device,
    )

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "history": history,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
    }

    with open(os.path.join(cfg.output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Done. test_loss={test_metrics['loss']:.4f}, "
        f"test_acc={test_metrics['accuracy']:.4f}"
    )

    return results


if __name__ == "__main__":
    cfg = TrainConfig()
    train_classifier(cfg)