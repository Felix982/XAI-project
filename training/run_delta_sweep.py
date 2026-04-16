# training/run_delta_sweep.py
from __future__ import annotations

import argparse
import json
import os
from typing import List

from training.train_diffusion import DiffusionTrainConfig, train_diffusion


def delta_to_name(delta: float) -> str:
    s = f"{delta:+.3f}"
    s = s.replace("+", "plus_").replace("-", "minus_").replace(".", "p")
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diffusion fine-tuning for multiple delta values.")
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.1, 0.2, -0.2],
        help="List of delta values to train.",
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--base-output-dir", type=str, default="./outputs/experiments")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--pretrained-repo-id", type=str, default="google/ddpm-cifar10-32")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.base_output_dir, exist_ok=True)

    all_results: List[dict] = []

    for delta in args.deltas:
        run_name = f"delta_{delta_to_name(delta)}"
        output_dir = os.path.join(args.base_output_dir, run_name, "train")

        cfg = DiffusionTrainConfig(
            data_root=args.data_root,
            image_size=args.image_size,
            num_channels=args.num_channels,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            delta=delta,
            pretrained_repo_id=args.pretrained_repo_id,
            num_train_timesteps=args.num_train_timesteps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            output_dir=output_dir,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            seed=args.seed,
            device=args.device if args.device is not None else DiffusionTrainConfig.device,
        )

        print("=" * 80)
        print(f"Starting training for delta={delta:+.3f}")
        print(f"Output dir: {output_dir}")
        print("=" * 80)

        results = train_diffusion(cfg)
        all_results.append(
            {
                "delta": delta,
                "output_dir": output_dir,
                "best_model_path": results["best_model_path"],
                "last_model_path": results["last_model_path"],
                "best_epoch": results["best_epoch"],
                "best_val_loss": results["best_val_loss"],
                "tensorboard_dir": results["tensorboard_dir"],
            }
        )

    summary_path = os.path.join(args.base_output_dir, "delta_sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nDone.")
    print(f"Saved sweep summary to: {summary_path}")


if __name__ == "__main__":
    main()