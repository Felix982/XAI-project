# sampling/run_visual_comparison.py
from __future__ import annotations

import argparse
import json
import os
from typing import List

from sampling.sample_diffusion import SampleConfig, generate_and_save_samples


def delta_to_name(delta: float) -> str:
    s = f"{delta:+.3f}"
    s = s.replace("+", "plus_").replace("-", "minus_").replace(".", "p")
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual comparisons for corrected vs uncorrected sampling.")
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.1, 0.2, -0.2],
        help="Delta values to evaluate.",
    )
    parser.add_argument("--base-output-dir", type=str, default="./outputs/experiments")
    parser.add_argument("--checkpoint-name", type=str, default="diffusion_best.pt")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--pretrained-repo-id", type=str, default="google/ddpm-cifar10-32")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary: List[dict] = []

    for delta in args.deltas:
        run_name = f"delta_{delta_to_name(delta)}"
        train_dir = os.path.join(args.base_output_dir, run_name, "train")
        checkpoint_path = os.path.join(train_dir, args.checkpoint_name)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        for use_corrected in [False, True]:
            strategy_name = "corrected" if use_corrected else "uncorrected"
            sample_dir = os.path.join(
                args.base_output_dir,
                run_name,
                f"samples_{strategy_name}",
            )

            cfg = SampleConfig(
                checkpoint_path=checkpoint_path,
                pretrained_repo_id=args.pretrained_repo_id,
                output_dir=sample_dir,
                num_classes=args.num_classes,
                image_size=args.image_size,
                num_channels=args.num_channels,
                delta=delta,
                use_corrected=use_corrected,
                num_inference_steps=args.num_inference_steps,
                num_train_timesteps=args.num_train_timesteps,
                batch_size=args.batch_size,
                class_label=args.class_label,
                seed=args.seed,
                device=args.device if args.device is not None else SampleConfig.device,
            )

            print("=" * 80)
            print(f"Sampling delta={delta:+.3f} | strategy={strategy_name}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Output dir: {sample_dir}")
            print("=" * 80)

            saved_paths = generate_and_save_samples(cfg)
            summary.append(
                {
                    "delta": delta,
                    "strategy": strategy_name,
                    "checkpoint_path": checkpoint_path,
                    "output_dir": sample_dir,
                    "saved_paths": saved_paths,
                    "seed": args.seed,
                }
            )

    summary_path = os.path.join(args.base_output_dir, "visual_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved sampling summary to: {summary_path}")


if __name__ == "__main__":
    main()