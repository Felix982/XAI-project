# evaluation/evaluate_generated_samples.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict

import torch

from models.classifier import SmallCNNClassifier
from sampling.sample_diffusion import SampleConfig, sample_class_conditional


@dataclass
class GeneratedSampleEvalConfig:
    # Diffusion checkpoint
    diffusion_checkpoint_path: str = "./outputs/diffusion/diffusion_best.pt"
    pretrained_repo_id: str = "google/ddpm-cifar10-32"

    # Classifier checkpoint
    classifier_checkpoint_path: str = "./outputs/classifier/classifier_best.pt"

    # Data/model settings
    image_size: int = 32
    num_channels: int = 3
    num_classes: int = 2

    # Sampling settings
    num_samples_per_class: int = 64
    num_inference_steps: int = 50
    num_train_timesteps: int = 1000
    seed: int = 42

    # Classifier architecture settings
    classifier_base_channels: int = 32
    classifier_dropout: float = 0.1

    # Output
    output_dir: str = "./outputs/generated_eval"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier(cfg: GeneratedSampleEvalConfig) -> SmallCNNClassifier:
    """
    Rebuild the classifier and load its checkpoint.
    """
    model = SmallCNNClassifier(
        in_channels=cfg.num_channels,
        base_channels=cfg.classifier_base_channels,
        dropout=cfg.classifier_dropout,
    ).to(cfg.device)

    ckpt = torch.load(cfg.classifier_checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_one_target_class(
    target_class: int,
    classifier: SmallCNNClassifier,
    cfg: GeneratedSampleEvalConfig,
) -> Dict[str, float]:
    """
    Generate samples for one target class, then score them with the classifier.
    """
    sample_cfg = SampleConfig(
        checkpoint_path=cfg.diffusion_checkpoint_path,
        pretrained_repo_id=cfg.pretrained_repo_id,
        output_dir=cfg.output_dir,
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
        num_inference_steps=cfg.num_inference_steps,
        num_train_timesteps=cfg.num_train_timesteps,
        batch_size=cfg.num_samples_per_class,
        class_label=target_class,
        seed=cfg.seed,
        device=cfg.device,
    )

    # Generate samples in [-1, 1].
    samples = sample_class_conditional(sample_cfg)

    # Classifier returns one logit for binary classification.
    logits = classifier(samples)
    probs_class1 = torch.sigmoid(logits)
    preds = (probs_class1 >= 0.5).long()

    # For binary classification:
    # - probability of class 1 = sigmoid(logit)
    # - probability of class 0 = 1 - sigmoid(logit)
    if target_class == 1:
        target_probs = probs_class1
    else:
        target_probs = 1.0 - probs_class1

    target_consistency = (preds == target_class).float().mean().item()

    return {
        "target_class": target_class,
        "num_samples": cfg.num_samples_per_class,
        "target_consistency": target_consistency,
        "mean_target_probability": target_probs.mean().item(),
        "mean_prob_class1": probs_class1.mean().item(),
    }


def evaluate_generated_samples(cfg: GeneratedSampleEvalConfig) -> Dict[str, object]:
    """
    Evaluate conditional generation quality using the trained classifier.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    classifier = load_classifier(cfg)

    results_per_class = {}
    for target_class in range(cfg.num_classes):
        metrics = evaluate_one_target_class(
            target_class=target_class,
            classifier=classifier,
            cfg=cfg,
        )
        results_per_class[f"class_{target_class}"] = metrics

        print(
            f"class={target_class} | "
            f"consistency={metrics['target_consistency']:.4f} | "
            f"mean_target_prob={metrics['mean_target_probability']:.4f}"
        )

    # Average consistency across classes.
    avg_consistency = sum(
        m["target_consistency"] for m in results_per_class.values()
    ) / cfg.num_classes

    summary = {
        "config": asdict(cfg),
        "per_class": results_per_class,
        "average_target_consistency": avg_consistency,
    }

    with open(os.path.join(cfg.output_dir, "generated_sample_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary