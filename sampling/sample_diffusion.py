from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List

import torch
from diffusers import DDIMScheduler
from PIL import Image

from models.conditional_unet import build_class_conditional_unet_from_pretrained


@dataclass
class SampleConfig:
    checkpoint_path: str
    pretrained_repo_id: str = "google/ddpm-cifar10-32"
    output_dir: str = "./outputs/samples"
    num_classes: int = 2

    image_size: int = 32
    num_channels: int = 3

    #delta
    delta: float = 0.0

    num_inference_steps: int = 50
    num_train_timesteps: int = 1000

    batch_size: int = 8
    class_label: int = 0
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def denormalize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    Convert from [-1, 1] to [0, 255] uint8.
    Expects shape (B, C, H, W).
    """
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = (x * 255.0).round().to(torch.uint8)
    return x


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int | None = None) -> None:
    """
    Save a simple image grid from a tensor of shape (B, C, H, W), uint8.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    b, c, h, w = images.shape
    if nrow is None:
        nrow = int(math.sqrt(b))
        nrow = max(1, nrow)
    ncol = math.ceil(b / nrow)

    grid = Image.new("RGB", (nrow * w, ncol * h))

    for idx in range(b):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray(img)
        x = (idx % nrow) * w
        y = (idx // nrow) * h
        grid.paste(pil_img, (x, y))

    grid.save(save_path)


def load_finetuned_conditional_unet(cfg: SampleConfig):
    """
    Rebuild the same conditional UNet, then load the fine-tuned weights.
    """
    model, _ = build_class_conditional_unet_from_pretrained(
        repo_id=cfg.pretrained_repo_id,
        num_classes=cfg.num_classes,
        device=cfg.device,
    )

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def sample_class_conditional(cfg: SampleConfig) -> torch.Tensor:
    """
    Sample a batch of images conditioned on one class label.
    Returns images in [-1, 1], shape (B, C, H, W).
    """
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)

    model = load_finetuned_conditional_unet(cfg)

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
    )
    scheduler.set_timesteps(cfg.num_inference_steps)

    # Start from Gaussian noise.
    x = torch.randn(
        cfg.batch_size,
        cfg.num_channels,
        cfg.image_size,
        cfg.image_size,
        device=cfg.device,
    )
    # Add delta to target pf forward process z_T=x_T
    delta_tensor=torch.full_like(x, fill_value=cfg.delta)
    x+=delta_tensor

    class_labels = torch.full(
        (cfg.batch_size,),
        fill_value=cfg.class_label,
        device=cfg.device,
        dtype=torch.long,
    )

    for t in scheduler.timesteps:
        # Predict noise at current step.
        noise_pred = model(x, t, class_labels=class_labels).sample

        # DDIM update step.
        step_output = scheduler.step(noise_pred, t, x)
        x = step_output.prev_sample

        #add delta to mean mu by adding it to x
        x+=delta_tensor

    return x


def generate_and_save_samples(cfg: SampleConfig) -> List[str]:
    """
    Generate one batch and save:
    - one grid image
    - individual images
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    samples = sample_class_conditional(cfg)
    samples_uint8 = denormalize_to_uint8(samples)

    saved_paths = []

    # Save grid.
    grid_path = os.path.join(
        cfg.output_dir,
        f"samples_class_{cfg.class_label}_grid.png",
    )
    save_image_grid(samples_uint8, grid_path)
    saved_paths.append(grid_path)

    # Save individual images.
    for i in range(samples_uint8.shape[0]):
        img = samples_uint8[i].permute(1, 2, 0).cpu().numpy()
        img_path = os.path.join(
            cfg.output_dir,
            f"class_{cfg.class_label}_sample_{i:03d}.png",
        )
        Image.fromarray(img).save(img_path)
        saved_paths.append(img_path)

    return saved_paths