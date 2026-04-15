from __future__ import annotations

from typing import Tuple, Dict, Any

import torch
from diffusers import UNet2DModel


def build_class_conditional_unet_from_pretrained(
    repo_id: str = "google/ddpm-cifar10-32",
    num_classes: int = 2,
    device: str | None = None,
    
) -> Tuple[UNet2DModel, Dict[str, Any]]:
    """
    Load the pretrained unconditional UNet, then instantiate the same architecture
    with class conditioning enabled and copy over all matching pretrained weights.

    New class-embedding weights are initialized randomly.
    """
    # Load pretrained unconditional model.
    base_unet = UNet2DModel.from_pretrained(repo_id)

    # Copy config and enable class conditioning.
    cfg = dict(base_unet.config)
    cfg["num_class_embeds"] = num_classes
    # Keep class_embed_type=None: docs say this uses a learnable embedding matrix.
    # That is the simplest option for discrete labels.
    cfg["class_embed_type"] = None

    conditional_unet = UNet2DModel(**cfg)

    # Load pretrained weights wherever shapes match.
    load_result = conditional_unet.load_state_dict(base_unet.state_dict(), strict=False)

    if device is not None:
        conditional_unet = conditional_unet.to(device)

    info = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "base_config": cfg,
    }
    return conditional_unet, info


@torch.no_grad()
def sanity_check_conditional_unet(
    model: UNet2DModel,
    batch_size: int = 4,
    image_size: int = 32,
    num_train_timesteps: int = 1000,  # scheduler setting, not UNet config
    device: str = "cpu",
):
    """
    Quick shape test for the conditional UNet.
    """
    x = torch.randn(
        batch_size,
        model.config.in_channels,
        image_size,
        image_size,
        device=device,
    )

    # Sample random diffusion timesteps.
    t = torch.randint(
        low=0,
        high=num_train_timesteps,
        size=(batch_size,),
        device=device,
    )

    # Sample random class labels.
    y = torch.randint(
        low=0,
        high=model.config.num_class_embeds,
        size=(batch_size,),
        device=device,
    )

    out = model(x, t, class_labels=y)
    return out.sample.shape