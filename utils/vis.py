import torch


def denormalize_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0