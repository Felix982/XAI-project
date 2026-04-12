# data/transforms.py

from torchvision import transforms


def get_image_transform(image_size: int = 32, num_channels: int = 1):
    """
    Build the image transform used by both classifier and diffusion code.

    Args:
        image_size: Final square image size.
        num_channels: 1 for grayscale, 3 for RGB.

    Returns:
        A torchvision transform pipeline.
    """
    if num_channels == 1:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                  # [0, 1], shape (1, H, W)
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
        ])

    if num_channels == 3:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                                  # [0, 1], shape (3, H, W) if as_rgb=True
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -> [-1, 1]
        ])

    raise ValueError(f"Unsupported num_channels={num_channels}")