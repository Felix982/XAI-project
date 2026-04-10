# data/transforms.py

from torchvision import transforms


def get_image_transform(image_size: int = 32):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),                  # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])