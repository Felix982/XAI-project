from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from medmnist import PneumoniaMNIST

from data.transforms import get_image_transform


class PneumoniaMNISTDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        image_size: int = 32,
        num_channels: int = 1,
        root: str = "./data",
        download: bool = True,
        transform=None,
    ):
        """
        Wrapper around MedMNIST PneumoniaMNIST.

        Args:
            split: 'train', 'val', or 'test'
            image_size: final square size
            num_channels: 1 for grayscale, 3 for RGB
            root: dataset directory
            download: whether to download if missing
            transform: optional custom transform
        """
        self.split = split
        self.num_channels = num_channels
        self.transform = transform or get_image_transform(
            image_size=image_size,
            num_channels=num_channels,
        )

        self.dataset = PneumoniaMNIST(
            split=split,
            root=root,
            download=download,
            transform=self.transform,
            as_rgb=(num_channels == 3),  # key change
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.dataset[idx]

        # MedMNIST labels can come as arrays like [0] or [1].
        if not torch.is_tensor(label):
            label = torch.tensor(label)

        label = label.squeeze().long()

        return {
            "image": image.float(),
            "label": label,
            "index": idx,
        }


def get_dataloader(
    split: str,
    batch_size: int = 64,
    image_size: int = 32,
    num_channels: int = 1,
    root: str = "./data",
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    download: bool = True,
):
    dataset = PneumoniaMNISTDataset(
        split=split,
        image_size=image_size,
        num_channels=num_channels,
        root=root,
        download=download,
    )

    if shuffle is None:
        shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def get_dataloaders(
    batch_size: int = 64,
    image_size: int = 32,
    num_channels: int = 1,
    root: str = "./data",
    num_workers: int = 2,
    download: bool = True,
):
    train_loader = get_dataloader(
        split="train",
        batch_size=batch_size,
        image_size=image_size,
        num_channels=num_channels,
        root=root,
        shuffle=True,
        num_workers=num_workers,
        download=download,
    )

    val_loader = get_dataloader(
        split="val",
        batch_size=batch_size,
        image_size=image_size,
        num_channels=num_channels,
        root=root,
        shuffle=False,
        num_workers=num_workers,
        download=download,
    )

    test_loader = get_dataloader(
        split="test",
        batch_size=batch_size,
        image_size=image_size,
        num_channels=num_channels,
        root=root,
        shuffle=False,
        num_workers=num_workers,
        download=download,
    )

    return train_loader, val_loader, test_loader