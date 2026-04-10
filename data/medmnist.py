# data/medmnist.py

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
        root: str = "./data",
        download: bool = True,
        transform=None,
    ):
        self.split = split
        self.transform = transform or get_image_transform(image_size)

        self.dataset = PneumoniaMNIST(
            split=split,
            root=root,
            download=download,
            transform=self.transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.dataset[idx]

        # MedMNIST labels are often arrays like [0] or [1]
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
    root: str = "./data",
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    download: bool = True,
):
    dataset = PneumoniaMNISTDataset(
        split=split,
        image_size=image_size,
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