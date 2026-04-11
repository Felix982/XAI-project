import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels, dropout=dropout),           # 32 -> 16
            ConvBlock(base_channels, base_channels * 2, dropout=dropout),     # 16 -> 8
            ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout), # 8 -> 4
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_channels * 4, 1)  # single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x).squeeze(1)  # shape: (B,)
        return logits