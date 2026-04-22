from __future__ import annotations

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        channels: list[int] | tuple[int, ...] = (32, 64, 128),
        kernel_sizes: list[int] | tuple[int, ...] = (7, 5, 3),
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have the same length.")

        layers: list[nn.Module] = []
        in_channels = input_size
        for out_channels, kernel_size in zip(channels, kernel_sizes, strict=False):
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        feats = self.feature_extractor(x)
        feats = feats.transpose(1, 2)

        if lengths is not None:
            max_len = feats.shape[1]
            mask = torch.arange(max_len, device=feats.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)
            feats = feats.masked_fill(~mask, 0.0)
            pooled = feats.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)
        else:
            pooled = feats.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)
