from __future__ import annotations

import torch
from torch import nn


class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        conv_channels: list[int] | tuple[int, ...] = (32, 64),
        kernel_sizes: list[int] | tuple[int, ...] = (7, 5),
        pool_kernel: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError('conv_channels and kernel_sizes must match in length.')

        conv_layers: list[nn.Module] = []
        in_channels = input_size
        self.pool_kernel = pool_kernel
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes, strict=False):
            padding = kernel_size // 2
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_kernel),
                ]
            )
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def _downsample_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        down = lengths.clone()
        num_pools = sum(isinstance(m, nn.MaxPool1d) for m in self.conv)
        for _ in range(num_pools):
            down = torch.div(down, self.pool_kernel, rounding_mode='floor').clamp(min=1)
        return down

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        down_lengths = self._downsample_lengths(lengths)
        packed = nn.utils.rnn.pack_padded_sequence(x, down_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        features = self.dropout(features)
        return self.fc(features)
