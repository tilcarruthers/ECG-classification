from __future__ import annotations

import torch
from torch import nn


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        attention_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, attention_hidden_size),
            nn.Tanh(),
            nn.Linear(attention_hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])

        logits = self.attention(lstm_out).squeeze(-1)
        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        logits = logits.masked_fill(mask, float('-inf'))
        weights = torch.softmax(logits, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)

        context = self.dropout(context)
        outputs = self.fc(context)
        return outputs, weights
