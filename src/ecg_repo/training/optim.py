from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    name = config.get('optimizer', 'adam').lower()
    lr = float(config['learning_rate'])
    weight_decay = float(config.get('weight_decay', 0.0))

    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    name = config.get('scheduler', 'reduce_on_plateau')
    if name in (None, 'none'):
        return None
    if name == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('scheduler_mode', 'max'),
            factor=float(config.get('scheduler_factor', 0.5)),
            patience=int(config.get('scheduler_patience', 3)),
        )
    raise ValueError(f'Unsupported scheduler: {name}')
