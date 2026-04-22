from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ecg_repo.utils.io import ensure_dir, write_json, write_yaml


def make_run_dir(runs_root: str | Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / f"{timestamp}_{experiment_name}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")
    return run_dir


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def save_run_metadata(
    run_dir: str | Path,
    config: dict[str, Any],
    history: dict[str, Any],
) -> None:
    run_dir = Path(run_dir)
    write_yaml(config, run_dir / "config_resolved.yaml")
    write_json(history, run_dir / "history.json")
