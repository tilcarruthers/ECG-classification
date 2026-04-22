from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecg_repo.evaluation.metrics import (
    compute_binary_normal_abnormal_metrics,
    compute_classification_metrics,
)
from ecg_repo.evaluation.reports import save_metrics_report, save_predictions
from ecg_repo.utils.checkpointing import save_checkpoint


@dataclass
class EpochResult:
    loss: float
    metrics: dict
    y_true: list[int]
    y_pred: list[int]
    beat_ids: list[str]
    record_ids: list[int]


def _forward(model: torch.nn.Module, batch: dict, device: torch.device):
    inputs = batch['inputs'].to(device)
    labels = batch['labels'].to(device)
    lengths = batch['lengths'].to(device)

    outputs = model(inputs, lengths) if 'lengths' in batch else model(inputs)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs
    return logits, labels


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    positive_labels: list[int] | tuple[int, ...] = (1, 2),
    grad_clip_norm: float | None = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []
    all_beat_ids: list[str] = []
    all_record_ids: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            logits, labels = _forward(model, batch, device)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.item())
            preds = logits.argmax(dim=1)

            all_true.extend(labels.detach().cpu().numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())
            all_beat_ids.extend(batch['beat_ids'])
            all_record_ids.extend(batch['record_ids'])

    y_true_np = np.asarray(all_true)
    y_pred_np = np.asarray(all_pred)

    multiclass = compute_classification_metrics(y_true_np, y_pred_np, labels=[0, 1, 2])
    binary = compute_binary_normal_abnormal_metrics(y_true_np, y_pred_np, positive_labels=positive_labels)
    metrics = {'multiclass': multiclass, 'binary_normal_abnormal': binary}

    return EpochResult(
        loss=total_loss / max(len(loader), 1),
        metrics=metrics,
        y_true=all_true,
        y_pred=all_pred,
        beat_ids=all_beat_ids,
        record_ids=all_record_ids,
    )


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    run_dir,
    num_epochs: int,
    monitor: str = 'val_macro_f1',
    early_stopping_patience: int = 6,
    positive_labels: list[int] | tuple[int, ...] = (1, 2),
    grad_clip_norm: float | None = None,
) -> dict:
    history = {'epochs': []}
    best_score = float('-inf')
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            positive_labels=positive_labels,
            grad_clip_norm=grad_clip_norm,
        )
        val_result = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            positive_labels=positive_labels,
        )

        val_macro_f1 = val_result.metrics['multiclass']['macro_f1']
        epoch_summary = {
            'epoch': epoch,
            'train_loss': train_result.loss,
            'val_loss': val_result.loss,
            'train_macro_f1': train_result.metrics['multiclass']['macro_f1'],
            'val_macro_f1': val_macro_f1,
            'train_metrics': train_result.metrics,
            'val_metrics': val_result.metrics,
        }
        history['epochs'].append(epoch_summary)

        if scheduler is not None:
            scheduler.step(val_macro_f1)

        current_score = val_macro_f1 if monitor == 'val_macro_f1' else -val_result.loss
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_summary,
                path=run_dir / 'artifacts' / 'best_model.pt',
            )
            save_metrics_report(val_result.metrics, run_dir / 'artifacts' / 'best_val_metrics.json')
            save_predictions(
                beat_ids=val_result.beat_ids,
                record_ids=val_result.record_ids,
                y_true=val_result.y_true,
                y_pred=val_result.y_pred,
                split='val',
                path=run_dir / 'artifacts' / 'val_predictions.csv',
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            break

    history['best_epoch'] = best_epoch
    history['best_score'] = best_score
    return history
