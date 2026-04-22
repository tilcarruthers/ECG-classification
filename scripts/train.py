from __future__ import annotations

import argparse
from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecg_repo.data.collate import pad_collate
from ecg_repo.data.dataset import ECGBeatDataset
from ecg_repo.data.loading import get_records_list, infer_record_keys, load_ecg_dataset
from ecg_repo.models.bilstm import BiLSTMClassifier
from ecg_repo.models.bilstm_attention import BiLSTMAttentionClassifier
from ecg_repo.models.cnn1d import CNN1DClassifier
from ecg_repo.models.cnn_lstm import CNNLSTMClassifier
from ecg_repo.training.losses import compute_class_weights
from ecg_repo.training.optim import build_optimizer, build_scheduler
from ecg_repo.training.trainer import train_model
from ecg_repo.utils.checkpointing import make_run_dir, save_run_metadata
from ecg_repo.utils.io import deep_update, read_yaml
from ecg_repo.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--defaults', default='configs/defaults.yaml')
    parser.add_argument('--beat-table', default=None)
    parser.add_argument('--dataset-id', default=None)
    return parser.parse_args()


def build_model(config: dict) -> torch.nn.Module:
    model_cfg = config['model']
    name = model_cfg['name']

    if name == 'cnn1d':
        return CNN1DClassifier(
            input_size=model_cfg.get('input_size', 1),
            channels=model_cfg.get('channels', [32, 64, 128]),
            kernel_sizes=model_cfg.get('kernel_sizes', [7, 5, 3]),
            num_classes=model_cfg.get('num_classes', 3),
            dropout=model_cfg.get('dropout', 0.2),
        )
    if name == 'bilstm':
        return BiLSTMClassifier(
            input_size=model_cfg.get('input_size', 1),
            hidden_size=model_cfg.get('hidden_size', 128),
            num_layers=model_cfg.get('num_layers', 2),
            num_classes=model_cfg.get('num_classes', 3),
            dropout=model_cfg.get('dropout', 0.3),
        )
    if name == 'bilstm_attention':
        return BiLSTMAttentionClassifier(
            input_size=model_cfg.get('input_size', 1),
            hidden_size=model_cfg.get('hidden_size', 128),
            num_layers=model_cfg.get('num_layers', 2),
            num_classes=model_cfg.get('num_classes', 3),
            dropout=model_cfg.get('dropout', 0.3),
            attention_hidden_size=model_cfg.get('attention_hidden_size', 128),
        )
    if name == 'cnn_lstm':
        return CNNLSTMClassifier(
            input_size=model_cfg.get('input_size', 1),
            conv_channels=model_cfg.get('conv_channels', [32, 64]),
            kernel_sizes=model_cfg.get('kernel_sizes', [7, 5]),
            pool_kernel=model_cfg.get('pool_kernel', 2),
            hidden_size=model_cfg.get('hidden_size', 128),
            num_layers=model_cfg.get('num_layers', 2),
            num_classes=model_cfg.get('num_classes', 3),
            dropout=model_cfg.get('dropout', 0.3),
        )
    raise ValueError(f'Unsupported model: {name}')


def main() -> None:
    args = parse_args()
    defaults = read_yaml(args.defaults)
    experiment = read_yaml(args.config)
    config = deep_update(defaults, experiment)

    if args.beat_table is not None:
        config['data']['beat_table_path'] = args.beat_table
    if args.dataset_id is not None:
        config['data']['hf_dataset_id'] = args.dataset_id

    seed_everything(int(config['project']['seed']))

    dataset = load_ecg_dataset(config['data']['hf_dataset_id'])
    records = get_records_list(dataset, split='train')
    keys = infer_record_keys(records[0])

    beat_table = pd.read_csv(config['data']['beat_table_path'])
    label_column = config['data']['label_column']
    beat_table = beat_table.loc[beat_table[label_column].isin(config['data']['allowed_labels'])].copy()

    train_df = beat_table.loc[beat_table[config['data']['split_column']] == 'train'].reset_index(drop=True)
    val_df = beat_table.loc[beat_table[config['data']['split_column']] == 'val'].reset_index(drop=True)

    train_dataset = ECGBeatDataset(
        records=records,
        beat_table=train_df,
        signal_key=keys['signal'],
        normalize=config['data']['normalization'] == 'zscore_per_beat',
        augment=bool(config['data'].get('train_augment', False)),
    )
    val_dataset = ECGBeatDataset(
        records=records,
        beat_table=val_df,
        signal_key=keys['signal'],
        normalize=config['data']['normalization'] == 'zscore_per_beat',
        augment=False,
    )

    collate_fn = partial(
        pad_collate,
        max_length=int(config['data']['max_length']),
        pad_value=float(config['data']['pad_value']),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=True,
        num_workers=int(config['data']['num_workers']),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=False,
        num_workers=int(config['data']['num_workers']),
        collate_fn=collate_fn,
    )

    model = build_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    class_weights = compute_class_weights(
        labels=train_df[label_column].to_numpy(),
        classes=config['data']['allowed_labels'],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = build_optimizer(model, config['training'])
    scheduler = build_scheduler(optimizer, config['training'])

    run_dir = make_run_dir(config['paths']['runs_root'], config['model']['name'])
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        num_epochs=int(config['training']['epochs']),
        monitor=config['training']['monitor'],
        early_stopping_patience=int(config['training']['early_stopping_patience']),
        positive_labels=config['evaluation']['positive_labels'],
        grad_clip_norm=float(config['training']['grad_clip_norm']) if config['training'].get('grad_clip_norm') is not None else None,
    )
    save_run_metadata(run_dir, config, history)
    print(f'Training complete. Run saved to: {run_dir}')


if __name__ == '__main__':
    main()
