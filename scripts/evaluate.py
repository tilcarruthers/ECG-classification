from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecg_repo.data.collate import pad_collate
from ecg_repo.data.dataset import ECGBeatDataset
from ecg_repo.data.loading import get_records_list, infer_record_keys, load_ecg_dataset
from ecg_repo.evaluation.plots import save_confusion_matrix_figure
from ecg_repo.evaluation.reports import save_metrics_report, save_predictions
from ecg_repo.models.bilstm import BiLSTMClassifier
from ecg_repo.models.bilstm_attention import BiLSTMAttentionClassifier
from ecg_repo.models.cnn1d import CNN1DClassifier
from ecg_repo.models.cnn_lstm import CNNLSTMClassifier
from ecg_repo.training.trainer import run_epoch
from ecg_repo.utils.io import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
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
    run_dir = Path(args.run_dir)
    config = read_yaml(run_dir / 'config_resolved.yaml')

    checkpoint = torch.load(run_dir / 'artifacts' / 'best_model.pt', map_location='cpu')
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    dataset = load_ecg_dataset(config['data']['hf_dataset_id'])
    records = get_records_list(dataset, split='train')
    keys = infer_record_keys(records[0])

    beat_table = pd.read_csv(config['data']['beat_table_path'])
    test_df = beat_table.loc[beat_table[config['data']['split_column']] == 'test'].reset_index(drop=True)

    test_dataset = ECGBeatDataset(
        records=records,
        beat_table=test_df,
        signal_key=keys['signal'],
        normalize=config['data']['normalization'] == 'zscore_per_beat',
        augment=False,
    )
    collate_fn = partial(
        pad_collate,
        max_length=int(config['data']['max_length']),
        pad_value=float(config['data']['pad_value']),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=False,
        num_workers=int(config['data']['num_workers']),
        collate_fn=collate_fn,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    result = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        positive_labels=config['evaluation']['positive_labels'],
    )

    metrics = result.metrics
    save_metrics_report(metrics, run_dir / 'artifacts' / 'test_metrics.json')
    save_predictions(
        beat_ids=result.beat_ids,
        record_ids=result.record_ids,
        y_true=result.y_true,
        y_pred=result.y_pred,
        split='test',
        path=run_dir / 'artifacts' / 'test_predictions.csv',
    )

    confusion = torch.tensor(metrics['multiclass']['confusion_matrix']).cpu().numpy()
    save_confusion_matrix_figure(
        confusion=confusion,
        labels=['NOR', 'SVEB', 'VEB'],
        path=run_dir / 'artifacts' / 'test_confusion_matrix.png',
    )
    print(f"Saved evaluation outputs to {run_dir / 'artifacts'}")


if __name__ == '__main__':
    main()
