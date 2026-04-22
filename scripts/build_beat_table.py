from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ecg_repo.data.filtering import filter_beats_by_length
from ecg_repo.data.loading import get_records_list, infer_record_keys, load_class_mapping, load_ecg_dataset
from ecg_repo.data.segmentation import get_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', default='dpelacani/ecg-led2-cleaned')
    parser.add_argument('--split', default='train')
    parser.add_argument('--splits-path', default='outputs/splits/record_splits.csv')
    parser.add_argument('--segmentation-method', default='aligned_interval')
    parser.add_argument('--min-length', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--drop-label', type=int, action='append', default=[3])
    parser.add_argument('--outpath', default='outputs/processed/beat_table.csv')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_ecg_dataset(args.dataset_id)
    mapping = load_class_mapping(args.dataset_id)
    records = get_records_list(dataset, split=args.split)
    keys = infer_record_keys(records[0])

    splits_df = pd.read_csv(args.splits_path)
    split_lookup = splits_df.set_index('record_id')['split'].to_dict()

    rows = []
    for record_id, record in tqdm(list(enumerate(records)), desc='Building beat table'):
        signal = np.asarray(record[keys['signal']], dtype=np.float32)
        beat_locs = np.asarray(record[keys['beat_locs']], dtype=int)
        beat_labels = np.asarray(record[keys['beat_labels']], dtype=int)
        segments = get_segments(signal, beat_locs, method=args.segmentation_method)

        if len(segments) != len(beat_labels):
            min_len = min(len(segments), len(beat_labels))
            segments = segments[:min_len]
            beat_labels = beat_labels[:min_len]

        for segment, raw_label in zip(segments, beat_labels, strict=False):
            rows.append(
                {
                    'beat_id': f'{record_id}_{segment.beat_idx}',
                    'record_id': record_id,
                    'patient_id': record[keys['patient_id']],
                    'source_db': record[keys['source_db']],
                    'split': split_lookup.get(record_id, 'unassigned'),
                    'beat_idx': int(segment.beat_idx),
                    'start_idx': int(segment.start_idx),
                    'end_idx': int(segment.end_idx),
                    'length': int(segment.length),
                    'segmentation_method': segment.method,
                    'label_raw': int(raw_label),
                    'label_name_raw': mapping.get(str(int(raw_label)), f'UNKNOWN_{raw_label}'),
                    'label_mapped': int(raw_label),
                }
            )

    beat_table = pd.DataFrame(rows)
    beat_table = filter_beats_by_length(
        beat_table,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    if args.drop_label:
        beat_table = beat_table.loc[~beat_table['label_mapped'].isin(args.drop_label)].reset_index(drop=True)

    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    beat_table.to_csv(outpath, index=False)

    print(f'Saved beat table to {outpath}')
    print(beat_table['split'].value_counts().to_dict())
    print(beat_table['label_mapped'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
