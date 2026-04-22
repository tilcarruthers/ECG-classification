from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_repo.data.loading import get_records_list, infer_record_keys, load_class_mapping, load_ecg_dataset
from ecg_repo.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', default='dpelacani/ecg-led2-cleaned')
    parser.add_argument('--split', default='train')
    parser.add_argument('--outdir', default='reports/tables')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_ecg_dataset(args.dataset_id)
    mapping = load_class_mapping(args.dataset_id)
    records = get_records_list(dataset, split=args.split)
    keys = infer_record_keys(records[0])

    rows = []
    label_counts: dict[int, int] = {}
    for record_id, record in enumerate(records):
        patient_id = record[keys['patient_id']]
        source_db = record[keys['source_db']]
        signal = np.asarray(record[keys['signal']])
        beat_locs = np.asarray(record[keys['beat_locs']], dtype=int)
        beat_labels = np.asarray(record[keys['beat_labels']], dtype=int)

        unique, counts = np.unique(beat_labels, return_counts=True)
        for label, count in zip(unique, counts, strict=False):
            label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)

        rows.append(
            {
                'record_id': record_id,
                'patient_id': patient_id,
                'source_db': source_db,
                'sampling_rate': record[keys['sampling_rate']],
                'signal_length': int(len(signal)),
                'num_beats': int(len(beat_locs)),
            }
        )

    records_df = pd.DataFrame(rows)
    records_df.to_csv(outdir / 'record_level_audit.csv', index=False)

    summary = {
        'num_records': int(len(records_df)),
        'num_unique_patients': int(records_df['patient_id'].nunique()),
        'patients_with_multiple_records': int((records_df['patient_id'].value_counts() > 1).sum()),
        'source_db_counts': records_df['source_db'].value_counts().to_dict(),
        'label_counts_raw': {str(k): int(v) for k, v in sorted(label_counts.items())},
        'class_mapping': mapping,
        'inferred_keys': keys,
    }
    write_json(summary, outdir / 'dataset_audit_summary.json')
    print(f'Saved audit outputs under {outdir}')


if __name__ == '__main__':
    main()
