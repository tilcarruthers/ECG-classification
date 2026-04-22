from __future__ import annotations

import argparse
from pathlib import Path

from ecg_repo.data.loading import load_class_mapping, load_ecg_dataset
from ecg_repo.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', default='dpelacani/ecg-led2-cleaned')
    parser.add_argument('--outdir', default='outputs/bootstrap')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_ecg_dataset(args.dataset_id)
    mapping = load_class_mapping(args.dataset_id)

    summary = {
        'dataset_id': args.dataset_id,
        'splits': {split_name: len(dataset[split_name]) for split_name in dataset.keys()},
        'class_mapping': mapping,
    }
    write_json(summary, outdir / 'bootstrap_summary.json')
    print(f"Saved dataset bootstrap summary to {outdir / 'bootstrap_summary.json'}")


if __name__ == '__main__':
    main()
