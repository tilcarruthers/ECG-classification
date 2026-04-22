from __future__ import annotations

from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ecg_repo.utils.io import read_json


DEFAULT_DATASET_ID = "dpelacani/ecg-led2-cleaned"


def load_ecg_dataset(dataset_id: str = DEFAULT_DATASET_ID):
    return load_dataset(dataset_id)


def load_class_mapping(dataset_id: str = DEFAULT_DATASET_ID) -> dict[str, str]:
    mapping_path = hf_hub_download(
        repo_id=dataset_id,
        filename="class_mapping.json",
        repo_type="dataset",
    )
    return read_json(mapping_path)


def get_records_list(dataset, split: str = "train") -> list[dict[str, Any]]:
    return [record for record in dataset[split]]


def infer_record_keys(record: dict[str, Any]) -> dict[str, str]:
    key_candidates = {
        "patient_id": ["patient_id", "subject_id", "patient", "subject"],
        "source_db": ["source_db", "dataset_name", "source"],
        "sampling_rate": ["sampling_rate", "fs", "sample_rate"],
        "signal": ["signal", "ecg", "ecg_signal", "waveform"],
        "beat_locs": ["beat_locs", "beat_locations", "r_peaks", "beat_indices"],
        "beat_labels": ["beat_labels", "labels", "beat_types"],
    }

    resolved: dict[str, str] = {}
    for canonical, options in key_candidates.items():
        found = next((candidate for candidate in options if candidate in record), None)
        if found is None:
            raise KeyError(
                f"Could not infer required key for '{canonical}' from record keys: {list(record.keys())}"
            )
        resolved[canonical] = found
    return resolved
