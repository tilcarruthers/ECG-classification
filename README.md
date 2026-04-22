# ECG Beat Classification

A clean PyTorch repository for **ECG beat classification** built by refactoring an MSc notebook into a reproducible machine learning project.

The focus of this repository is not clinical deployment or leaderboard chasing. It is a compact, defensible experiment stack for **beat-level ECG classification** with careful attention to:

- leakage-aware data splitting
- explicit segmentation assumptions
- reproducible training runs
- class imbalance handling
- honest evaluation and limitations

## Overview

This project studies **supervised ECG beat classification** using beat segments extracted from longer ECG recordings. The current repository is scoped around a **3-class classification task** after excluding ambiguous or weakly defined `UNK` beats:

- `0` → `NOR` (normal beat)
- `1` → `SVEB` (supraventricular ectopic beat)
- `2` → `VEB` (ventricular ectopic beat)

The original notebook also touched on “anomaly detection”, but the more accurate framing for the work in this repository is:

> **supervised beat-level ECG classification**, with optional secondary analysis of **normal vs abnormal** behaviour.

## Why this refactor exists

The source notebook contained the entire pipeline in one place: data loading, beat extraction, filtering, splitting, model definitions, training, evaluation, and discussion. That is fine for exploration, but weak for reproducibility and difficult to trust methodologically.

This repository restructures the project into a small, coherent ML codebase with a clear boundary between:

- reusable source code in `src/`
- experiment configs in `configs/`
- runnable entry points in `scripts/`
- analysis-only notebooks in `notebooks/`
- saved outputs in `outputs/` and `reports/`

## Methodological focus

A major goal of this refactor is to avoid building a polished repository around a flawed experimental setup.

### 1. Grouped splits instead of naive beat-level splits

A random beat-level split can leak record-specific structure across training and validation. The default workflow in this repo therefore generates **grouped train/validation/test splits**, using `patient_id` when available and falling back to `record_id` otherwise.

### 2. Segmentation is treated as a first-class design choice

Beat segmentation is central to the task and easy to get subtly wrong. Rather than freezing the original notebook behaviour without scrutiny, this repo makes segmentation explicit and testable. Multiple strategies are supported so they can be audited visually before large retrains.

### 3. Variable-length sequences are handled properly

The original notebook padded sequences aggressively and used recurrent models in a way that could let padding influence model behaviour. In this refactor, recurrent models consume true sequence lengths, use packed sequences where appropriate, and attention layers can mask padded positions.

### 4. Evaluation is intentionally conservative

The emphasis is on **macro-aware metrics**, per-class reporting, and transparent limitations rather than inflated claims. The repo is designed for a small and credible model comparison, not for overstating performance.

## Planned experiment stack

The architecture scope is intentionally narrow.

### Baselines
- **1D CNN**
- **BiLSTM**
- **BiLSTM + attention**

### Optional stronger final model
- **CNN-LSTM**, but only if it earns its place under the corrected split policy

## Repository structure

```text
ecg-beat-classification/
├── README.md
├── pyproject.toml
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
├── scripts/
├── src/
├── notebooks/
├── tests/
├── outputs/
└── reports/
```

## Getting started

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1) Download and audit the dataset

```bash
python scripts/bootstrap_data.py
python scripts/audit_dataset.py --outdir reports/tables
```

### 2) Create grouped splits

```bash
python scripts/make_splits.py \
  --group-key patient_id \
  --val-size 0.15 \
  --test-size 0.15 \
  --seed 42 \
  --outpath outputs/splits/record_splits.csv
```

### 3) Build the beat table

```bash
python scripts/build_beat_table.py \
  --splits-path outputs/splits/record_splits.csv \
  --segmentation-method aligned_interval \
  --min-length 32 \
  --max-length 256 \
  --drop-label 3 \
  --outpath outputs/processed/beat_table.csv
```

### 4) Train a baseline

```bash
python scripts/train.py \
  --config configs/experiments/cnn1d_baseline.yaml
```

### 5) Evaluate a run

```bash
python scripts/evaluate.py \
  --run-dir outputs/runs/<timestamp>_cnn1d_baseline
```

## Notebooks

The notebooks are intentionally thin and analysis-focused:

- `01_dataset_audit.ipynb`
- `02_segmentation_sanity_checks.ipynb`
- `03_results_analysis.ipynb`

They are meant to **read saved outputs** from the repo pipeline rather than duplicate model code, training loops, or preprocessing logic.

## Current repository status

This repository currently provides the refactored project scaffold, data pipeline, model definitions, training/evaluation utilities, and tests.

The next stage is to retrain the baseline stack cleanly under the grouped split policy and then populate:

- final metrics tables
- confusion matrices
- qualitative plots
- a completed Results section

Until that is done, the repo should be read as a **reproducible experimental framework** rather than a finished benchmark report.

## Evaluation philosophy

The intended evaluation emphasis is:

- **macro F1**
- **balanced accuracy**
- per-class precision / recall / F1
- confusion matrices
- optional collapsed **normal vs abnormal** analysis

Accuracy alone is not a reliable summary here because of class imbalance.

## Limitations

This project should be interpreted conservatively.

- It is **not** a clinical decision-support system.
- It does **not** currently claim patient-level generalisation beyond what grouped splits support.
- It does **not** claim state-of-the-art performance.
- It does **not** treat exploratory notebook results as final evidence.
- The final conclusions will depend heavily on the validated segmentation choice and leakage-safe split strategy.

## Roadmap

- validate segmentation strategy with visual sanity checks
- retrain the baseline stack under grouped splits
- add final metrics, plots, and error analysis
- stabilise the experiment configs
- add CI and pre-commit checks once the training/evaluation flow is fully settled

## Project goals

This repository is meant to signal strong applied ML and research-engineering practice:

- clean PyTorch code
- reproducible experiments
- explicit assumptions
- defensible evaluation
- limited but coherent model scope
- honest communication of results and limitations

