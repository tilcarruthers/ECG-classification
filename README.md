# ECG Beat Classification

A clean, reproducible PyTorch repository for **ECG beat classification** built from an MSc notebook refactor.

This repo is intentionally scoped as an **applied ML / research engineering project**, not a clinical system. The goal is to turn an exploratory notebook into a defensible experiment stack with:

- explicit data and segmentation assumptions
- grouped train/val/test splits to reduce leakage risk
- reproducible configs and run directories
- a small, coherent model stack
- honest evaluation and limitations

## Current scope

The repo is set up for **3-class beat classification** after excluding ambiguous / sparse `UNK` beats:

- `0` → `NOR`
- `1` → `SVEB`
- `2` → `VEB`

The original notebook also discussed anomaly detection. In this refactor, the primary task is framed more precisely as:

> supervised beat-level ECG classification, with optional secondary binary analysis of **normal vs abnormal** beats.

## What changed from the notebook

The notebook mixed together:

- data loading
- beat extraction
- filtering
- train/validation splitting
- model definitions
- training loops
- evaluation
- discussion text

This repo separates those concerns and fixes the most important methodological issues before retraining:

1. **Grouped splits first**
   - avoid random beat-level train/validation splits that can leak record-level information

2. **Segmentation treated as an explicit design choice**
   - the original beat extraction logic should be audited, not inherited blindly
   - segmentation outputs should be sanity-checked before large retrains

3. **Variable-length handling done properly**
   - recurrent models use sequence lengths and packed sequences
   - attention masks padded positions

4. **Evaluation made more honest**
   - model selection should rely on validation macro F1 / balanced metrics, not accuracy alone
   - binary `normal vs abnormal` analysis is computed explicitly from collapsed predictions instead of by subsetting multiclass labels and reusing weighted scores

## Recommended experiment order

Do **not** start by adding architectures.

Use this order:

1. `bootstrap_data.py`
2. `audit_dataset.py`
3. `make_splits.py`
4. `build_beat_table.py`
5. segmentation sanity notebook
6. baseline training
7. evaluation cleanup
8. only then stronger extensions

## Minimal intended experiment stack

Keep the story tight:

1. `cnn1d_baseline`
2. `bilstm_baseline`
3. `bilstm_attention`
4. `cnn_lstm_final` only if it earns its place under the grouped split

## Repository layout

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

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data bootstrap and audit

```bash
python scripts/bootstrap_data.py
python scripts/audit_dataset.py --outdir reports/tables
```

## Split generation

Default split policy is grouped by patient when available, otherwise record.

```bash
python scripts/make_splits.py   --group-key patient_id   --val-size 0.15   --test-size 0.15   --seed 42   --outpath outputs/splits/record_splits.csv
```

## Beat table construction

This creates the main metadata table used for training and analysis.

```bash
python scripts/build_beat_table.py   --splits-path outputs/splits/record_splits.csv   --segmentation-method aligned_interval   --min-length 32   --max-length 256   --drop-label 3   --outpath outputs/processed/beat_table.csv
```

## Training

```bash
python scripts/train.py   --config configs/experiments/cnn1d_baseline.yaml
```

Or override paths explicitly:

```bash
python scripts/train.py   --config configs/experiments/bilstm_attention.yaml   --beat-table outputs/processed/beat_table.csv
```

## Evaluation

```bash
python scripts/evaluate.py   --run-dir outputs/runs/<timestamp>_bilstm_attention
```

## Notebooks

The notebooks are intentionally thin:

- `01_dataset_audit.ipynb`
- `02_segmentation_sanity_checks.ipynb`
- `03_results_analysis.ipynb`

They should **consume saved outputs** from `outputs/` and `reports/`, not recreate training or core preprocessing logic.

## What not to claim

Do not claim any of the following unless you genuinely validate them later:

- patient-level generalisation, unless your grouped split proves it
- screening utility or clinical readiness
- anomaly detection in the unsupervised / novelty-detection sense
- state-of-the-art performance
- architecture superiority unless every compared model is retrained cleanly under the same corrected split

## Known open questions to settle before final README/results

- whether the dataset’s beat markers should be treated as beat onsets, beat centres, or interval boundaries
- whether `aligned_interval` or a centered-window strategy gives the most faithful beat representation
- whether augmentation helps once the split policy is corrected
- whether the CNN-LSTM actually outperforms the smaller baseline stack under the new setup

## Suggested final README structure after retraining

1. Project overview
2. Dataset and label policy
3. Segmentation and preprocessing
4. Split strategy and leakage mitigation
5. Model stack
6. Evaluation protocol
7. Results
8. Limitations
9. Reproducibility
