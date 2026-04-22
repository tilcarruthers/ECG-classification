PYTHON ?= python

.PHONY: install lint format test bootstrap audit splits beat-table train evaluate

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

bootstrap:
	$(PYTHON) scripts/bootstrap_data.py

audit:
	$(PYTHON) scripts/audit_dataset.py --outdir reports/tables

splits:
	$(PYTHON) scripts/make_splits.py --outpath outputs/splits/record_splits.csv

beat-table:
	$(PYTHON) scripts/build_beat_table.py --splits-path outputs/splits/record_splits.csv --outpath outputs/processed/beat_table.csv

train:
	$(PYTHON) scripts/train.py --config configs/experiments/bilstm_baseline.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --run-dir $(RUN_DIR)
