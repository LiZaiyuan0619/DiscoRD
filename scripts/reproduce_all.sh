#!/usr/bin/env bash
# Every step behind the main results, collection by collection.
# The data scripts download from GitHub and the Hugging Face Hub; the OLLB details
# datasets need HF_TOKEN (see data/README.md).  Set WORKERS to the number of CPU
# processes to use.  Each block can be run on its own.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-8}

# AlpacaEval (lightest: 60 cells)
python data/prepare_alpacaeval.py
python scripts/run_alpacaeval.py --config configs/alpacaeval.yaml --workers "$WORKERS" --baselines --extras

# DataDecide
python data/prepare_datadecide.py
python scripts/run_datadecide.py --config configs/datadecide.yaml --workers "$WORKERS" --baselines --extras

# WILD
python data/prepare_wild.py
python scripts/run_wild.py --config configs/wild.yaml --workers "$WORKERS" --baselines --extras

# Open LLM Leaderboard, second version (OLLB-150 and OLLB-708)
python data/prepare_ollb.py manifest
python data/prepare_ollb.py download --accept-access
python data/prepare_ollb.py build
python scripts/run_ollb.py --config configs/ollb_150.yaml --workers "$WORKERS" --baselines --extras
python scripts/run_ollb.py --config configs/ollb_708.yaml --workers "$WORKERS" --baselines --extras
# OLLB-temporal: build data/processed/ollb/ollb_temporal/ as described in data/README.md, then
# python scripts/run_ollb.py --config configs/ollb_temporal.yaml --workers "$WORKERS" --baselines --extras

# Tables: averages over cells, and paired tests against DiscoRD
python scripts/summarize.py outputs/ollb_150 outputs/ollb_708 outputs/wild outputs/datadecide outputs/alpacaeval \
    --tests --out outputs/summary
