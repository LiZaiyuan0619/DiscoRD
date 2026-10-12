# Disagree to Deploy: Discordance Retrieval for Sparse Evaluation of LLMs

This repository contains the code of **DiscoRD** (Discordance Retrieval with fusion), the
evaluation protocol of the paper, the published estimators it is compared with, and the
scripts that prepare the four benchmark collections and run every setting of the main
results.

**Sparse evaluation.** A new model, the *candidate*, answers only *A* anchor questions of a
benchmark, and its full-benchmark score is estimated from these answers. Public
leaderboards also keep the per-question answers of many past models, which we read as an
*evaluation memory*.

* **Consensus retrieval** finds the past models nearest to the candidate and averages their
  full scores. Its estimate can never exceed the best score in the memory.
* **Discordance retrieval** keeps the candidate's own anchor mean and adds only the
  neighbours' *local corrections* (full score minus anchor mean).
* **DiscoRD** fuses consensus and discordance predictors with weights fitted on a
  validation set and smoothed towards equal weights.

Everything runs on CPU; a GPU only speeds up the SparseEval-MLP fits.

---

## Repository layout

```
DiscoRD/
├── src/discordrd/            the method and the evaluation protocol (importable package)
│   ├── distances.py          score distance and response distance between models
│   ├── retrieval.py          consensus / discordance retrieval, choice of k on the validation set
│   ├── predictors.py         anchor mean, Ridge transfer
│   ├── sparseeval_mlp.py     SparseEval-MLP (re-implementation, see THIRD_PARTY.md)
│   ├── fusion.py             simplex weights, smoothing towards equal weights
│   ├── discord.py            DiscoRD: build the predictors of one benchmark and fuse them; ablations
│   ├── ladder.py             corrections built up from no memory (global, stratified)
│   ├── protocol.py           splits, anchor draws, 3 seeds x 10 anchor draws
│   ├── metrics.py            the 31 metrics (ties broken in expectation)
│   ├── stats.py              paired Wilcoxon tests with Holm correction
│   ├── data.py               the common score-matrix format
│   ├── pipelines/            one cell of each collection: ollb, wild, datadecide, alpacaeval
│   └── baselines/            the published estimators behind one interface
├── data/                     download and preprocessing scripts (no data is stored here)
├── scripts/                  run scripts per collection, summary tables, full pipeline
├── configs/                  one YAML file per setting of the paper
└── tests/                    unit tests and synthetic smoke tests
```

## Installation

```bash
conda create -n discordrd python=3.10 -y && conda activate discordrd
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0   # the CPU build is enough
pip install -r requirements.txt          # pinned versions used for the paper
pip install -e .                         # makes `import discordrd` available
pytest -q                                # unit tests and synthetic smoke tests, about 30 seconds
```

Versions used in the paper: Python 3.10, NumPy 2.2, SciPy 1.15, scikit-learn 1.7,
PyTorch 2 (CPU build), pandas 2.3.

## Using DiscoRD on your own data

The method needs, for one target benchmark: the full scores and anchor means of the past
models, the anchor means of the validation models and candidates (and the validation
models' full scores), two distance matrices, and the estimates of the non-retrieval
consensus predictors. `pipelines/common.py::single_benchmark_discord` assembles all of
this from three score matrices:

```python
import numpy as np
from discordrd.pipelines.common import single_benchmark_discord

# memory, valid, query: (n_models, n_questions) per-question scores in [0, 1]
anchors = np.random.default_rng(0).permutation(memory.shape[1])[:20]
output, data = single_benchmark_discord(memory, valid, query, anchors, fit_seed=0)
output.query      # DiscoRD estimates of the candidates' full scores
output.weights    # fusion weights of the six predictors
output.k          # number of neighbours chosen by each retrieval predictor
```

`discordrd.DiscoRD` works on any precomputed inputs (`discordrd.TargetData`), for example to
let another estimator supply the consensus estimate:

```python
from discordrd import DiscoRD, DISCORDANCE_PREDICTORS
model = DiscoRD(predictors=("My estimator", *DISCORDANCE_PREDICTORS))   # K = 3
```

## Data

| Collection | Settings | Source | Licence | Script |
|---|---|---|---|---|
| Open LLM Leaderboard v2 | OLLB-150, OLLB-708 | `open-llm-leaderboard/contents` and per-model `-details` datasets (Hugging Face Hub) | per-model terms on the Hub | `data/prepare_ollb.py` |
| Open LLM Leaderboard v1 | OLLB-temporal | SparseEval's preprocessed v1 matrices and the archived v1 results (Hugging Face Hub) | per-model terms on the Hub | described in `data/README.md` |
| WILD | WILD | `kensho/WILD` (Hugging Face Hub) | Apache-2.0 | `data/prepare_wild.py` |
| DataDecide | DataDecide | DataDecide binary matrix released with the item response scaling-law study (Hugging Face Hub) | ODC-BY | `data/prepare_datadecide.py` |
| AlpacaEval | AlpacaEval | annotations in `tatsu-lab/alpaca_eval` (GitHub, pinned commit) | CC BY-NC 4.0 | `data/prepare_alpacaeval.py` |

The WILD, DataDecide and AlpacaEval sources are pinned to a revision or commit; the Open LLM
Leaderboard is archived and read through its latest results. Only per-question scores are
read; no question text and no personal information. See `data/README.md` for details.

## Reproducing the experiments

```bash
python data/prepare_alpacaeval.py
python scripts/run_alpacaeval.py --config configs/alpacaeval.yaml --workers 16 --baselines --extras
python scripts/summarize.py outputs/alpacaeval --tests
```

The other collections follow the same pattern (`scripts/reproduce_all.sh` lists every step):

| Paper | Command |
|---|---|
| Table 1, OLLB-150 / OLLB-708 / OLLB-temporal | `python scripts/run_ollb.py --config configs/ollb_150.yaml --baselines` (and `ollb_708`, `ollb_temporal`) |
| Table 1, WILD | `python scripts/run_wild.py --config configs/wild.yaml --baselines` |
| Table 1, DataDecide | `python scripts/run_datadecide.py --config configs/datadecide.yaml --baselines` |
| Table 1, AlpacaEval | `python scripts/run_alpacaeval.py --config configs/alpacaeval.yaml --baselines` |
| Table 2 (ablations, corrections from no memory) | add `--extras` to any of the above |
| significance tests against DiscoRD | `python scripts/summarize.py <output dirs> --tests` |
| all 31 metrics | `python scripts/summarize.py <output dirs> --all` |

Each run script writes `metrics.csv` (the 31 metrics of every cell and method),
`fusion_weights.csv` (weights and chosen k), and with `--save-predictions` every estimate.
A *cell* is one benchmark in one run; a setting averages its cells. Every setting has 30
runs (3 seeds x 10 draws of anchor questions).

**Compute.** The cost is dominated by the SparseEval-MLP fits (one per benchmark and run);
the predictors and the fusion of DiscoRD add about 0.15 s per benchmark and run. AlpacaEval
(60 cells) is the lightest setting. The Open LLM Leaderboard, WILD and DataDecide need
thousands of fits and are best spread over many CPU processes (`--workers`).

For reference, the MAE (%) of DiscoRD reported in Table 1 of the paper:

| OLLB-150 | | OLLB-708 | | OLLB-temporal | | WILD | | | DataDecide | | AlpacaEval | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A=20 | A=100 | A=20 | A=100 | A=20 | A=100 | A=10 | A=20 | A=50 | A=20 | A=100 | A=20 | A=100 |
| 1.93 | 1.30 | 2.50 | 1.73 | 2.14 | 1.41 | 5.20 | 4.38 | 3.37 | 2.87 | 1.60 | 3.69 | 1.77 |

## Where each part of the paper lives

| Paper | Code |
|---|---|
| Section 3, consensus and discordance retrieval | `retrieval.py` |
| Section 3, fusion and smoothing | `fusion.py`, `discord.py` |
| Appendix A, predictors (SparseEval-MLP, Ridge transfer, Score-kNN, Response-kNN, Discordance Score-kNN, Discordance Response-kNN) | `sparseeval_mlp.py`, `predictors.py`, `discord.py` |
| Appendix A, distances and neighbour weights | `distances.py`, `retrieval.py::knn_predict` |
| Appendix A, per-ecosystem details | `pipelines/ollb.py`, `pipelines/wild.py`, `pipelines/datadecide.py`, `pipelines/alpacaeval.py` |
| Appendix A, another method as the consensus estimate | `discord.py` (`DiscoRD(predictors=...)`) |
| Appendix C, data and splits | `data/`, `protocol.py` |
| Appendix C, baselines | `baselines/` |
| Appendix C, metrics | `metrics.py` |
| Appendix C, runs and tests | `protocol.py`, `stats.py`, `scripts/summarize.py` |
| Table 2, ablations | `discord.py::ABLATIONS`, `ladder.py` |

## Licence

The code is released under the MIT licence (`LICENSE`). Third-party code keeps its
original licence and attribution (`THIRD_PARTY.md`). The data are not redistributed; each
collection keeps the licence of its source.

## Citation

The citation will be added after the anonymous review period.
