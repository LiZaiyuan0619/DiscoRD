# Data

No data is stored in this repository. The scripts below download the public per-question
results of each collection, pinned to a fixed revision, and write them in one common format
under `data/processed/` (raw downloads go to `data/raw/`; both are git-ignored).

**Common format** (`discordrd.data`): one `.npz` file per benchmark with

| key | shape | content |
|---|---|---|
| `responses` | (models, questions) | per-question score in [0, 1] (binary except AlpacaEval) |
| `model_names` | (models,) | model identifiers |
| `item_ids` | (questions,) | question identifiers |
| `item_weights` | (questions,) | weight of each question in the full score (sums to one) |

The full-benchmark score of a model is `responses @ item_weights`. Only scores are used:
the methods read no question text and no personal information.

| Collection | Models | Benchmarks | Questions per benchmark |
|---|---|---|---|
| OLLB-150 / OLLB-708 | 1,090 / 1,798 (the roles change with the split seed) | 6 | 756-12,032 |
| OLLB-temporal | 3,815 | 6 | 817-14,042 |
| WILD | 65 | 24 (with at least 500 questions) | 541-13,869 |
| DataDecide | 6,612 checkpoints | 9 | 500-10,042 |
| AlpacaEval | 218 | 1 | 790 |

---

## AlpacaEval — `prepare_alpacaeval.py`

* Source: the AlpacaEval repository on GitHub (`tatsu-lab/alpaca_eval`, commit
  `cd543a149df89434d8a54582c0151c0b945c3d20`), licence CC BY-NC 4.0. For every result
  directory we read the per-instruction annotations of the `weighted_alpaca_eval_gpt4_turbo`
  judge against the reference model `gpt4_1106_preview` (about 1 GB in total; each file is
  checked against its Git blob hash).
* Score: `preference - 1`, a continuous value in [0, 1].
* Selection (no score is looked at): same judge and reference; drop directories marked as
  gamed; keep models covering at least 95% of the instructions; one directory per generator;
  drop models with identical outputs; keep the instructions answered by every kept model.
  Result: 218 models x 790 instructions.

```bash
python data/prepare_alpacaeval.py --out data/processed/alpacaeval
```

## DataDecide — `prepare_datadecide.py`

* Source: the binary per-question matrix of DataDecide released on the Hugging Face Hub with
  the item response scaling-law study (dataset `yuhengtu/irsl_datadecide_data`, file
  `3_binary_matrix.parquet`, revision `e9a43934067fc3d203e9aea90b1ba5931a412d9b`, checked by
  SHA-256), licence ODC-BY.
* Rows are checkpoints (data mix, model size, step); the script writes the nine benchmarks of
  the paper and `checkpoints.csv`.
* The runs are split once by pretraining data mix (60% / 20% / 20%,
  `protocol.data_mix_split`); the memory is the final checkpoint of each memory run, four
  checkpoints sampled along each run, or every checkpoint (`pipelines/datadecide.py`).

```bash
python data/prepare_datadecide.py --out data/processed/datadecide
```

## WILD — `prepare_wild.py`

* Source: `kensho/WILD` on the Hugging Face Hub (file `data.parquet`, revision
  `fe90808058cbfe6748ce6d17f1af8e3bc087a681`), licence Apache-2.0.
* A question is `subtask::item_id`. A (benchmark, question) with conflicting duplicate scores
  is dropped for every model; each benchmark keeps the questions answered by every model.
* The pipeline uses the 24 benchmarks with at least 500 questions and holds out one
  organisation at a time (`protocol.organisation_folds`).

```bash
python data/prepare_wild.py --out data/processed/wild
```

## Open LLM Leaderboard, second version — `prepare_ollb.py`

* Source: the `open-llm-leaderboard/contents` table and the per-model `-details` datasets on
  the Hugging Face Hub. The details datasets are gated: export `HF_TOKEN` and pass
  `--accept-access` to request access. Only per-question results are read.
* Per-question scores: IFEval strict prompt-level and instruction-level accuracy; BBH, GPQA
  and MuSR `acc_norm`; MATH Lvl 5 `exact_match`; MMLU-Pro `acc`. BBH, IFEval and MuSR
  macro-average their subtasks, which the item weights reproduce; every model's matrix score
  must match its official leaderboard score within 0.01 (MATH Lvl 5 is exempt).
* OLLB-150: 1,200 models uploaded before 2025, sampled by upload quarter; the 1,090 with
  complete records are split by whole organisations into memory (640), transfer (150, fits
  Ridge transfer), validation (150) and candidates (150), once per split seed.
* OLLB-708: the same memory, transfer and validation models; the candidates are the 708
  models uploaded from January to March 2025 with complete, verified records.

```bash
export HF_TOKEN=...
python data/prepare_ollb.py manifest
python data/prepare_ollb.py download --accept-access
python data/prepare_ollb.py build            # writes data/processed/ollb/{ollb_150,ollb_708}/
```

Each setting directory holds the six matrices and `splits.csv`, which gives the role of every
model for each split seed (`memory`, `transfer`, `validation` or `candidate`).

## Open LLM Leaderboard, first version — OLLB-temporal

OLLB-temporal uses the same directory layout (`data/processed/ollb/ollb_temporal/`), built as
follows; we do not include a script for this step.

* Per-question results: the preprocessed matrices of the first version of the leaderboard
  released with SparseEval (Hugging Face dataset `iridescentttt/SparseEval_preprocess_data`:
  ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande), with equal question weights.
* Dates: the time stamp of each model's archived result file in
  `open-llm-leaderboard-old/results`; models with more than one result file are left out.
* Split at 1 April 2024. Earlier models are split by whole organisations into transfer (300),
  validation (300) and memory (2,585), once per split seed. The candidates are the 630 later
  models from organisations with no earlier model, after removing any later model whose
  answers agree with an earlier model on at least 99% of the questions.
