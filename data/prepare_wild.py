"""Build the WILD score matrices (one file per benchmark).

Source: the per-question results of WILD (Apache-2.0) on the Hugging Face Hub,
dataset ``kensho/WILD``, file ``data.parquet`` (136 MB), pinned to one
revision.  Each row is the score of one model on one question of one
benchmark (``task``); a question is identified by ``subtask::item_id``.

Processing, decided without looking at any score:
* a (benchmark, question) that has conflicting duplicate scores for any model
  is dropped for every model;
* for each benchmark, keep the questions answered by every model;
* models are sorted by name and every question has equal weight.

The paper uses the 24 benchmarks with at least 500 questions; the pipeline
applies that filter (``discordrd.pipelines.wild.MIN_QUESTIONS``).

Usage
-----
    python data/prepare_wild.py --out data/processed/wild
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from discordrd.data import ResponseMatrix, save_matrix  # noqa: E402

REPOSITORY = "kensho/WILD"
REVISION = "fe90808058cbfe6748ce6d17f1af8e3bc087a681"
FILE = "data.parquet"
SIZE = 136130280


def download(target: Path) -> Path:
    if target.exists() and target.stat().st_size == SIZE:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{FILE}"
    partial = target.with_suffix(".part")
    with urlopen(Request(url, headers={"User-Agent": "discordrd-data"}), timeout=120) as r, \
            partial.open("wb") as f:
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
    if partial.stat().st_size != SIZE:
        raise ValueError("unexpected file size; the download is kept as .part")
    partial.replace(target)
    return target


def long_table(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100000,
                                                   columns=["model", "task", "subtask", "item_id", "score"]):
        b = batch.to_pandas()
        chunks.append(pd.DataFrame({"model": b.model.astype(str), "task": b.task.astype(str),
                                    "item": b.subtask.astype(str) + "::" + b.item_id.astype(str),
                                    "score": b.score.astype(float)}))
    frame = pd.concat(chunks, ignore_index=True)
    for c in ("model", "task", "item"):
        frame[c] = frame[c].astype("category")
    return frame


def drop_ambiguous(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop every (benchmark, question) with conflicting duplicate scores, for all models."""
    key = ["model", "task", "item"]
    duplicates = frame[frame.duplicated(key, keep=False)]
    conflict = duplicates.groupby(key, observed=True).score.nunique()
    excluded = conflict[conflict > 1].reset_index()[["task", "item"]].drop_duplicates()
    marked = frame.merge(excluded.assign(ambiguous=True), on=["task", "item"], how="left",
                         validate="many_to_one")
    clean = marked[marked.ambiguous.isna()].drop(columns="ambiguous")
    if not clean.score.between(0, 1).all():
        raise ValueError("scores outside [0, 1]")
    return clean.drop_duplicates(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/processed/wild"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/wild") / FILE)
    args = parser.parse_args()
    frame = drop_ambiguous(long_table(download(args.raw)))
    args.out.mkdir(parents=True, exist_ok=True)
    for task, g in frame.groupby("task", observed=True, sort=True):
        common = g.pivot(index="model", columns="item", values="score").dropna(axis=1)
        if not common.shape[1]:
            continue
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task))
        save_matrix(args.out / f"{slug}.npz",
                    ResponseMatrix(str(task), common.to_numpy(float),
                                   common.index.to_numpy(dtype=str),
                                   common.columns.to_numpy(dtype=str),
                                   np.full(common.shape[1], 1 / common.shape[1])))
        print(f"{task}: {common.shape[0]} models x {common.shape[1]} questions")


if __name__ == "__main__":
    main()
