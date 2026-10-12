"""Build the DataDecide score matrices (nine benchmarks, one row per checkpoint).

Source: the binary per-question matrix of DataDecide (ODC-BY) as released on
the Hugging Face Hub with the item response scaling-law study (Truong et al.,
2026), dataset ``yuhengtu/irsl_datadecide_data``, file
``3_binary_matrix.parquet`` (45 MB), pinned to one revision and checked by
SHA-256.  Each row is one checkpoint of one pretraining run, identified by its
data mix, model size and step; each column is one question of one benchmark.

Output: ``<out>/<benchmark>.npz`` for the nine benchmarks of the paper and
``<out>/checkpoints.csv`` with the data mix, size and step of every row.

Usage
-----
    python data/prepare_datadecide.py --out data/processed/datadecide
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from discordrd.data import ResponseMatrix, save_matrix  # noqa: E402
from discordrd.pipelines.datadecide import BENCHMARKS  # noqa: E402

REPOSITORY = "yuhengtu/irsl_datadecide_data"
REVISION = "e9a43934067fc3d203e9aea90b1ba5931a412d9b"
FILE = "3_binary_matrix.parquet"
SHA256 = "50617e6982c1fc566776d7b3bc925bc7a1be6215aebfe5658573d177ae7ff04b"
META = ["model_data_mix", "model_size", "model_step", "model_split"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(target: Path) -> Path:
    if target.exists() and sha256(target) == SHA256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{FILE}"
    partial = target.with_suffix(".part")
    with urlopen(Request(url, headers={"User-Agent": "discordrd-data"}), timeout=120) as r, \
            partial.open("wb") as f:
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
    if sha256(partial) != SHA256:
        raise ValueError("checksum mismatch; the downloaded file is kept as .part")
    partial.replace(target)
    return target


def main() -> None:
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/processed/datadecide"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/datadecide") / FILE)
    args = parser.parse_args()
    path = download(args.raw)
    table = pq.read_table(path, columns=META, use_threads=False).to_pandas()
    index = table.index.to_frame(index=False) if not set(META) <= set(table.columns) else table[META]
    index = index.reset_index(drop=True)
    names = (index.model_data_mix.astype(str) + "::" + index.model_size.astype(str)
             + "::" + index.model_step.astype(str)).to_numpy()
    columns = {}
    for name in pq.ParquetFile(path).schema_arrow.names:
        if name in META:
            continue
        task, item = ast.literal_eval(name)
        columns.setdefault(str(task), []).append((name, str(item)))
    args.out.mkdir(parents=True, exist_ok=True)
    index[["model_data_mix", "model_size", "model_step"]].assign(model_name=names).to_csv(
        args.out / "checkpoints.csv", index=False)
    for task in BENCHMARKS:
        if task not in columns:
            raise KeyError(f"benchmark {task} not found in {FILE}")
        cols = [c for c, _ in columns[task]]
        values = pq.ParquetFile(path).read(columns=cols, use_threads=False).to_pandas().to_numpy(float)
        item_ids = np.asarray([i for _, i in columns[task]])
        save_matrix(args.out / f"{task}.npz",
                    ResponseMatrix(task, values, names, item_ids,
                                   np.full(values.shape[1], 1 / values.shape[1])))
        print(f"{task}: {values.shape[0]} checkpoints x {values.shape[1]} questions")


if __name__ == "__main__":
    main()
