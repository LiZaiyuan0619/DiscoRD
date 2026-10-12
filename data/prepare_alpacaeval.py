"""Build the AlpacaEval score matrix from the public annotations.

Source: the per-instruction annotations released in the AlpacaEval repository
(https://github.com/tatsu-lab/alpaca_eval, CC BY-NC 4.0), pinned to one
commit.  For every model directory under ``results/`` we read the annotations
of the ``weighted_alpaca_eval_gpt4_turbo`` judge against the same reference
model (gpt4_1106_preview) and use ``preference - 1`` in [0, 1] as the score of
the model on that instruction.

Selection, decided without looking at any score:
* the same judge and reference throughout;
* drop result directories marked as gamed;
* keep models that cover at least 95% of the maximum number of instructions;
* one directory per generator (most coverage, then name) and no two models
  with identical outputs on the common instructions;
* keep the instructions answered by every kept model (with the same reference output).

No model or judge is queried.  Only scores are stored; instruction texts are
replaced by a hash.

Usage
-----
    python data/prepare_alpacaeval.py --out data/processed/alpacaeval
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from discordrd.data import ResponseMatrix, save_matrix  # noqa: E402

REPOSITORY = "tatsu-lab/alpaca_eval"
COMMIT = "cd543a149df89434d8a54582c0151c0b945c3d20"
JUDGE = "weighted_alpaca_eval_gpt4_turbo"
REFERENCE = "gpt4_1106_preview"


def fetch(url: str, limit: int | None = None) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "discordrd-data"}), timeout=60) as r:
                raw = r.read() if limit is None else r.read(limit + 1)
            if limit is not None and len(raw) > limit:
                raise ValueError("download exceeds the declared size")
            return raw
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def annotation_entries() -> list[dict]:
    url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(fetch(url))
    if tree.get("truncated"):
        raise RuntimeError("the repository tree listing is truncated")
    return [e for e in tree["tree"] if len(e["path"].split("/")) == 4
            and e["path"].startswith("results/") and e["path"].endswith(f"/{JUDGE}/annotations.json")]


def compact(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        value = r.get("preference")
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or not 1 <= value <= 2:
            raise ValueError("preference outside [1, 2]")
        if r.get("annotator") != JUDGE:
            raise ValueError("mixed judge")
        if not all(isinstance(r.get(k), str) for k in ("instruction", "output_1", "output_2",
                                                       "generator_1", "generator_2")):
            raise ValueError("missing identity fields")
        key = json.dumps([r.get("dataset", ""), r["instruction"]], ensure_ascii=False)
        rows.append({"item": key, "score": value - 1, "reference": r["generator_1"],
                     "generator": r["generator_2"],
                     "reference_output_sha256": hashlib.sha256(r["output_1"].encode()).hexdigest(),
                     "output_sha256": hashlib.sha256(r["output_2"].encode()).hexdigest()})
    if len({r["item"] for r in rows}) != len(rows):
        raise ValueError("duplicate instructions in one annotation file")
    if len({r["generator"] for r in rows}) != 1:
        raise ValueError("several generators in one result directory")
    if any(r["reference"] != REFERENCE for r in rows):
        raise ValueError("reference model differs")
    return rows


def download(entry: dict, cache: Path) -> dict:
    name = entry["path"].split("/")[1]
    target = cache / f"{name}.json"
    if target.exists():
        return json.loads(target.read_text())
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{quote(entry['path'], safe='/')}"
    raw = fetch(url, max(entry["size"], 1) + 1024)
    digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if digest != entry["sha"]:
        raise ValueError(f"{name}: download does not match the pinned Git blob")
    record = {"directory": name}
    try:
        record["rows"] = compact(json.loads(raw))
    except (ValueError, TypeError, KeyError) as exc:
        record["excluded_reason"] = str(exc)
    target.write_text(json.dumps(record))
    return record


def normalized_items(records):
    """Recover a missing dataset label only from an identical instruction text."""
    labels = {}
    for rec in records:
        for row in rec["rows"]:
            dataset, instruction = json.loads(row["item"])
            if dataset:
                labels.setdefault(instruction, set()).add(dataset)
    if any(len(v) != 1 for v in labels.values()):
        raise ValueError("ambiguous dataset label for an identical instruction")
    out = []
    for rec in records:
        rows = []
        for row in rec["rows"]:
            dataset, instruction = json.loads(row["item"])
            if not dataset and instruction in labels:
                dataset = next(iter(labels[instruction]))
            rows.append({**row, "item": json.dumps([dataset, instruction], ensure_ascii=False)})
        if len({row["item"] for row in rows}) != len(rows):
            raise ValueError("duplicate items after label recovery")
        out.append({**rec, "rows": rows})
    return out


def build_matrix(records) -> ResponseMatrix:
    eligible = []
    for r in sorted(records, key=lambda v: v["directory"]):
        if r.get("excluded_reason") or "gamed" in r["directory"].lower():
            continue
        eligible.append(r)
    eligible = normalized_items(eligible)
    max_items = max(len(r["rows"]) for r in eligible)
    filtered = [r for r in eligible if len(r["rows"]) >= .95 * max_items]
    filtered.sort(key=lambda r: (-len(r["rows"]), r["directory"]))
    seen, kept = set(), []
    for r in filtered:
        generator = r["rows"][0]["generator"]
        if generator not in seen:
            seen.add(generator)
            kept.append(r)
    maps = [{row["item"]: row for row in rec["rows"]} for rec in kept]
    common = set.intersection(*(set(m) for m in maps))
    common = sorted(k for k in common if len({m[k]["reference_output_sha256"] for m in maps}) == 1)
    seen_outputs, keep = set(), []
    for i, m in enumerate(maps):
        identity = tuple(m[k]["output_sha256"] for k in common)
        if identity not in seen_outputs:
            seen_outputs.add(identity)
            keep.append(i)
    names = np.array([kept[i]["directory"] for i in keep])
    scores = np.array([[maps[i][k]["score"] for k in common] for i in keep], float)
    item_ids = np.array([hashlib.sha256(k.encode()).hexdigest()[:16] for k in common])
    return ResponseMatrix("alpacaeval", scores, names, item_ids,
                          np.full(len(common), 1 / len(common)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/processed/alpacaeval"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/alpacaeval"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    entries = annotation_entries()
    print(f"{len(entries)} annotation files at commit {COMMIT[:10]}")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, e, args.cache) for e in entries]
        for future in as_completed(futures):
            records.append(future.result())
    matrix = build_matrix(records)
    save_matrix(args.out / "alpacaeval.npz", matrix)
    print(f"saved {matrix.n_models} models x {matrix.n_questions} instructions to {args.out}")


if __name__ == "__main__":
    main()
