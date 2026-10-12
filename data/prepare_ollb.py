"""Build the Open LLM Leaderboard (v2) score matrices for OLLB-150 and OLLB-708.

Source: the second version of the Open LLM Leaderboard on the Hugging Face
Hub: the ``open-llm-leaderboard/contents`` table (one row per evaluated model,
with upload date and official scores) and the per-model ``-details`` datasets
with per-question results.  The details datasets are gated: set ``HF_TOKEN``
to a token whose account may request access (``--accept-access`` sends the
request).  Only per-question scores are read, never model weights.

Steps (all decided without looking at estimates):

1. ``manifest``  models uploaded before 2025-01-01 that are available on the
   Hub, with a unique name and finite scores on all six benchmarks; a sample
   of 1,200 stratified by upload quarter (seed 20260827).  The models uploaded
   from 2025-01-01 to 2025-03-31 form the OLLB-708 candidate pool.
2. ``download``  per-question records of every model on the six benchmarks
   (IFEval: strict prompt-level and instruction-level scores; BBH, GPQA, MuSR:
   acc_norm; MATH Lvl 5: exact_match; MMLU-Pro: acc).
3. ``build``     keep the models whose records cover the reference question
   set of every benchmark and reproduce the official score (within 0.01,
   except MATH Lvl 5); split the sampled models into memory (640), transfer
   (150), validation (150) and candidates (150) by whole organisations, once
   per split seed; write one matrix per benchmark and ``splits.csv`` for
   ``ollb_150`` (candidates: the 150 held-out models) and ``ollb_708``
   (candidates: the 2025 models; the other roles as in ``ollb_150``).

Usage
-----
    export HF_TOKEN=...
    python data/prepare_ollb.py manifest
    python data/prepare_ollb.py download --accept-access
    python data/prepare_ollb.py build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from discordrd.data import ResponseMatrix, save_matrix  # noqa: E402
from discordrd.protocol import SEEDS  # noqa: E402

HUB = "https://huggingface.co"
CONTENTS_REPO = "open-llm-leaderboard/contents"
PILOT_END = pd.Timestamp("2025-01-01")
SEALED_END = pd.Timestamp("2025-04-01")
SAMPLE_SEED = 20260827
N_SAMPLE = 1200

# benchmark: (raw score column in contents, config marker, metric column, aggregation)
BENCHMARKS = {
    "ifeval": ("IFEval Raw", "__leaderboard_ifeval", "ifeval_strict_prompt_instruction", "config_macro"),
    "bbh": ("BBH Raw", "__leaderboard_bbh_", "acc_norm", "config_macro"),
    "math_lvl5": ("MATH Lvl 5 Raw", "__leaderboard_math_", "exact_match", "item_micro"),
    "gpqa": ("GPQA Raw", "__leaderboard_gpqa_", "acc_norm", "item_micro"),
    "musr": ("MUSR Raw", "__leaderboard_musr_", "acc_norm", "config_macro"),
    "mmlu_pro": ("MMLU-PRO Raw", "__leaderboard_mmlu_pro", "acc", "item_micro"),
}
RAW_COLUMNS = [v[0] for v in BENCHMARKS.values()]
ROLE_OF = {"base_train": "memory", "transfer_train": "transfer", "valid": "validation", "test": "candidate"}


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
def _request(url: str, *, token: str | None = None, method: str = "GET", data: bytes | None = None):
    headers = {"User-Agent": "discordrd-data"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(8):
        try:
            with urlopen(Request(url, headers=headers, method=method, data=data), timeout=120) as r:
                return r.read()
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
        except OSError:
            if attempt == 7:
                raise
        time.sleep(min(60, 2 ** attempt))


def parquet_listing(repo: str, token: str | None) -> dict:
    """{config: {split: [parquet urls]}} of a Hub dataset."""
    return json.loads(_request(f"{HUB}/api/datasets/{repo}/parquet", token=token))


def read_parquet_url(url: str, token: str | None, columns: list[str]) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(_request(url, token=token))
        handle.flush()
        return pd.read_parquet(handle.name, columns=columns)


# ---------------------------------------------------------------------------
# step 1: manifest
# ---------------------------------------------------------------------------
def load_contents(token: str | None, cache: Path) -> pd.DataFrame:
    if not cache.exists():
        listing = parquet_listing(CONTENTS_REPO, token)
        urls = [u for splits in listing.values() for u in splits.get("train", [])]
        frames = [read_parquet_url(u, token, None) for u in urls]
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(cache)
    data = pd.read_parquet(cache).copy()
    data["upload_ts"] = pd.to_datetime(data["Upload To Hub Date"], errors="coerce")
    data["valid_fullname"] = data.fullname.astype(str).str.count("/").eq(1)
    data["organization"] = None
    data["details_repo"] = None
    valid = data.valid_fullname
    data.loc[valid, "organization"] = data.loc[valid, "fullname"].map(lambda n: n.split("/", 1)[0].lower())
    data.loc[valid, "details_repo"] = data.loc[valid, "fullname"].map(
        lambda n: "open-llm-leaderboard/" + n.replace("/", "__") + "-details")
    return data


def eligible(contents: pd.DataFrame, start=None, end=PILOT_END) -> pd.DataFrame:
    d = contents
    keep = (d.upload_ts.notna()
            & (d.upload_ts < end) & (True if start is None else d.upload_ts >= start)
            & d["Available on the hub"].fillna(False).astype(bool)
            & d.valid_fullname.fillna(False).astype(bool)
            & ~d.fullname.duplicated(keep=False)
            & np.isfinite(d[RAW_COLUMNS].to_numpy(float)).all(axis=1))
    out = d.loc[keep].copy()
    out["time_stratum"] = out.upload_ts.dt.to_period("Q").astype(str)
    return out


def stratified_sample(data: pd.DataFrame, n_models: int, seed: int) -> np.ndarray:
    """Sample proportionally to upload quarter (largest remainders), then shuffle."""
    rng = np.random.default_rng(seed)
    groups = [(key, np.asarray(idx, dtype=int))
              for key, idx in data.groupby("time_stratum", sort=True).indices.items()]
    ideal = np.asarray([len(idx) * n_models / len(data) for _, idx in groups])
    take = np.floor(ideal).astype(int)
    order = np.argsort(-(ideal - take), kind="stable")
    take[order[:n_models - int(take.sum())]] += 1
    chosen = []
    for (_, idx), count in zip(groups, take):
        if count:
            chosen.extend(rng.choice(idx, size=int(count), replace=False).tolist())
    chosen = np.asarray(chosen, dtype=int)
    rng.shuffle(chosen)
    return chosen


def assign_cohorts(organizations, seed: int) -> np.ndarray:
    """Whole organisations to transfer / validation / test cohorts of exact sizes; the rest is the memory."""
    organizations = np.asarray(organizations, dtype=str)
    n = len(organizations)
    size = 200 if n >= 1200 else 150 if n >= 800 else 50
    targets = {"transfer_train": size, "valid": size, "test": size}
    org_sizes = {org: int(np.count_nonzero(organizations == org)) for org in np.unique(organizations)}

    def exact_subset(remaining, target):
        reachable = {0: ()}
        for org in remaining:
            for total in sorted(tuple(reachable), reverse=True):
                new_total = total + org_sizes[org]
                if new_total <= target and new_total not in reachable:
                    reachable[new_total] = (*reachable[total], org)
            if target in reachable:
                return set(reachable[target])
        return None

    all_orgs = np.asarray(sorted(org_sizes), dtype=object)
    for attempt in range(256):
        rng = np.random.default_rng(seed + 17 + attempt * 1009)
        remaining = all_orgs.copy()
        rng.shuffle(remaining)
        chosen = {}
        for cohort, target in targets.items():
            selected = exact_subset(remaining.tolist(), target)
            if selected is None:
                break
            chosen[cohort] = selected
            remaining = np.asarray([x for x in remaining if x not in selected], dtype=object)
        else:
            labels = np.full(n, "base_train", dtype=object)
            for cohort, orgs in chosen.items():
                labels[np.isin(organizations, list(orgs))] = cohort
            return labels
    raise ValueError("could not construct exact organisation-disjoint cohorts")


def cmd_manifest(args) -> None:
    contents = load_contents(args.token, args.raw / "contents.parquet")
    pool = eligible(contents)
    sample = pool.iloc[stratified_sample(pool, N_SAMPLE, SAMPLE_SEED)].reset_index(drop=True)
    sealed = eligible(contents, start=PILOT_END, end=SEALED_END).reset_index(drop=True)
    columns = ["eval_name", "fullname", "organization", "details_repo", "upload_ts", "time_stratum", *RAW_COLUMNS]
    args.raw.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(args.raw / "manifest_sample.csv", index=False)
    sealed[columns].to_csv(args.raw / "manifest_2025.csv", index=False)
    print(f"{len(sample)} sampled models before 2025, {len(sealed)} models from 2025-Q1")


# ---------------------------------------------------------------------------
# step 2: per-question records
# ---------------------------------------------------------------------------
def records_for(repo: str, bench: str, token: str | None, listing: dict) -> pd.DataFrame:
    _, marker, metric, _ = BENCHMARKS[bench]
    parts = [(config, url) for config, splits in sorted(listing.items()) if marker in config
             for url in splits.get("latest", [])]
    if not parts:
        raise ValueError(f"{repo}: no {bench} results")
    rows = []
    for config, url in parts:
        if bench == "ifeval":
            frame = read_parquet_url(url, token, ["doc_id", "prompt_level_strict_acc", "inst_level_strict_acc"])
            for rec in frame.itertuples(index=False):
                rows.append((f"{config}::prompt_strict", f"prompt_strict:{rec.doc_id}",
                             rec.prompt_level_strict_acc))
                rows.extend((f"{config}::instruction_strict", f"instruction_strict:{rec.doc_id}:{j}", v)
                            for j, v in enumerate(list(rec.inst_level_strict_acc)))
        else:
            frame = read_parquet_url(url, token, ["doc_id", metric])
            rows.extend((config, d, s) for d, s in zip(frame.doc_id, frame[metric]))
    return pd.DataFrame(rows, columns=["config", "doc_id", "score"])


def cmd_download(args) -> None:
    models = pd.concat([pd.read_csv(args.raw / "manifest_sample.csv"),
                        pd.read_csv(args.raw / "manifest_2025.csv")], ignore_index=True)

    def one(row):
        target = args.raw / "records"
        if all((target / b / f"{row.eval_name}.parquet").exists() for b in BENCHMARKS):
            return
        if args.accept_access:
            _request(f"{HUB}/datasets/{row.details_repo}/ask-access", token=args.token,
                     method="POST", data=b"{}")
        listing = parquet_listing(row.details_repo, args.token)
        for bench in BENCHMARKS:
            path = target / bench / f"{row.eval_name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                records_for(row.details_repo, bench, args.token, listing).to_parquet(path)
            except Exception as error:  # a missing benchmark excludes the model later
                print(f"{row.eval_name} {bench}: {type(error).__name__}: {error}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, models.itertuples(index=False)))


# ---------------------------------------------------------------------------
# step 3: matrices and splits
# ---------------------------------------------------------------------------
def clean_records(records: pd.DataFrame) -> pd.DataFrame:
    out = records[["config", "doc_id", "score"]].copy()
    score = pd.to_numeric(out.score, errors="coerce").to_numpy(float)
    rounded = np.rint(score)
    if not np.isfinite(score).all() or not np.allclose(score, rounded, atol=1e-8):
        raise ValueError("scores are not binary")
    out["score"] = rounded.astype(np.uint8)
    out["config"] = out.config.astype(str).map(
        lambda v: v[v.index("__leaderboard_") + 2:] if "__leaderboard_" in v else v)
    if out.duplicated(["config", "doc_id"]).any():
        raise ValueError("duplicate questions")
    out["item_id"] = out.config.astype(str) + "::" + out.doc_id.astype(str)
    return out.sort_values("item_id", kind="stable").reset_index(drop=True)


def item_weights(clean: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "item_micro":
        return np.full(len(clean), 1.0 / len(clean))
    counts = clean.groupby("config").size().to_dict()
    return np.asarray([1.0 / (len(counts) * counts[c]) for c in clean.config])


def load_benchmark(raw: Path, bench: str, names) -> dict[str, pd.DataFrame]:
    out = {}
    for name in names:
        path = raw / "records" / bench / f"{name}.parquet"
        if path.exists():
            try:
                out[name] = clean_records(pd.read_parquet(path))
            except ValueError:
                pass
    return out


def cmd_build(args) -> None:
    sample = pd.read_csv(args.raw / "manifest_sample.csv")
    sealed = pd.read_csv(args.raw / "manifest_2025.csv")
    names = pd.concat([sample.eval_name, sealed.eval_name]).astype(str).tolist()
    official = pd.concat([sample, sealed]).set_index("eval_name")
    tables, reference, complete = {}, {}, set(names)
    for bench in BENCHMARKS:
        tables[bench] = load_benchmark(args.raw, bench, names)
        # the reference question set: the most common one among the sampled models
        keys = pd.Series({n: tuple(t.item_id) for n, t in tables[bench].items()
                          if n in set(sample.eval_name)})
        reference[bench] = keys.value_counts().index[0]
        complete &= {n for n, t in tables[bench].items() if tuple(t.item_id) == reference[bench]}
    matrices, admitted = {}, set(complete)
    for bench, (raw_column, _, _, mode) in BENCHMARKS.items():
        rows = [n for n in names if n in complete]
        Y = np.stack([tables[bench][n].score.to_numpy(np.uint8) for n in rows])
        w = item_weights(tables[bench][rows[0]], mode)
        error = np.abs(Y @ w - official.loc[rows, raw_column].to_numpy(float))
        if bench != "math_lvl5":
            admitted -= {n for n, e in zip(rows, error) if e >= 0.01}
        matrices[bench] = (Y, np.asarray(rows), np.asarray(reference[bench]), w)
    pilot = sample[sample.eval_name.isin(admitted)].reset_index(drop=True)
    pilot = pilot.iloc[stratified_sample(pilot, len(pilot), SAMPLE_SEED)].reset_index(drop=True)
    later = sealed.eval_name[sealed.eval_name.isin(admitted)].astype(str).tolist()
    splits = pd.DataFrame({"model_name": pilot.eval_name.astype(str)})
    for seed in SEEDS:
        splits[f"seed{seed}"] = [ROLE_OF[c] for c in assign_cohorts(pilot.organization, seed)]
    settings = {
        "ollb_150": splits,
        "ollb_708": pd.concat([
            splits.assign(**{f"seed{s}": splits[f"seed{s}"].where(splits[f"seed{s}"] != "candidate", "unused")
                             for s in SEEDS}),
            pd.DataFrame({"model_name": later, **{f"seed{s}": "candidate" for s in SEEDS}})],
            ignore_index=True),
    }
    for setting, roles in settings.items():
        out = args.out / setting
        out.mkdir(parents=True, exist_ok=True)
        roles.to_csv(out / "splits.csv", index=False)
        for bench, (Y, rows, items, w) in matrices.items():
            position = {n: i for i, n in enumerate(rows)}
            take = np.asarray([position[n] for n in roles.model_name], int)
            save_matrix(out / f"{bench}.npz",
                        ResponseMatrix(bench, Y[take].astype(float), rows[take], items, w))
        print(f"{setting}: {len(roles)} models, roles of seed 42: "
              f"{roles.seed42.value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("manifest", "download", "build"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ollb"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/ollb"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--accept-access", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    {"manifest": cmd_manifest, "download": cmd_download, "build": cmd_build}[args.step](args)


if __name__ == "__main__":
    main()
