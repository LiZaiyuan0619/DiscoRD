"""End-to-end smoke tests on a tiny synthetic matrix (a few seconds on a CPU)."""
import numpy as np
import pandas as pd

from discordrd.baselines import BASELINES, predict_baselines
from discordrd.data import ResponseMatrix
from discordrd.discord import PREDICTORS
from discordrd.pipelines import alpacaeval, wild
from discordrd.pipelines.common import single_benchmark_discord

FAST = {"num_epochs": 300, "eval_every": 50, "patience": 3}


def test_single_benchmark_discord_runs(synthetic):
    rng = np.random.default_rng(0)
    anchors = rng.permutation(synthetic.shape[1])[:20]
    out, data = single_benchmark_discord(synthetic[:70], synthetic[70:95], synthetic[95:], anchors,
                                         fit_seed=0, sparseeval_options=FAST)
    assert out.query.shape == (25,) and np.isfinite(out.query).all()
    assert set(out.weights) == set(PREDICTORS) and np.isclose(sum(out.weights.values()), 1)


def test_alpacaeval_cell_with_extras(synthetic):
    matrix = ResponseMatrix("toy", synthetic, np.array([f"m{i}" for i in range(len(synthetic))]),
                            np.arange(synthetic.shape[1]).astype(str),
                            np.full(synthetic.shape[1], 1 / synthetic.shape[1]))
    result = alpacaeval.run_cell(matrix, seed=0, repeat=0, budget=20, sparseeval_options=FAST,
                                 extras=True)
    rows = pd.DataFrame(result.metric_rows())
    assert {"DiscoRD", "- discordance predictors", "Stratified correction, Q=4"} <= set(rows.method)
    assert rows.MAE.notna().all()


def test_wild_cell_pools_folds(synthetic):
    names = np.array([f"org{i % 6}/model{i}" for i in range(48)])
    tasks = {f"t{j}": ResponseMatrix(f"t{j}", synthetic[:48, 100 * j:100 * (j + 1)], names,
                                     np.arange(100).astype(str), np.full(100, 0.01))
             for j in range(3)}
    suite = wild.Suite(tasks, min_questions=50)
    results = wild.run_cell(suite, seed=0, repeat=0, budget=10, epochs=200)
    assert len(results) == 3 and all(len(r.truth) == 48 for r in results)
    assert np.isfinite(results[0].predictions["DiscoRD"]).all()


def test_published_estimators_run(synthetic):
    rng = np.random.default_rng(0)
    anchors = rng.permutation(synthetic.shape[1])[:10]
    w = np.full(synthetic.shape[1], 1 / synthetic.shape[1])
    _, query = predict_baselines(synthetic[:70], synthetic[70:95], synthetic[95:], anchors, w,
                                 seed=0, mirt_epochs=3, tiny_epochs=3)
    assert set(query) == set(BASELINES) - {"SparseEval-MLP"}
    assert all(np.isfinite(v).all() and v.shape == (25,) for v in query.values())


def test_ollb_two_stage_cell(synthetic):
    from discordrd.pipelines import ollb
    names = np.array([f"model{i}" for i in range(120)])
    matrices = {f"b{j}": ResponseMatrix(f"b{j}", synthetic[:, 50 * j:50 * (j + 1)], names,
                                        np.arange(50).astype(str), np.full(50, 0.02))
                for j in range(6)}
    roles = {"memory": names[:60], "transfer": names[60:80], "validation": names[80:100],
             "candidate": names[100:]}
    fast = {"refinement_steps": 1, "proxy_epochs": 5, "num_epochs": 100, "eval_every": 20, "patience": 2}
    stage1 = {b: ollb.fit_sparseeval(m, roles, 5, seed=0, repeat=0, **fast) for b, m in matrices.items()}
    results = ollb.run_cell(matrices, roles, stage1, "toy", 5, 0, 0, extras=True)
    assert len(results) == 6 and all(np.isfinite(r.predictions["DiscoRD"]).all() for r in results)


def test_datadecide_cell(synthetic):
    from discordrd.pipelines import datadecide
    meta = pd.DataFrame([(f"mix{m}", size, step) for m in range(12) for size in ("4M", "20M")
                         for step in range(5)], columns=["model_data_mix", "model_size", "model_step"])
    index = datadecide.checkpoint_index(meta)
    names = (meta.model_data_mix + "::" + meta.model_size + "::" + meta.model_step.astype(str)).to_numpy()
    matrix = ResponseMatrix("toy", synthetic, names, np.arange(300).astype(str), np.full(300, 1 / 300))
    for memory in datadecide.MEMORIES:
        result = datadecide.run_cell(matrix, index, memory, seed=0, repeat=0, budget=20, epochs=200)
        assert np.isfinite(result.predictions["DiscoRD"]).all()


def test_published_estimators_shared_form(synthetic):
    anchors = np.arange(0, 300, 30)
    w = np.full(300, 1 / 300)
    ids = np.array([f"leaderboard_ifeval::prompt_strict::{i % 7}" for i in range(300)])
    _, query = predict_baselines(synthetic[:70], synthetic[70:95], synthetic[95:], anchors, w,
                                 seed=0, form="shared", mirt_epochs=3, item_ids=ids, benchmark="ifeval")
    assert all(np.isfinite(v).all() for v in query.values())
