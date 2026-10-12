"""Published estimators compared with DiscoRD, behind one interface.

Sources (Appendix C): tinyBenchmarks-IRT / -pIRT / -gpIRT (Maia Polo et al.,
2024); IRSL-CAT-1PL (Truong et al., 2026); Nested-MIRT-VOpt (Krumdick et al.,
2026); SparseEval-MLP and SparseEval-Linear (Zhang et al., 2026); mRMR+KRR and
Random+KRR (Bowyer et al., 2026); BenchPress (Zeng et al., 2026); the anchor
mean and AIPW (Zhang et al., 2026).  SparseEval-MLP is also a predictor inside
DiscoRD and lives in ``discordrd.sparseeval_mlp``.

Example
-------
>>> from discordrd.baselines import predict_baselines
>>> valid, query = predict_baselines(memory, valid_scores, query_scores, anchors,
...                                  item_weights, seed=42)
>>> query["Random+KRR"]
"""
from __future__ import annotations

import numpy as np

from .estimators import (aipw, anchor_mean_estimator, benchpress, irsl_cat_1pl, mrmr_krr,
                         nested_mirt_vopt, random_krr, sparseeval_linear, tinybenchmarks)

# name -> estimator; tinyBenchmarks returns its three variants at once.
ESTIMATORS = {
    "Anchor mean": anchor_mean_estimator,
    "AIPW": aipw,
    "BenchPress": benchpress,
    "IRSL-CAT-1PL": irsl_cat_1pl,
    "mRMR+KRR": mrmr_krr,
    "Random+KRR": random_krr,
    "Nested-MIRT-VOpt": nested_mirt_vopt,
    "SparseEval-Linear": sparseeval_linear,
    "tinyBenchmarks": tinybenchmarks,
}
BASELINES = ("AIPW", "Anchor mean", "BenchPress", "IRSL-CAT-1PL", "mRMR+KRR", "Random+KRR",
             "Nested-MIRT-VOpt", "SparseEval-MLP", "SparseEval-Linear",
             "tinyBenchmarks-IRT", "tinyBenchmarks-pIRT", "tinyBenchmarks-gpIRT")


def predict_baselines(memory, valid, query, anchors, item_weights, *, seed: int = 42,
                      names=None, form: str = "own", **options):
    """Run the published estimators on one benchmark of one run.

    Parameters
    ----------
    memory, valid, query : per-question scores of the past models, the
        validation models and the candidates
    anchors : the shared anchor questions (methods that choose their own
        questions use only their number)
    item_weights : weights of the questions in the full score
    seed : random seed of the fits
    names : estimators to run (keys of ``ESTIMATORS``); all by default
    form : ``"shared"`` (Open LLM Leaderboard) or ``"own"`` (other collections)
    options : passed to the estimators, e.g. ``epochs``, ``item_ids``, ``benchmark``

    Returns
    -------
    (valid, query) : dicts of predictions keyed by method name
    """
    names = list(ESTIMATORS) if names is None else list(names)
    valid_out, query_out = {}, {}
    for name in names:
        result = ESTIMATORS[name](memory, valid, query, np.asarray(anchors, int),
                                  np.asarray(item_weights, float), seed=seed, form=form, **options)
        if isinstance(result, dict):
            for key, (v, q) in result.items():
                valid_out[key], query_out[key] = v, q
        else:
            valid_out[name], query_out[name] = result
    return valid_out, query_out


__all__ = ["BASELINES", "ESTIMATORS", "predict_baselines"]
