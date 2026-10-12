"""Simple predictors used inside DiscoRD: the anchor mean and Ridge transfer."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV

RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def anchor_mean(answers, anchors, item_weights=None, *, horvitz_thompson: bool = False) -> np.ndarray:
    """A model's mean answer on the anchor questions of one benchmark.

    Parameters
    ----------
    answers : array of shape (n_models, n_questions)
        Per-question scores of each model (0/1, or in [0, 1] for AlpacaEval).
    anchors : integer indices of the anchor questions.
    item_weights : weight of each question in the full-benchmark score
        (uniform when None).  Benchmarks that macro-average their subtasks give
        unequal weights.
    horvitz_thompson : if True, return the Horvitz-Thompson form
        ``sum_a w_a x_a * n_questions / n_anchors``, which equals the weighted
        mean when the anchors are drawn uniformly and the weights are uniform.
    """
    answers = np.asarray(answers)
    anchors = np.asarray(anchors, int)
    if item_weights is None:
        return answers[:, anchors].astype(float).mean(axis=1)
    w = np.asarray(item_weights, float)
    if horvitz_thompson:
        return answers[:, anchors] @ w[anchors] * (len(w) / len(anchors))
    mass = w[anchors]
    return answers[:, anchors].astype(float) @ mass / mass.sum()


def ridge_transfer(train_features, train_targets, valid_features, query_features):
    """Ridge regression fitted on past models; predictions clipped to [0, 1].

    The penalty is chosen by efficient leave-one-out cross-validation over 13
    values from 1e-3 to 1e3 (``RidgeCV``).  In Ridge transfer the features of
    a model are its estimates on the other benchmarks of the suite and the
    target is its full score on the target benchmark.  On single-benchmark
    collections the features are the anchor answers themselves.

    Returns
    -------
    (valid_predictions, query_predictions)
    """
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(np.asarray(train_features, float),
                                             np.asarray(train_targets, float))
    return (np.clip(model.predict(np.asarray(valid_features, float)), 0, 1),
            np.clip(model.predict(np.asarray(query_features, float)), 0, 1))
