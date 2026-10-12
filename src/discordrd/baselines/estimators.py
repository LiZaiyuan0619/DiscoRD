"""The published estimators compared with DiscoRD (Appendix C, "Baselines").

Every function has the same signature,

    f(memory, valid, query, anchors, item_weights, *, seed, **options) -> (valid_pred, query_pred)

where ``memory``, ``valid`` and ``query`` are per-question score matrices of the
past models, the validation models and the candidates.  Every method answers
A questions per benchmark.  Methods that do not choose their own questions
read only the shared ``anchors``; mRMR+KRR, IRSL-CAT-1PL and Nested-MIRT-VOpt
choose their own A questions in every collection, and tinyBenchmarks chooses
its own outside the Open LLM Leaderboard.  A choice inside a method (such as
the mixing weight of tinyBenchmarks-gpIRT) is made on the validation set.

``form="shared"`` gives the variant used on the Open LLM Leaderboard (all
methods read SparseEval's anchor questions; IRT parameters from a 1PL model of
the memory); ``form="own"`` gives the variant used on WILD, DataDecide and
AlpacaEval (scores in [0, 1] are read as soft labels where needed).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV

from ..predictors import anchor_mean
from . import irt
from .krr import KRR_ALPHAS, KRR_ALPHAS_WIDE, fit_polynomial2_krr, select_mrmr_miq

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
GP_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _clip(x):
    return np.clip(np.asarray(x, float), 0, 1)


def _truth(x, w):
    return np.asarray(x, float) @ np.asarray(w, float)


# ---------------------------------------------------------------------------
# Anchor mean, SparseEval-Linear, AIPW
# ---------------------------------------------------------------------------
def anchor_mean_estimator(memory, valid, query, anchors, item_weights, *, seed=0, **_):
    """The candidate's weighted mean answer on the anchor questions."""
    return (anchor_mean(valid, anchors, item_weights),
            anchor_mean(query, anchors, item_weights))


def sparseeval_linear(memory, valid, query, anchors, item_weights, *, seed=0, **_):
    """SparseEval-Linear: a linear (ridge) regression of the full score on the anchor answers."""
    a = np.asarray(anchors, int)
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(np.asarray(memory, float)[:, a],
                                             _truth(memory, item_weights))
    return (_clip(model.predict(np.asarray(valid, float)[:, a])),
            _clip(model.predict(np.asarray(query, float)[:, a])))


def aipw(memory, valid, query, anchors, item_weights, *, seed=0, form="own", **_):
    """AIPW: a 1PL prediction of every question, corrected by its residuals on the anchors."""
    a = np.asarray(anchors, int)
    w = np.asarray(item_weights, float)
    z = irt.memory_easiness(memory)
    out = []
    for x in (valid, query):
        values = []
        for row in np.asarray(x, float):
            theta = irt.theta_map_1pl(row[a], z[a])
            prob = expit(theta + z)
            if form == "shared":
                values.append(prob @ w / w.sum()
                              + (w[a] * (row[a] - prob[a])).sum() / w[a].sum())
            else:
                values.append(prob @ w + len(w) / len(a) * ((row[a] - prob[a]) @ w[a]))
        out.append(_clip(values))
    return tuple(out)


# ---------------------------------------------------------------------------
# Kernel ridge regression
# ---------------------------------------------------------------------------
def random_krr(memory, valid, query, anchors, item_weights, *, seed=0, form="own", **_):
    """Random+KRR: degree-2 polynomial KRR on the shared anchor answers."""
    a = np.asarray(anchors, int)
    fit = fit_polynomial2_krr(np.asarray(memory, float)[:, a], _truth(memory, item_weights),
                              alphas=KRR_ALPHAS if form == "shared" else KRR_ALPHAS_WIDE)
    return (_clip(fit.predict(np.asarray(valid, float)[:, a])),
            _clip(fit.predict(np.asarray(query, float)[:, a])))


def mrmr_krr(memory, valid, query, anchors, item_weights, *, seed=0, form="own",
             max_budget=None, **_):
    """mRMR+KRR: KRR on A questions chosen on the memory by mRMR (MIQ)."""
    budget = len(anchors)
    memory = np.asarray(memory)
    if np.isin(memory, [0, 1]).all():
        order = select_mrmr_miq(memory.astype(np.float32), max_anchors=max_budget or budget,
                                n_jobs=1).anchors
    else:
        from .mrmr_continuous import mrmr_miq_order
        order = mrmr_miq_order(memory.astype(float), max_budget or budget, seed=seed)
    selected = np.asarray(order, int)[:budget]
    fit = fit_polynomial2_krr(memory.astype(float)[:, selected], _truth(memory, item_weights),
                              alphas=KRR_ALPHAS if form == "shared" else KRR_ALPHAS_WIDE)
    return (_clip(fit.predict(np.asarray(valid, float)[:, selected])),
            _clip(fit.predict(np.asarray(query, float)[:, selected])))


# ---------------------------------------------------------------------------
# IRT-based estimators
# ---------------------------------------------------------------------------
def nested_mirt_vopt(memory, valid, query, anchors, item_weights, *, seed=0, form="own",
                     mirt_epochs=300, **_):
    """Nested-MIRT-VOpt: Nested MIRT on the memory, V-optimal questions, MAP ability."""
    w = np.asarray(item_weights, float)
    if form == "shared":
        params = irt.fit_nested_mirt(np.asarray(memory, float), k=3, nested_lambda=1.0,
                                     epochs=mirt_epochs)
    else:
        params = irt.fit_soft_mirt(memory, seed, k=3, nested_lambda=1.0, epochs=mirt_epochs)
    chosen = irt.v_optimal_order(params, len(anchors))
    out = []
    for x in (valid, query):
        out.append(_clip([irt.predict_probabilities(irt.map_theta(chosen, row[chosen], params), params) @ w
                          for row in np.asarray(x, float)]))
    return tuple(out)


def irsl_cat_1pl(memory, valid, query, anchors, item_weights, *, seed=0, form="own", **_):
    """IRSL-CAT-1PL: adaptive testing with a 1PL model of the memory.

    The first 20% of the budget spans the easiness range; each further
    question maximises the 1PL Fisher information at the current ability.
    """
    budget = len(anchors)
    w = np.asarray(item_weights, float)
    z = irt.memory_easiness(memory)
    out = []
    for x in (valid, query):
        values = []
        for row in np.asarray(x, float):
            theta = _cat_theta(row, z, budget)
            values.append(expit(theta + z).mean() if form == "shared" else expit(theta + z) @ w)
        out.append(_clip(values))
    return tuple(out)


def _span_indices(z, k, trim=0.10):
    """Initial questions spread over the easiness quantiles (IRSL's span initialisation)."""
    z = np.asarray(z, float)
    if k <= 0:
        return []
    lo, hi = np.quantile(z, [trim, 1.0 - trim])
    used = np.zeros(len(z), dtype=bool)
    out = []
    for target in np.linspace(lo, hi, num=k):
        distance = np.abs(z - target)
        distance[used] = np.inf
        j = int(np.argmin(distance))
        out.append(j)
        used[j] = True
    return out


def _cat_theta(row, z, budget):
    chosen = _span_indices(z, min(max(1, int(.2 * budget)), budget))
    remaining = np.ones(len(row), dtype=bool)
    remaining[chosen] = False
    while True:
        theta = irt.theta_mle_1pl(row[chosen], z[chosen])
        if len(chosen) == budget:
            return theta
        rem = np.flatnonzero(remaining)
        winner = int(rem[np.argmin(abs(theta + z[rem]))])
        chosen.append(winner)
        remaining[winner] = False


def _tiny_parts_shared(row, anchors, z, item_weights, balance):
    """IRT (mean of the anchor answers) and p-IRT (data part + 1PL extrapolation)."""
    y = row[anchors]
    theta = irt.theta_map_1pl(y, z[anchors])
    extrap = float(expit(theta + z) @ item_weights)
    data_part = float((balance[anchors] * y).mean())
    lam = len(anchors) / len(z)
    return float(y.mean()), lam * data_part + (1 - lam) * extrap


def _balance_weights(item_ids, benchmark):
    """tinyBenchmarks' balance weights over sub-groups of questions (IFEval: one group per prompt type)."""
    ids = np.asarray(item_ids, str)
    if benchmark == "ifeval":
        keys = ["::".join(i.split("::")[:2]) for i in ids]
    else:
        keys = list(ids)
    groups = {}
    for pos, key in enumerate(keys):
        groups.setdefault(key, []).append(pos)
    bw = np.ones(len(ids))
    for idx in groups.values():
        bw[np.asarray(idx, int)] = len(ids) / (len(groups) * len(idx))
    return bw


def _tiny_selected(params, budget, seed):
    """tinyBenchmarks' own questions: k-means on the item embeddings, one representative per cluster."""
    a = params.alpha[:, None] * np.column_stack((np.ones(len(params.alpha)), params.loading))
    b = params.alpha * params.difficulty
    features = np.column_stack((a, np.repeat((b / a.shape[1])[:, None], a.shape[1], axis=1)))
    km = KMeans(n_clusters=budget, random_state=seed, n_init=10).fit(features)
    distances = cdist(km.cluster_centers_, features)
    chosen = []
    for row in distances:
        order = np.argsort(row, kind="stable")
        chosen.append(next(int(i) for i in order if i not in chosen))
    return np.asarray(chosen), features


def tinybenchmarks(memory, valid, query, anchors, item_weights, *, seed=0, form="own",
                   item_ids=None, benchmark=None, tiny_epochs=1000, **_):
    """tinyBenchmarks-IRT, -pIRT and -gpIRT (Maia Polo et al., 2024).

    Returns {"tinyBenchmarks-IRT": (valid, query), "tinyBenchmarks-pIRT": ...,
    "tinyBenchmarks-gpIRT": ...}.  gp-IRT mixes the other two with a weight
    chosen on the validation set.
    """
    w = np.asarray(item_weights, float)
    vy = _truth(valid, w)
    parts = {}
    if form == "shared":
        a = np.asarray(anchors, int)
        z = irt.memory_easiness(memory)
        balance = _balance_weights(item_ids if item_ids is not None else np.arange(len(w)).astype(str),
                                   benchmark)
        for side, x in (("valid", valid), ("query", query)):
            parts[side] = np.asarray([_tiny_parts_shared(row, a, z, w, balance)
                                      for row in np.asarray(x, float)])
    else:
        params = irt.fit_soft_mirt(memory, seed, k=14, nested_lambda=0, epochs=tiny_epochs)
        selected, features = _tiny_selected(params, len(anchors), seed)
        assign = cdist(features, features[selected]).argmin(1)
        mass = np.bincount(assign, weights=w, minlength=len(anchors))
        unseen = np.setdiff1d(np.arange(len(w)), selected)
        for side, x in (("valid", valid), ("query", query)):
            rows = []
            for row in np.asarray(x, float):
                obs = row[selected]
                prob = irt.predict_probabilities(irt.map_theta(selected, obs, params), params)
                rows.append((obs @ mass, obs @ w[selected] + prob[unseen] @ w[unseen]))
            parts[side] = np.asarray(rows)
    grid = np.asarray(GP_LAMBDAS)
    losses = np.abs(parts["valid"][:, 0, None] * grid
                    + parts["valid"][:, 1, None] * (1 - grid) - vy[:, None]).mean(0)
    lam = float(grid[losses.argmin()])
    result = {}
    for name, col in (("tinyBenchmarks-IRT", 0), ("tinyBenchmarks-pIRT", 1)):
        result[name] = (_clip(parts["valid"][:, col]), _clip(parts["query"][:, col]))
    result["tinyBenchmarks-gpIRT"] = tuple(
        _clip(lam * parts[s][:, 0] + (1 - lam) * parts[s][:, 1]) for s in ("valid", "query"))
    return result


# ---------------------------------------------------------------------------
# BenchPress (matrix completion)
# ---------------------------------------------------------------------------
def als_complete(Y, memory_n, observed, *, rank=8, iterations=12, ridge=1.0) -> np.ndarray:
    """Complete the candidates' rows of a score matrix by ridge alternating least squares.

    The memory rows are fully observed; the other rows observe only the
    ``observed`` columns.  Returns the completed matrix, clipped to [0, 1].
    """
    Y = np.asarray(Y, float)
    n, m = Y.shape
    mask = np.zeros_like(Y, dtype=bool)
    mask[:memory_n, :] = True
    query_rows = np.arange(memory_n, n)
    mask[np.ix_(query_rows, observed)] = True
    col_mean = np.array([Y[mask[:, j], j].mean() for j in range(m)])
    Z = Y.copy()
    Z[~mask] = np.tile(col_mean, (n, 1))[~mask]
    U = np.zeros((n, rank))
    V = np.random.default_rng(0).uniform(-0.01, 0.01, size=(m, rank))
    for _ in range(iterations):
        for i in range(n):
            seen = mask[i]
            U[i] = np.linalg.solve(V[seen].T @ V[seen] + ridge * np.eye(rank), Z[i, seen] @ V[seen])
        for j in range(m):
            seen = mask[:, j]
            V[j] = np.linalg.solve(U[seen].T @ U[seen] + ridge * np.eye(rank), Z[seen, j] @ U[seen])
        filled = U @ V.T
        Z[~mask] = filled[~mask]
    return np.clip(Z, 0.0, 1.0)


def benchpress(memory, valid, query, anchors, item_weights, *, seed=0, **_):
    """BenchPress: complete each candidate's row from its anchor answers and the memory."""
    a = np.asarray(anchors, int)
    w = np.asarray(item_weights, float)
    memory = np.asarray(memory, float)
    out = []
    for x in (valid, query):
        hidden = np.zeros_like(np.asarray(x, float))
        hidden[:, a] = np.asarray(x, float)[:, a]
        completed = als_complete(np.vstack((memory, hidden)), len(memory), a)
        out.append(completed[len(memory):] @ w)
    return tuple(out)
