import numpy as np
import pytest

from discordrd.fusion import fusion_weights, simplex_weights, smoothed_weights


def test_simplex_weights_lie_on_the_simplex_and_recover_a_mixture():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, 3))
    y = X @ np.array([0.7, 0.3, 0.0])
    w = simplex_weights(X, y)
    assert np.all(w >= -1e-9) and np.isclose(w.sum(), 1)
    assert np.allclose(w, [0.7, 0.3, 0.0], atol=1e-3)


def test_smoothing_follows_k_over_k_plus_v():
    w = np.array([1.0, 0.0, 0.0, 0.0])
    lam = 4 / (4 + 12)
    assert np.allclose(smoothed_weights(w, 12), (1 - lam) * w + lam / 4)


@pytest.mark.parametrize("rule", ["smoothed", "fitted", "equal", "best_single"])
def test_every_rule_returns_simplex_weights(rule):
    rng = np.random.default_rng(1)
    X, y = rng.uniform(size=(30, 6)), rng.uniform(size=30)
    w = fusion_weights(X, y, rule)
    assert w.shape == (6,) and np.all(w >= -1e-9) and np.isclose(w.sum(), 1)
