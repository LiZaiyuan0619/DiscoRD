import numpy as np

from discordrd.metrics import ALL_METRICS, evaluate, expected_recall_at, expected_regret


def test_thirty_one_metrics():
    assert len(ALL_METRICS) == 31


def test_perfect_estimates():
    truth = np.linspace(0.1, 0.9, 60)
    m = evaluate(truth, truth)
    assert m["MAE"] == 0 and m["Recall@10"] == 1 and m["regret@1"] == 0 and m["Hit@1"] == 1


def test_ties_are_broken_in_expectation():
    truth = np.array([0.9, 0.8, 0.1, 0.0])
    pred = np.array([0.5, 0.5, 0.5, 0.0])
    # top-1 is one of three tied models: expected regret = 0.9 - mean(0.9, 0.8, 0.1)
    assert np.isclose(expected_regret(pred, truth), (0.9 - 0.6) * 100)
    # top-2 recall: two of the three tied slots, each holds a true top-2 model with prob 2/3
    assert np.isclose(expected_recall_at(pred, truth, 2), 2 / 3)


def test_top_k_metrics_need_k_candidates():
    truth = np.linspace(0, 1, 15)
    assert np.isnan(evaluate(truth, truth)["Recall@20"])
