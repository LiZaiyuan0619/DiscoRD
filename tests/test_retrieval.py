import numpy as np

from discordrd.distances import response_distances, score_distances
from discordrd.retrieval import consensus_retrieval, discordance_retrieval, knn_predict


def test_knn_predict_inverse_distance_weights():
    distances = np.array([[1.0, 2.0, 4.0]])
    values = np.array([0.2, 0.5, 0.9])
    # k = 2: weights 1 and 1/2 on the first two past models
    assert np.isclose(knn_predict(distances, values, 2)[0], (0.2 * 1 + 0.5 * 0.5) / 1.5)


def test_response_distance_counts_disagreements():
    memory = np.array([[1, 0, 1, 1], [0, 0, 0, 0]], float)
    query = np.array([[1, 1, 1, 1]], float)
    assert np.allclose(response_distances(memory, query), [[1.0, 2.0]])


def test_score_distance_standardises_over_memory():
    memory = np.array([[0.2, 10.0], [0.4, 30.0]])
    d = score_distances(memory, memory)
    assert np.allclose(np.diag(d), 0) and np.isclose(d[0, 1], np.sqrt(8))


def test_consensus_never_exceeds_memory_ceiling_but_discordance_can():
    rng = np.random.default_rng(1)
    memory_scores = rng.uniform(0.2, 0.6, size=40)
    memory_anchor = memory_scores + rng.normal(0, 0.02, size=40)
    dv, dq = rng.uniform(0.1, 1, (10, 40)), rng.uniform(0.1, 1, (5, 40))
    valid_truth = rng.uniform(0.2, 0.6, size=10)
    query_anchor = np.full(5, 0.95)  # far above every past model
    con = consensus_retrieval(dv, dq, memory_scores, valid_truth)
    dis = discordance_retrieval(dv, dq, memory_scores, memory_anchor, valid_truth,
                                query_anchor, valid_truth)
    assert con.query.max() <= memory_scores.max() + 1e-12
    assert dis.query.min() > memory_scores.max()


def test_k_is_chosen_on_validation_error():
    rng = np.random.default_rng(2)
    dv, dq = rng.uniform(0.1, 1, (20, 60)), rng.uniform(0.1, 1, (5, 60))
    memory = rng.uniform(size=60)
    valid_truth = knn_predict(dv, memory, 10)  # exactly the k = 10 prediction
    assert consensus_retrieval(dv, dq, memory, valid_truth).k == 10
