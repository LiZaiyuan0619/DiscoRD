"""DiscoRD: discordance retrieval for sparse evaluation of LLMs.

The method lives in a few small modules:

    distances     score distance and response distance between models
    retrieval     consensus and discordance retrieval, choice of k
    predictors    anchor mean and Ridge transfer
    fusion        simplex weights and smoothing towards equal weights
    discord       DiscoRD: build the predictors of one benchmark and fuse them

``sparseeval_mlp`` (the SparseEval-MLP predictor) needs PyTorch and is
imported on demand.  ``protocol``, ``metrics`` and ``stats`` implement the
evaluation of the paper, and ``pipelines`` runs one cell of each collection.
"""
from .discord import (CONSENSUS_PREDICTORS, DISCORDANCE_PREDICTORS, PREDICTORS,
                      DiscoRD, DiscoRDOutput, TargetData, retrieval_predictors)
from .distances import response_distances, score_distances
from .fusion import WEIGHT_RULES, fusion_weights, simplex_weights, smoothed_weights
from .metrics import ALL_METRICS, MAIN_METRICS, evaluate
from .predictors import anchor_mean, ridge_transfer
from .retrieval import K_GRID, consensus_retrieval, discordance_retrieval, knn_predict

__version__ = "0.1.0"

__all__ = [
    "PREDICTORS", "CONSENSUS_PREDICTORS", "DISCORDANCE_PREDICTORS",
    "DiscoRD", "DiscoRDOutput", "TargetData", "retrieval_predictors",
    "response_distances", "score_distances",
    "WEIGHT_RULES", "fusion_weights", "simplex_weights", "smoothed_weights",
    "ALL_METRICS", "MAIN_METRICS", "evaluate",
    "anchor_mean", "ridge_transfer",
    "K_GRID", "consensus_retrieval", "discordance_retrieval", "knn_predict",
]
