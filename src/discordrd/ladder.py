"""Corrections built up from no memory (the upper block of the ablation table).

Each estimate starts from the candidate's anchor mean g_c and adds a
correction learned from the memory's local corrections delta_j = y_j - g_j:

    Anchor mean (no memory)       g_c
    Global correction             g_c + mean_j delta_j
    Stratified correction, Q      g_c + mean of delta_j over the past models whose
                                  anchor mean falls in the same of Q quantile bands as g_c

Discordance retrieval replaces the bands by the k nearest past models under a
distance between models (``retrieval.py``).
"""
from __future__ import annotations

import numpy as np

from .discord import TargetData

STRATA = (2, 4, 8, 16)


def global_correction(memory_scores, memory_anchor_means, anchor_means) -> np.ndarray:
    delta = np.asarray(memory_scores, float) - np.asarray(memory_anchor_means, float)
    return np.asarray(anchor_means, float) + float(np.mean(delta))


def stratified_correction(memory_scores, memory_anchor_means, anchor_means, q: int) -> np.ndarray:
    pool_g = np.asarray(memory_anchor_means, float)
    edges = np.quantile(pool_g, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    delta = np.asarray(memory_scores, float) - pool_g
    band_means = []
    for band in range(q):
        inside = (pool_g >= edges[band]) & (pool_g < edges[band + 1])
        band_means.append(float(delta[inside].mean()) if inside.any() else 0.0)
    g = np.asarray(anchor_means, float)
    band_idx = np.clip(np.digitize(g, edges) - 1, 0, q - 1)
    return g + np.asarray(band_means)[band_idx]


def ladder_estimates(data: TargetData) -> dict[str, np.ndarray]:
    """The no-memory ladder for the candidates of one target benchmark."""
    py, pg, qg = data.memory_scores, data.memory_anchor_means, data.query_anchor_means
    out = {"Anchor mean (no memory)": np.asarray(qg, float),
           "Global correction": global_correction(py, pg, qg)}
    for q in STRATA:
        out[f"Stratified correction, Q={q}"] = stratified_correction(py, pg, qg, q)
    return out
