"""Per-question score matrices: the common input format of every collection.

The preparation scripts in ``data/`` write one ``.npz`` file per benchmark with

    responses     (n_models, n_questions) per-question scores in [0, 1]
    model_names   (n_models,) model identifiers
    item_ids      (n_questions,) question identifiers
    item_weights  (n_questions,) weights of the questions in the full score, summing to one

The full-benchmark score of a model is ``responses @ item_weights``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResponseMatrix:
    name: str
    responses: np.ndarray
    model_names: np.ndarray
    item_ids: np.ndarray
    item_weights: np.ndarray

    @property
    def n_models(self) -> int:
        return int(self.responses.shape[0])

    @property
    def n_questions(self) -> int:
        return int(self.responses.shape[1])

    def scores(self, rows=None) -> np.ndarray:
        """Full-benchmark scores of all models, or of the given rows."""
        y = self.responses if rows is None else self.responses[rows]
        return y @ self.item_weights

    def subset_models(self, names) -> "ResponseMatrix":
        """The same benchmark restricted to (and ordered by) the given model names."""
        position = {m: i for i, m in enumerate(self.model_names)}
        rows = np.asarray([position[m] for m in names], int)
        return ResponseMatrix(self.name, self.responses[rows], self.model_names[rows],
                              self.item_ids, self.item_weights)


def validate(matrix: ResponseMatrix, *, binary: bool | None = None) -> ResponseMatrix:
    y, w = matrix.responses, matrix.item_weights
    if y.ndim != 2 or y.shape != (len(matrix.model_names), len(matrix.item_ids)):
        raise ValueError(f"{matrix.name}: matrix shape {y.shape} does not match its ids")
    if not np.isfinite(y).all() or y.min() < 0 or y.max() > 1:
        raise ValueError(f"{matrix.name}: scores must be finite and in [0, 1]")
    if w.shape != (y.shape[1],) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError(f"{matrix.name}: item weights must be non-negative and sum to one")
    if len(np.unique(matrix.model_names)) != len(matrix.model_names):
        raise ValueError(f"{matrix.name}: duplicate model names")
    if binary and not np.isin(y, (0, 1)).all():
        raise ValueError(f"{matrix.name}: expected binary scores")
    return matrix


def save_matrix(path, matrix: ResponseMatrix) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate(matrix)
    np.savez_compressed(path, responses=matrix.responses,
                        model_names=np.asarray(matrix.model_names, dtype=str),
                        item_ids=np.asarray(matrix.item_ids, dtype=str),
                        item_weights=np.asarray(matrix.item_weights, float),
                        name=np.asarray(matrix.name))


def load_matrix(path) -> ResponseMatrix:
    with np.load(path, allow_pickle=False) as z:
        name = str(z["name"]) if "name" in z.files else Path(path).stem
        matrix = ResponseMatrix(name, z["responses"].astype(float),
                                z["model_names"].astype(str), z["item_ids"].astype(str),
                                z["item_weights"].astype(float))
    return validate(matrix)


def load_collection(directory, names=None) -> dict[str, ResponseMatrix]:
    """All benchmarks of a collection (``<directory>/<benchmark>.npz``), sorted by name."""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if names is not None:
        wanted = set(names)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise FileNotFoundError(f"no .npz matrices under {directory}; run the script in data/ first")
    return {f.stem: load_matrix(f) for f in files}
