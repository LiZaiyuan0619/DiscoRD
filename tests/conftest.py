import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def synthetic():
    """Binary answers of 120 models on 300 questions from a 1PL model."""
    rng = np.random.default_rng(0)
    ability = rng.normal(size=120)
    easiness = rng.normal(size=300)
    p = 1 / (1 + np.exp(-(ability[:, None] + easiness[None, :])))
    return (rng.random(p.shape) < p).astype(float)
