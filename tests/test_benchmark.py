import numpy as np
import pytest
from src.benchmark import ALGORITHMS, Parameters, run_algorithm


@pytest.mark.parametrize("name", ALGORITHMS)
def test_algorithm_shapes_and_timing(name):
    rng = np.random.default_rng(123)
    X = np.r_[rng.normal(loc=0, scale=.15, size=(30, 3)),
              rng.normal(loc=3, scale=.15, size=(30, 3))]
    p = Parameters(k=2, eps=.7, min_samples=3, repeats=2, warmups=0)
    labels, stats, raw = run_algorithm(X, name, p, measure_memory=False)
    assert labels.shape == (60,)
    assert stats["median_detik"] > 0
    assert stats["repetisi"] == 2
    assert len(raw) == 2
