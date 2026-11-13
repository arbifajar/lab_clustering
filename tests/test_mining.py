from collections import Counter
from itertools import combinations
from math import ceil

import numpy as np
import pytest

from src.mining import ALGORITHMS, min_count_for


def oracle(baskets, min_count, L):
    all_items = sorted(set().union(*baskets))
    result = {}
    for n in range(1, L + 1):
        for combo in combinations(all_items, n):
            count = sum(set(combo).issubset(b) for b in baskets)
            if count >= min_count:
                result[combo] = count
    return result


@pytest.mark.parametrize("seed", list(range(14)))
def test_random_small_oracle(seed):
    gen = np.random.default_rng(seed)
    baskets = [frozenset(ch for ch in "ABCDE" if gen.random() < .4) for _ in range(8)]
    for k in (1, 2, 4):
        for L in (1, 2, 3):
            expected = oracle(baskets, k, L)
            for name, algorithm in ALGORITHMS.items():
                assert algorithm(baskets, k, L) == expected, (name, seed, k, L)


def test_known_support_and_rules():
    baskets = [frozenset({"a", "b"}), frozenset({"a", "b", "c"}), frozenset({"b", "c"})]
    assert all(fn(baskets, 2) == {("a",): 2, ("b",): 3, ("b", "c"): 2,
                                     ("a", "b"): 2, ("c",): 2} for fn in ALGORITHMS.values())
    assert min_count_for(19960, .03) == 599
    assert min_count_for(4990, .05) == 250
    assert min_count_for(9980, .05) == 499


def test_duplicate_stock_code_one_presence():
    baskets = [frozenset(["a", "a", "b"]), frozenset(["a"])]
    assert all(fn(baskets, 1)[("a",)] == 2 for fn in ALGORITHMS.values())
