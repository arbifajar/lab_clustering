"""Aturan asosiasi berasal dari frequent itemset dan support count bersama."""
from __future__ import annotations

from itertools import combinations
from typing import Mapping

import pandas as pd


def make_rules(
    counts: Mapping[tuple[str, ...], int],
    n_invoices: int,
    min_confidence: float = .30,
    pair_only: bool = True,
    min_lift: float = 0.0,
    products: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Hitung A->B: conf=count(A∪B)/count(A); lift=conf/support(B)."""
    if n_invoices < 1 or not 0 <= min_confidence <= 1 or min_lift < 0:
        raise ValueError("Parameter aturan tidak sah.")
    products = products or {}
    rows = []
    for itemset, joint in counts.items():
        if len(itemset) < 2 or (pair_only and len(itemset) != 2):
            continue
        for size in range(1, len(itemset)):
            for left in combinations(itemset, size):
                left_set = set(left)
                right = tuple(x for x in itemset if x not in left_set)
                a = counts.get(left)
                b = counts.get(right)
                if a is None or b is None or a == 0 or b == 0:
                    raise ValueError("Subset dari frequent itemset tidak tersedia: data mining tidak konsisten.")
                confidence = joint / a
                lift = confidence / (b / n_invoices)
                if confidence + 1e-12 < min_confidence or lift + 1e-12 < min_lift:
                    continue
                rows.append({
                    "Antecedent": ", ".join(left),
                    "Consequent": ", ".join(right),
                    "Nama antecedent": " | ".join(products.get(i, i) for i in left),
                    "Nama consequent": " | ".join(products.get(i, i) for i in right),
                    "Count A∩B": joint,
                    "Support (%)": 100 * joint / n_invoices,
                    "Confidence (%)": 100 * confidence,
                    "Lift": lift,
                })
    columns = ["Antecedent", "Consequent", "Nama antecedent", "Nama consequent",
               "Count A∩B", "Support (%)", "Confidence (%)", "Lift"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Lift", "Support (%)", "Antecedent", "Consequent"],
        ascending=[False, False, True, True], kind="stable",
    ).reset_index(drop=True)
