"""Tiga strategi frequent itemset mining eksak yang independen.

Hasil masing-masing: dict[tuple[StockCode,...], support_count].
Tuple diurutkan alfabetis dan hanya itemset dengan panjang <= max_len.
Tidak ada algoritma clustering dalam modul ini.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import ceil
from typing import Iterable, Sequence

Transaction = frozenset[str]
ItemsetCounts = dict[tuple[str, ...], int]


def min_count_for(n_invoices: int, min_support: float) -> int:
    """Ambang support count: ceil(s * n), dengan koreksi floating-point."""
    if n_invoices < 1:
        raise ValueError("Minimal satu invoice diperlukan.")
    if not 0 < min_support <= 1:
        raise ValueError("Min support harus lebih dari 0 dan maksimal 1.")
    return max(1, ceil(min_support * n_invoices - 1e-10))


def _validate(baskets: Sequence[Transaction], min_count: int, max_len: int) -> None:
    if min_count < 1:
        raise ValueError("Min count minimal 1.")
    if max_len not in (1, 2, 3):
        raise ValueError("max_len hanya mendukung 1, 2, 3 sesuai protokol slide.")
    if not baskets:
        raise ValueError("Tidak ada transaksi.")


def apriori(baskets: Sequence[Transaction], min_count: int, max_len: int = 3) -> ItemsetCounts:
    """Apriori: level-wise candidate generation, subset prune, scan setiap level."""
    _validate(baskets, min_count, max_len)
    counts1 = Counter(item for basket in baskets for item in basket)
    prev = {(item,): count for item, count in counts1.items() if count >= min_count}
    found = dict(sorted(prev.items()))
    items = set(k[0] for k in prev)
    for k in range(2, max_len + 1):
        if len(prev) < 2:
            break
        prev_keys = sorted(prev)
        prev_set = set(prev_keys)
        candidates: set[tuple[str, ...]] = set()
        for i, left in enumerate(prev_keys):
            for right in prev_keys[i + 1:]:
                # Join hanya jika prefix k-2 sama (untuk k=2 prefix kosong).
                if left[:-1] != right[:-1]:
                    if k > 2:
                        break
                    continue
                candidate = left + (right[-1],)
                if all(sub in prev_set for sub in combinations(candidate, k - 1)):
                    candidates.add(candidate)
        if not candidates:
            break
        level_count = Counter()
        # Satu invoice/keranjang dihitung paling banyak sekali per itemset.
        for basket in baskets:
            filtered = sorted(basket.intersection(items))
            if len(filtered) >= k:
                for combo in combinations(filtered, k):
                    if combo in candidates:
                        level_count[combo] += 1
        prev = {candidate: count for candidate, count in level_count.items() if count >= min_count}
        found.update(prev)
    return dict(sorted(found.items()))


class _Node:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: str | None, parent: _Node | None):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[str, _Node] = {}
        self.link: _Node | None = None


def _fp_tree(
    paths: Iterable[tuple[Iterable[str], int]],
    min_count: int,
) -> tuple[_Node, dict[str, tuple[int, _Node]]] | None:
    """Bangun FP-tree, header pointer ke node terakhir untuk tiap item."""
    paths = [(tuple(path), weight) for path, weight in paths if weight > 0]
    freq = Counter()
    for path, weight in paths:
        for item in path:
            freq[item] += weight
    freq = Counter({item: count for item, count in freq.items() if count >= min_count})
    if not freq:
        return None
    order = {item: idx for idx, item in enumerate(sorted(freq, key=lambda item: (-freq[item], item)))}
    root = _Node(None, None)
    header_tail: dict[str, _Node] = {}
    header_first: dict[str, _Node] = {}
    for path, weight in paths:
        filtered = sorted((item for item in path if item in freq), key=order.__getitem__)
        parent = root
        for item in filtered:
            if item not in parent.children:
                node = _Node(item, parent)
                parent.children[item] = node
                if item in header_tail:
                    header_tail[item].link = node
                else:
                    header_first[item] = node
                header_tail[item] = node
            parent = parent.children[item]
            parent.count += weight
    return root, {item: (freq[item], header_first[item]) for item in freq}


def fp_growth(baskets: Sequence[Transaction], min_count: int, max_len: int = 3) -> ItemsetCounts:
    """FP-Growth: FP-tree + conditional pattern base, tidak membuat kandidat Apriori."""
    _validate(baskets, min_count, max_len)
    root_header = _fp_tree(((basket, 1) for basket in baskets), min_count)
    if root_header is None:
        return {}
    result: ItemsetCounts = {}

    def mine(header: dict[str, tuple[int, _Node]], suffix: tuple[str, ...]) -> None:
        # Reverse support order adalah aturan mining FP-Growth standar.
        for item in sorted(header, key=lambda i: (header[i][0], i)):
            support, head = header[item]
            pattern = tuple(sorted(suffix + (item,)))
            result[pattern] = support
            if len(pattern) == max_len:
                continue
            conditional: list[tuple[tuple[str, ...], int]] = []
            node: _Node | None = head
            while node is not None:
                path: list[str] = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional.append((tuple(path), node.count))
                node = node.link
            child = _fp_tree(conditional, min_count)
            if child is not None:
                mine(child[1], pattern)

    mine(root_header[1], tuple())
    return dict(sorted(result.items()))


def eclat(baskets: Sequence[Transaction], min_count: int, max_len: int = 3) -> ItemsetCounts:
    """Eclat: tidset vertikal bitset Python; perluasan DFS melalui irisan."""
    _validate(baskets, min_count, max_len)
    bitsets: dict[str, int] = defaultdict(int)
    for tid, basket in enumerate(baskets):
        flag = 1 << tid
        for item in basket:
            bitsets[item] |= flag
    items = [(item, mask) for item, mask in sorted(bitsets.items()) if mask.bit_count() >= min_count]
    result: ItemsetCounts = {}

    def dfs(prefix: tuple[str, ...], candidates: list[tuple[str, int]]) -> None:
        for idx, (item, tidset) in enumerate(candidates):
            itemset = prefix + (item,)
            result[itemset] = tidset.bit_count()
            if len(itemset) == max_len:
                continue
            new_candidates = []
            for other_item, other_mask in candidates[idx + 1:]:
                intersection = tidset & other_mask
                if intersection.bit_count() >= min_count:
                    new_candidates.append((other_item, intersection))
            if new_candidates:
                dfs(itemset, new_candidates)

    dfs((), items)
    return dict(sorted(result.items()))


ALGORITHMS = {"Apriori": apriori, "FP-Growth": fp_growth, "Eclat": eclat}


def mine(name: str, baskets: Sequence[Transaction], min_count: int, max_len: int = 3) -> ItemsetCounts:
    if name not in ALGORITHMS:
        raise ValueError(f"Algoritma tidak dikenal: {name}")
    return ALGORITHMS[name](baskets, min_count, max_len)
