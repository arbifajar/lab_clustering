"""Load basket, susun subset bersarang dan tampilkan audit dataset."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_baskets(path: Path | None = None) -> tuple[tuple[str, ...], tuple[frozenset[str], ...]]:
    source = path or ROOT / "data" / "transactions.csv.gz"
    names = []
    baskets = []
    with gzip.open(source, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            invoice = row["InvoiceNo"].strip()
            items = json.loads(row["ItemsJson"])
            if not invoice or not items or any(not x or not isinstance(x, str) for x in items):
                raise ValueError("Invoice atau item kosong ditemukan pada berkas transaksi.")
            names.append(invoice)
            baskets.append(frozenset(items))
    if len(names) != len(set(names)):
        raise ValueError("InvoiceNo ganda dalam CSV pra-pemrosesan.")
    return tuple(names), tuple(baskets)


def load_audit(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "data" / "audit.json").read_text(encoding="utf-8"))


def load_products(path: Path | None = None) -> dict[str, str]:
    file = path or ROOT / "data" / "products.csv"
    with file.open(encoding="utf-8", newline="") as fh:
        return {row["StockCode"]: row["Description"] for row in csv.DictReader(fh)}


def nested_indices(n: int, seed: int, fraction: float) -> np.ndarray:
    """Sampel bersarang dari random.Random(seed).shuffle, cocok dengan PPT acuan."""
    if n < 1 or not 0 < fraction <= 1:
        raise ValueError("Proporsi data tidak sah.")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return np.asarray(sorted(indices[:max(1, int(n * fraction))]), dtype=np.int64)


def baskets_subset(baskets: tuple[frozenset[str], ...], seed: int, fraction: float) -> list[frozenset[str]]:
    return [baskets[int(i)] for i in nested_indices(len(baskets), seed, fraction)]


def counts_frame(counts: dict[tuple[str, ...], int], total: int, product_names: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for itemset, count in counts.items():
        rows.append({
            "Panjang": len(itemset), "Kode barang": ", ".join(itemset),
            "Nama barang": " | ".join((product_names or {}).get(i, i) for i in itemset),
            "Support count": count, "Support (%)": round(100 * count / total, 4),
        })
    return pd.DataFrame(rows, columns=["Panjang", "Kode barang", "Nama barang", "Support count", "Support (%)"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()
