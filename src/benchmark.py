"""Pengujian benchmark; waktu median ulangan dan memori satu run terpisah.

Periode waktu: materialisasi struktur data internal + proses mining.
Di luar waktu: pembacaan file, filter awal, pemilihan subset, plotting, ekspor, aturan.
"""
from __future__ import annotations

import gc
import os
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .data import baskets_subset
from .mining import ALGORITHMS, ItemsetCounts, min_count_for


@dataclass
class BenchmarkResult:
    summary: pd.DataFrame
    durations: pd.DataFrame
    memory: pd.DataFrame
    counts_by_scenario: dict[tuple[int, int], ItemsetCounts]
    metadata: dict


def _stats(durations: list[float]) -> dict:
    arr = np.asarray(durations, dtype=float)
    avg = float(arr.mean())
    return {
        "Median (s)": float(np.median(arr)),
        "Rata-rata (s)": avg,
        "SD (s)": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        "Q1 (s)": float(np.quantile(arr, .25)),
        "Q3 (s)": float(np.quantile(arr, .75)),
        "CV (%)": float(100 * np.std(arr, ddof=1) / avg) if len(arr) > 1 and avg > 0 else np.nan,
    }


def run_benchmark(
    baskets: tuple[frozenset[str], ...],
    fractions: Iterable[int] = (25, 50, 100),
    supports: Iterable[int] = (3, 5, 8),
    repeats: int = 3,
    seed: int = 42,
    max_len: int = 3,
    measure_memory: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> BenchmarkResult:
    """Semua tiga algoritma diuji pada setiap kombinasi fraksi-support.

    fractions/supports adalah persen bulat. Sah jika 0<fraksi<=100 dan 1<=support<=20.
    Digunakan `random.Random(seed).shuffle` untuk sampel bersarang mengikuti slide.
    """
    fractions, supports = sorted(set(map(int, fractions))), sorted(set(map(int, supports)))
    if not fractions or any(not 0 < x <= 100 for x in fractions):
        raise ValueError("Proporsi pengujian harus antara 1% dan 100%.")
    if not supports or any(not 1 <= x <= 20 for x in supports):
        raise ValueError("Support harus 1–20%; protokol utama slide adalah 3, 5, 8%.")
    if not 1 <= repeats <= 10:
        raise ValueError("Pengulangan harus 1–10.")
    if max_len not in (1, 2, 3):
        raise ValueError("Batas panjang itemset hanya 1, 2 atau 3.")
    if not baskets:
        raise ValueError("Tidak terdapat invoice untuk dievaluasi.")
    summary_rows, duration_rows, memory_rows = [], [], []
    canonical_by_scenario = {}
    total_steps = len(fractions) * len(supports) * len(ALGORITHMS)
    step = 0
    for fraction in fractions:
        selected = baskets_subset(baskets, seed, fraction / 100)
        total = len(selected)
        for support in supports:
            count_limit = min_count_for(total, support / 100)
            expected: ItemsetCounts | None = None
            for algorithm_name, algorithm in ALGORITHMS.items():
                step += 1
                if progress:
                    progress(step - 1, total_steps, f"{fraction}% data / support {support}% / {algorithm_name}")
                measured = []
                last_result: ItemsetCounts | None = None
                for run in range(1, repeats + 1):
                    gc.collect()
                    start = time.perf_counter()
                    current = algorithm(selected, count_limit, max_len)
                    measured_seconds = time.perf_counter() - start
                    measured.append(measured_seconds)
                    if last_result is not None and last_result != current:
                        raise AssertionError(f"Hasil tidak deterministik pada {algorithm_name}.")
                    last_result = current
                    duration_rows.append({
                        "Proporsi (%)": fraction, "Support minimum (%)": support,
                        "Jumlah invoice": total, "Min count": count_limit,
                        "Algoritma": algorithm_name, "Ulangan": run,
                        "Waktu (s)": measured_seconds,
                    })
                assert last_result is not None
                if expected is None:
                    expected = last_result
                    canonical_by_scenario[(fraction, support)] = expected
                if last_result != expected:
                    raise AssertionError(
                        f"VALIDASI GAGAL pada {fraction}%/{support}%: hasil {algorithm_name} "
                        "tidak identik dengan Apriori (itemset dan support count)."
                    )
                py_peak_mib = np.nan
                delta_rss_mib = np.nan
                if measure_memory:
                    # Tidak digabung dengan pengulangan waktu: tracemalloc menambah overhead.
                    gc.collect()
                    try:
                        import psutil
                        proc = psutil.Process(os.getpid())
                        before = proc.memory_info().rss
                    except (ImportError, OSError):
                        proc, before = None, 0
                    tracemalloc.start()
                    try:
                        mem_result = algorithm(selected, count_limit, max_len)
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    if mem_result != expected:
                        raise AssertionError(f"Hasil run memori {algorithm_name} tidak konsisten.")
                    py_peak_mib = peak / 1024**2
                    if proc is not None:
                        delta_rss_mib = (proc.memory_info().rss - before) / 1024**2
                    del mem_result
                stats = _stats(measured)
                row = {
                    "Proporsi (%)": fraction, "Support minimum (%)": support,
                    "Jumlah invoice": total, "Min count": count_limit,
                    "Algoritma": algorithm_name,
                    "Itemset frequent": len(last_result),
                    "Singleton": sum(len(x) == 1 for x in last_result),
                    "Pasangan": sum(len(x) == 2 for x in last_result),
                    "Tripel": sum(len(x) == 3 for x in last_result),
                    "Kesetaraan output": "COCOK",
                    "Ulangan waktu": repeats,
                    **stats,
                    "Peak Python (MiB)": py_peak_mib,
                    "Delta RSS setelah (MiB)": delta_rss_mib,
                }
                summary_rows.append(row)
                memory_rows.append({
                    "Proporsi (%)": fraction, "Support minimum (%)": support,
                    "Jumlah invoice": total, "Algoritma": algorithm_name,
                    "Peak tracemalloc (MiB)": py_peak_mib,
                    "Delta RSS akhir (MiB)": delta_rss_mib,
                    "Run memori independen": 1 if measure_memory else 0,
                })
    if progress:
        progress(total_steps, total_steps, "Selesai: keluaran ketiga algoritma cocok pada seluruh skenario.")
    metadata = {
        "nama_eksperimen": "Kelompok 2 / Topik 3 - Association Rule Mining",
        "diambil_pada_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "jumlah_invoice_sumber": len(baskets),
        "proporsi_persen": fractions, "support_persen": supports,
        "min_support_count": "ceil(support_fraction * jumlah_invoice)",
        "seed": seed, "metode_sampling": "random.Random(seed).shuffle(invoice_urut); subset bersarang",
        "panjang_itemset_maksimum": max_len, "ulangan": repeats,
        "memori": "Satu run terpisah memakai peak tracemalloc; delta RSS bukan puncak RAM dan bisa negatif.",
        "cakupan_waktu": "Membangun struktur internal + mining; tanpa file I/O, cleaning, sampling, aturan atau ekspor.",
        "catatan": "Pengukuran deskriptif bergantung mesin, proses, dan implementasi; bukan Big-O.",
    }
    return BenchmarkResult(pd.DataFrame(summary_rows), pd.DataFrame(duration_rows),
                           pd.DataFrame(memory_rows), canonical_by_scenario, metadata)
