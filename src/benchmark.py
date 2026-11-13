"""Benchmark tiga algoritma; semua dilatih pada fitur yang sama.

- Waktu: perf_counter dan model baru setiap pengulangan, tanpa preprocessing.
- Memori: tracemalloc peak (alokasi Python), dan sampel peak RSS melalui psutil.
  Keduanya dicatat terpisah dan tidak dianggap sebagai memori native eksak.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import platform
import threading
import time
import tracemalloc

import numpy as np
import psutil
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans


ALGORITHMS = ("K-Means", "DBSCAN", "Agglomerative")

@dataclass(frozen=True)
class Parameters:
    k: int = 4
    eps: float = 0.50
    min_samples: int = 8
    seed: int = 42
    repeats: int = 3
    warmups: int = 0


def new_model(name: str, p: Parameters):
    if name == "K-Means":
        return KMeans(n_clusters=p.k, n_init=10, random_state=p.seed)
    if name == "DBSCAN":
        return DBSCAN(eps=p.eps, min_samples=p.min_samples, algorithm="auto", n_jobs=1)
    if name == "Agglomerative":
        return AgglomerativeClustering(n_clusters=p.k, linkage="ward")
    raise ValueError(f"Algoritma tidak dikenal: {name}")


def _fit_predict(X: np.ndarray, name: str, p: Parameters) -> np.ndarray:
    model = new_model(name, p)
    return np.asarray(model.fit_predict(X), dtype=int)


def memory_one_run(X: np.ndarray, name: str, p: Parameters, poll_interval: float = 0.015) -> dict:
    """Pengukuran RAM satu fit terpisah dari run timing, bukan replikasi memori."""
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    readings = [baseline]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                readings.append(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop.wait(poll_interval)

    tracer = threading.Thread(target=poll, daemon=True)
    tracemalloc.start()
    tracer.start()
    try:
        _fit_predict(X, name, p)
        readings.append(process.memory_info().rss)
    finally:
        stop.set()
        tracer.join(timeout=1.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "python_peak_mib": float(peak / (1024**2)),
        "rss_awal_mib": float(baseline / (1024**2)),
        "rss_peak_sampel_mib": float(max(readings) / (1024**2)),
        "rss_delta_peak_mib": float(max(0, max(readings) - baseline) / (1024**2)),
    }


def run_algorithm(
    X: np.ndarray, name: str, p: Parameters, measure_memory: bool = True
) -> tuple[np.ndarray, dict, list[dict]]:
    if len(X) < 3 or (name != "DBSCAN" and (p.k < 2 or p.k >= len(X))):
        raise ValueError("K-Means/Agglomerative membutuhkan 2 <= k < jumlah data; data minimal 3.")
    if p.repeats < 1:
        raise ValueError("Jumlah pengulangan minimal satu.")
    for _ in range(p.warmups):
        _fit_predict(X, name, p)
    durations = []
    final_labels = None
    for repeat in range(1, p.repeats + 1):
        start = time.perf_counter()
        labels = _fit_predict(X, name, p)
        seconds = time.perf_counter() - start
        durations.append(seconds)
        final_labels = labels
    arr = np.asarray(durations)
    output = {
        "algorithm": name,
        "median_detik": float(np.median(arr)),
        "mean_detik": float(arr.mean()),
        "std_detik": float(arr.std(ddof=1)) if len(arr) >= 2 else np.nan,
        "q1_detik": float(np.quantile(arr, 0.25)),
        "q3_detik": float(np.quantile(arr, 0.75)),
        "cv_persen": float(arr.std(ddof=1) / arr.mean() * 100) if len(arr) >= 2 and arr.mean() > 0 else np.nan,
        "repetisi": len(arr),
    }
    if measure_memory:
        output.update(memory_one_run(X, name, p))
    else:
        output.update({"python_peak_mib": np.nan, "rss_awal_mib": np.nan,
                       "rss_peak_sampel_mib": np.nan, "rss_delta_peak_mib": np.nan})
    raw = [{"algorithm": name, "ulangan": i, "detik": float(t)} for i, t in enumerate(durations, 1)]
    return final_labels, output, raw


def environment_info() -> dict:
    import numpy
    import pandas
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": SKLEARN_VERSION,
        "cpu_logis": psutil.cpu_count(logical=True),
        "ram_total_gib": round(psutil.virtual_memory().total / (1024**3), 2),
    }
