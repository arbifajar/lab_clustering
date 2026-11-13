"""Evaluasi clustering tanpa label kebenaran (unsupervised)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray, seed: int = 42, max_silhouette: int = 1200
) -> dict:
    """DBSCAN label -1 = noise; skor internal hanya pada titik non-noise.

    Skor kosong jika <2 klaster atau >=jumlah titik; jangan paksa menjadi nol.
    Sampling silhouette boleh dilakukan demi O(n²); DB/CH pada semua non-noise.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or labels.ndim != 1 or len(X) != len(labels):
        raise ValueError("Dimensi matriks fitur dan label tidak cocok.")
    mask = labels != -1
    selected = labels[mask]
    unique, counts = np.unique(selected, return_counts=True)
    n_clusters = len(unique)
    n_valid = int(mask.sum())
    output = {
        "jumlah_data": int(len(labels)),
        "jumlah_klaster": int(n_clusters),
        "jumlah_noise": int((~mask).sum()),
        "persen_noise": float((~mask).sum() / len(labels) * 100) if len(labels) else np.nan,
        "cakupan_persen": float(mask.sum() / len(labels) * 100) if len(labels) else np.nan,
        "ukuran_klaster_min": int(counts.min()) if counts.size else 0,
        "ukuran_klaster_max": int(counts.max()) if counts.size else 0,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "n_silhouette": 0,
        "valid_internal": False,
        "catatan_evaluasi": "",
    }
    if n_clusters < 2 or n_clusters >= n_valid:
        output["catatan_evaluasi"] = "Skor internal tidak terdefinisi: perlu 2 hingga n-1 klaster non-noise."
        return output
    X_valid = X[mask]
    try:
        # Pastikan tiap klaster muncul pada sample bila sampling diterapkan.
        if n_valid > max_silhouette:
            rng = np.random.default_rng(seed)
            per_label_indices = [np.flatnonzero(selected == u) for u in unique]
            if len(per_label_indices) >= max_silhouette:
                output["catatan_evaluasi"] = "Terlalu banyak klaster untuk sampel silhouette."
            else:
                mandatory = np.array([rng.choice(ix) for ix in per_label_indices], dtype=int)
                other = np.setdiff1d(np.arange(n_valid), mandatory, assume_unique=True)
                remaining = rng.choice(other, size=max_silhouette - len(mandatory), replace=False)
                chosen = np.sort(np.concatenate([mandatory, remaining]))
                output["silhouette"] = float(silhouette_score(X_valid[chosen], selected[chosen]))
                output["n_silhouette"] = int(len(chosen))
        else:
            output["silhouette"] = float(silhouette_score(X_valid, selected))
            output["n_silhouette"] = n_valid
        output["davies_bouldin"] = float(davies_bouldin_score(X_valid, selected))
        output["calinski_harabasz"] = float(calinski_harabasz_score(X_valid, selected))
        output["valid_internal"] = bool(np.isfinite(output["davies_bouldin"]) and np.isfinite(output["calinski_harabasz"]))
        if n_valid > max_silhouette:
            output["catatan_evaluasi"] = f"Silhouette menggunakan sampel terstratifikasi {output['n_silhouette']} titik; DB dan CH seluruh data non-noise."
    except ValueError as exc:
        output["catatan_evaluasi"] = f"Sebagian metrik tidak dapat dihitung: {exc}"
    return output


def cluster_details(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Profil klaster di satuan bisnis asli (bukan skala fitur)."""
    local = rfm.copy()
    local["Klaster"] = np.where(labels == -1, "Noise", np.char.add("Klaster ", labels.astype(str)))
    result = local.groupby("Klaster", sort=True).agg(
        Pelanggan=("CustomerID", "count"),
        Recency_median=("Recency", "median"),
        Frequency_median=("Frequency", "median"),
        Monetary_median=("Monetary", "median"),
        Monetary_total=("Monetary", "sum"),
    ).reset_index()
    result["Persen_pelanggan"] = (result["Pelanggan"] / len(local) * 100).round(2)
    return result
