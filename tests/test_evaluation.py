import numpy as np
import pandas as pd
from src.evaluation import evaluate_clustering, cluster_details


def test_noise_excluded_from_metrics():
    X = np.array([[0, 0], [0.05, 0.1], [0.1, 0], [4, 4], [4.1, 4], [4, 4.1], [99, 99]])
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    result = evaluate_clustering(X, labels)
    assert result["jumlah_klaster"] == 2
    assert result["jumlah_noise"] == 1
    assert result["n_silhouette"] == 6
    assert result["silhouette"] > .90
    assert result["valid_internal"]


def test_invalid_partition_not_forced_zero():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    for labels in ([-1] * 4, [0] * 4, [0, 1, 2, 3]):
        r = evaluate_clustering(X, labels)
        assert np.isnan(r["silhouette"])
        assert np.isnan(r["davies_bouldin"])


def test_profiles_include_noise():
    X = pd.DataFrame({"CustomerID": ["1", "2", "3", "4"], "Recency": [1, 2, 3, 4],
                      "Frequency": [1, 2, 3, 4], "Monetary": [10, 20, 30, 40]})
    r = cluster_details(X, np.array([0, 0, 1, -1]))
    assert set(r.Klaster) == {"Klaster 0", "Klaster 1", "Noise"}
    assert r.Pelanggan.sum() == 4
