import numpy as np
import pandas as pd
import pytest
from src.preprocessing import build_customer_rfm, make_features, nested_subsample_indices, validate_rfm


def example_retail():
    return pd.DataFrame({
        "InvoiceNo": ["100", "100", "101", "C102", "103", "104", "105", "106", "107"],
        "StockCode": ["A", "B", "B", "A", "C", "D", "", "K", "L"],
        "Quantity": [2, 1, 3, -1, 1, 1, 1, 1, 1],
        "InvoiceDate": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-05", "2021-01-02", "2021-01-05", "2021-01-08", "2021-01-09", "2021-01-09", "2021-01-10"]),
        "UnitPrice": [10, 5, 2, 1, 0, 1, 4, 2, 3],
        "CustomerID": [1, 1, 1, 1, 2, np.nan, 3, 2, 3],
    })


def test_rfm_distinct_invoice_and_filters():
    df, audit = build_customer_rfm(example_retail())
    assert audit["baris_awal"] == 9
    assert audit["baris_valid_aturan_asosiasi"] == 6
    assert audit["baris_valid_rfm"] == 5
    assert len(df) == 3
    row1 = df[df.CustomerID == "1"].iloc[0]
    assert row1.Frequency == 2   # dua baris dari invoice 100 -> tetap satu
    assert row1.Monetary == 31  # 20+5+6
    assert row1.Recency == 6      # referensi 2021-01-11; pembelian terakhir 5 Jan
    row2 = df[df.CustomerID == "2"].iloc[0]
    assert row2.Frequency == 1
    assert row2.Recency == 2


def test_scaling_and_nested_sampling():
    frame = pd.DataFrame({"CustomerID": [str(x) for x in range(20)],
                          "Recency": np.arange(1, 21),
                          "Frequency": np.arange(1, 21),
                          "Monetary": np.arange(1, 21) * 3})
    X, transformed = make_features(frame, "RobustScaler", 0.99)
    assert X.shape == (20, 3)
    assert np.isfinite(X).all()
    a = nested_subsample_indices(20, 0.25, 42)
    b = nested_subsample_indices(20, 0.50, 42)
    assert set(a).issubset(set(b))


def test_invalid_rfm_rejected():
    with pytest.raises(ValueError):
        validate_rfm(pd.DataFrame({"CustomerID": [1, 1, 2, 3],
                                   "Recency": [2, 2, 3, 4],
                                   "Frequency": [1] * 4,
                                   "Monetary": [2] * 4}))
