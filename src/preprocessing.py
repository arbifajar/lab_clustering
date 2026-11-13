"""Pembersihan UCI Online Retail dan pembuatan fitur RFM per pelanggan.

Data asli memiliki granularitas satu baris barang, BUKAN satu pelanggan.
Tidak termasuk pembatalan/retur; Monetary = nilai penjualan positif bruto.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

REQUIRED_COLUMNS = (
    "InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID",
)
RFM_COLUMNS = ("Recency", "Frequency", "Monetary")


def file_sha256(source: bytes | Path) -> str:
    if isinstance(source, bytes):
        return hashlib.sha256(source).hexdigest()
    digest = hashlib.sha256()
    with Path(source).open("rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_online_retail(source: bytes | Path) -> pd.DataFrame:
    """Baca hanya kolom yang dipakai. Bisa >30 detik pada Excel 541 ribu baris."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    if isinstance(source, Path) or (isinstance(source, str) and str(source).endswith(".xlsx")):
        return pd.read_excel(source, engine="openpyxl", usecols=lambda c: c in REQUIRED_COLUMNS)
    return pd.read_excel(source, engine="openpyxl", usecols=lambda c: c in REQUIRED_COLUMNS)


def build_customer_rfm(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Kembalikan RFM raw tanpa transformasi dan laporan audit pembersihan.

    One invoice -> dapat mengandung banyak baris, Frequency dihitung per invoice
    unik per CustomerID. Hari referensi adalah tanggal transaksi terakhir + 1 hari.
    """
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")

    total = len(raw)
    df = raw.loc[:, REQUIRED_COLUMNS].copy()
    invoice = df["InvoiceNo"].astype("string").str.strip()
    stock = df["StockCode"].astype("string").str.strip()
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    customer = pd.to_numeric(df["CustomerID"], errors="coerce")
    valid_base = (
        invoice.notna() & invoice.ne("") & ~invoice.str.upper().str.startswith("C", na=False)
        & stock.notna() & stock.ne("")
        & quantity.gt(0) & price.gt(0)
    ).fillna(False)
    valid_rfm = valid_base & customer.notna() & dates.notna()
    df = pd.DataFrame({
        "CustomerID": customer[valid_rfm].astype("int64").astype("string"),
        "InvoiceNo": invoice[valid_rfm].astype("string"),
        "InvoiceDate": dates[valid_rfm],
        "Sales": (quantity[valid_rfm] * price[valid_rfm]).astype(float),
    })
    if df.empty:
        raise ValueError("Tidak ada transaksi valid dengan CustomerID untuk segmentasi RFM.")
    reference = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    stats = df.groupby("CustomerID", sort=True).agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum"),
    ).reset_index()
    stats["Recency"] = (reference - stats["LastPurchase"].dt.normalize()).dt.days.astype(int)
    stats["Monetary"] = stats["Monetary"].round(2)
    stats = stats.loc[:, ["CustomerID", *RFM_COLUMNS]].sort_values("CustomerID").reset_index(drop=True)
    if (stats[["Recency", "Frequency", "Monetary"]] <= 0).any().any():
        raise ValueError("Nilai RFM di luar rentang positif; periksa input.")
    audit = {
        "baris_awal": int(total),
        "baris_valid_aturan_asosiasi": int(valid_base.sum()),
        "baris_valid_rfm": int(valid_rfm.sum()),
        "baris_tanpa_customer_id_setelah_filter_dasar": int((valid_base & customer.isna()).sum()),
        "jumlah_invoice_valid_dasar": int(invoice[valid_base].nunique()),
        "jumlah_invoice_valid_rfm": int(df["InvoiceNo"].nunique()),
        "jumlah_pelanggan": int(len(stats)),
        "tanggal_referensi": str(reference.date()),
        "tanggal_minimum_rfm": str(df["InvoiceDate"].min().date()),
        "tanggal_maksimum_rfm": str(df["InvoiceDate"].max().date()),
        "jumlah_invoice_dibatalkan_baris": int(invoice.str.upper().str.startswith("C", na=False).sum()),
        "baris_quantity_nonpositif": int(quantity.le(0).sum()),
        "baris_harga_nonpositif": int(price.le(0).sum()),
        "catatan": "Filter bersifat gabungan dan hitungan per kategori dapat bertumpang tindih.",
    }
    return stats, audit


def validate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    if not set(["CustomerID", *RFM_COLUMNS]).issubset(df.columns):
        raise ValueError("CSV RFM harus memuat CustomerID, Recency, Frequency, Monetary.")
    out = df[["CustomerID", *RFM_COLUMNS]].copy()
    out["CustomerID"] = out["CustomerID"].astype("string")
    for col in RFM_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out.isna().any().any() or (out[list(RFM_COLUMNS)] <= 0).any().any():
        raise ValueError("RFM mengandung nilai kosong, tidak numerik, atau nonpositif.")
    if out["CustomerID"].duplicated().any():
        raise ValueError("CustomerID harus unik; satu baris mewakili satu pelanggan.")
    if len(out) < 4:
        raise ValueError("Minimal empat pelanggan diperlukan untuk eksperimen.")
    return out.reset_index(drop=True)


def nested_subsample_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    """Subset bersarang; urutan indeks hasil diurutkan agar plot stabil."""
    if not 0 < fraction <= 1:
        raise ValueError("Proporsi harus di rentang (0, 1].")
    size = min(n, max(1, int(np.floor(n * fraction))))
    return np.sort(np.random.default_rng(seed).permutation(n)[:size])


def make_features(
    rfm: pd.DataFrame, scaler_type: str = "StandardScaler", winsor_quantile: float = 1.0
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fitur log1p(RFM) lalu scaling; fit dilakukan pada subset yang diuji.

    Kebocoran informasi antarskenario dihindari: scaler hanya fit pada subset.
    winsor_quantile<1 memotong nilai ekstrim pada batas kuantil pada subset.
    """
    if not 0.90 <= winsor_quantile <= 1.0:
        raise ValueError("Kuantil batas harus antara 0,90 dan 1,00.")
    input_data = validate_rfm(rfm)
    numeric = input_data.loc[:, list(RFM_COLUMNS)].astype(float).copy()
    if winsor_quantile < 1:
        for col in RFM_COLUMNS:
            numeric[col] = numeric[col].clip(upper=numeric[col].quantile(winsor_quantile))
    transformed = np.log1p(numeric)
    scaler = StandardScaler() if scaler_type == "StandardScaler" else RobustScaler()
    if scaler_type not in {"StandardScaler", "RobustScaler"}:
        raise ValueError("Scaler tidak dikenal.")
    X = scaler.fit_transform(transformed).astype(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Hasil transformasi tidak seluruhnya terbatas dan numerik.")
    return X, transformed
