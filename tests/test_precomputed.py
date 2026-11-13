from pathlib import Path
import json
import pandas as pd
from src.preprocessing import validate_rfm

ROOT = Path(__file__).resolve().parents[1]


def test_precomputed_matches_audit():
    frame = validate_rfm(pd.read_csv(ROOT / "data/customer_rfm.csv", dtype={"CustomerID": str}))
    audit = json.loads((ROOT / "data/audit.json").read_text(encoding="utf-8"))
    assert len(frame) == audit["jumlah_pelanggan"]
    assert (frame[["Recency", "Frequency", "Monetary"]] > 0).all().all()
