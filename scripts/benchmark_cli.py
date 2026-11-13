"""Menjalankan benchmark clustering tanpa browser (opsional).

Contoh:
 PYTHONPATH=. python scripts/benchmark_cli.py --limit 600 --repeats 2 --memory
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.benchmark import ALGORITHMS, Parameters, environment_info, run_algorithm
from src.evaluation import evaluate_clustering, cluster_details
from src.preprocessing import make_features, validate_rfm


def main():
    ap = argparse.ArgumentParser(description="Benchmark clustering pelanggan Online Retail")
    ap.add_argument("--limit", type=int, default=1200, help="Maksimum 3000 pelanggan")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ratios", type=int, nargs="+", default=[25, 50, 100], choices=[25, 50, 100])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eps", type=float, default=.50)
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--scaler", choices=["StandardScaler", "RobustScaler"], default="StandardScaler")
    ap.add_argument("--winsor", type=float, default=1.0, help="Kuantil 0.90 sampai 1.0")
    ap.add_argument("--memory", action="store_true", help="Ukur memori satu fit tambahan per kondisi")
    ap.add_argument("--output", type=Path, default=ROOT / "hasil")
    args = ap.parse_args()
    if not 2 <= args.k or not 2 <= args.min_samples:
        ap.error("k dan min-samples minimal dua")
    if not 0.9 <= args.winsor <= 1.0:
        ap.error("--winsor harus 0.9–1.0")
    df = validate_rfm(pd.read_csv(ROOT / "data/customer_rfm.csv", dtype={"CustomerID": str}))
    limit = min(args.limit, 3000, len(df))
    if limit < 50 or args.k >= int(limit * min(args.ratios) / 100):
        ap.error("Batasi data minimal 50 dan pastikan k < jumlah data pada skenario terkecil")
    params = Parameters(k=args.k, min_samples=args.min_samples, eps=args.eps,
                        repeats=args.repeats, seed=args.seed)
    order = np.random.default_rng(args.seed).permutation(len(df))[:limit]
    summaries = []
    times = []
    profiles = []
    for ratio in sorted(set(args.ratios)):
        size = int(np.floor(limit * ratio / 100))
        sample = df.iloc[np.sort(order[:size])].reset_index(drop=True)
        X, _ = make_features(sample, args.scaler, args.winsor)
        for algorithm in ALGORITHMS:
            print(f"[{ratio}% | n={size} | {algorithm}]", flush=True)
            labels, benchmark, raw = run_algorithm(X, algorithm, params, measure_memory=args.memory)
            score = evaluate_clustering(X, labels, seed=args.seed)
            summaries.append({"proporsi": ratio, "n": size, "algoritma": algorithm,
                              **benchmark, **score})
            times.extend({"proporsi": ratio, "n": size, **r} for r in raw)
            current = cluster_details(sample, labels)
            current.insert(0, "algoritma", algorithm)
            current.insert(0, "proporsi", ratio)
            profiles.append(current)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output / "ringkasan.csv", index=False)
    pd.DataFrame(times).to_csv(args.output / "ulangan_waktu.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(args.output / "profil_klaster.csv", index=False)
    metadata = {"parameter": {**vars(args), "output": args.output.name}, "environment": environment_info(),
                "catatan": "Semua algoritma memakai subset sama; metrik kualitas DBSCAN mengecualikan noise."}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    print(f"Selesai. Berkas hasil: {args.output}")


if __name__ == "__main__":
    main()
