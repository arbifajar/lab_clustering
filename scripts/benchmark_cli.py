"""Benchmark melalui terminal tanpa Streamlit, bisa menghasilkan CSV/JSON untuk laporan.

Contoh:
python scripts/benchmark_cli.py --full --repeats 10
python scripts/benchmark_cli.py --repeats 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import run_benchmark
from src.data import load_baskets, load_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='9 kombinasi proporsi/support sesuai PPT')
    parser.add_argument('--repeats', type=int, default=3, help='Jumlah pengulangan waktu, 1 sampai 10')
    parser.add_argument('--no-memory', action='store_true')
    parser.add_argument('--output', default='contoh_hasil')
    args = parser.parse_args()
    _, baskets = load_baskets()
    fractions, supports = ((25, 50, 100), (3, 5, 8)) if args.full else ((25,), (3,))

    def progress(done, total, message):
        print(f'{done}/{total} {message}', flush=True)

    result = run_benchmark(baskets, fractions, supports, args.repeats,
                           measure_memory=not args.no_memory, progress=progress)
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    result.summary.to_csv(out / 'ringkasan.csv', index=False)
    result.durations.to_csv(out / 'ulangan_waktu.csv', index=False)
    result.memory.to_csv(out / 'memori.csv', index=False)
    (out / 'metadata.json').write_text(
        json.dumps({**result.metadata, 'audit': load_audit()}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result.summary[['Proporsi (%)', 'Support minimum (%)', 'Algoritma', 'Itemset frequent', 'Median (s)']].to_string(index=False))


if __name__ == '__main__':
    main()
