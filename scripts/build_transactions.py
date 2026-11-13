"""Konversi UCI Online Retail XLSX ke basket per invoice (streaming, hemat RAM).

Dijalankan sekali sebelum deploy; tidak diperlukan untuk membuka aplikasi karena
`data/transactions.csv.gz` dan `data/audit.json` sudah disertakan.
Jalankan dari akar proyek: python scripts/build_transactions.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_str(cell, shared: list[str]) -> str:
    val = cell.find(NS + "v")
    typ = cell.get("t")
    if val is not None and val.text is not None:
        return shared[int(val.text)] if typ == "s" else val.text
    if typ == "inlineStr":
        return "".join(cell.itertext())
    return ""


def build(xlsx: Path, outdir: Path) -> dict:
    audit = Counter()
    baskets: dict[str, set[str]] = defaultdict(set)
    product_names: defaultdict[str, Counter[str]] = defaultdict(Counter)
    with ZipFile(xlsx) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as fh:
                for _, elem in etree.iterparse(fh, events=("end",), tag=NS + "si"):
                    shared.append("".join(elem.itertext()))
                    elem.clear()
        with z.open("xl/worksheets/sheet1.xml") as fh:
            for _, row in etree.iterparse(fh, events=("end",), tag=NS + "row"):
                if int(row.get("r", 0)) <= 1:
                    row.clear()
                    continue
                audit["baris_awal"] += 1
                cells = {cell.get("r", "")[0]: cell_str(cell, shared)
                         for cell in row if cell.get("r")}
                invoice = cells.get("A", "").strip()
                stock = cells.get("B", "").strip()
                description = cells.get("C", "").strip()
                qty_raw = cells.get("D", "")
                price_raw = cells.get("F", "")
                cid = cells.get("G", "").strip()
                cancelled = invoice.upper().startswith("C")
                audit["baris_invoice_dibatalkan"] += int(cancelled)
                try:
                    qty = float(qty_raw)
                except (ValueError, TypeError):
                    qty = 0.0
                try:
                    price = float(price_raw)
                except (ValueError, TypeError):
                    price = 0.0
                audit["baris_quantity_nonpositif"] += int(qty <= 0)
                audit["baris_harga_nonpositif"] += int(price <= 0)
                if not invoice or cancelled or not stock or qty <= 0 or price <= 0:
                    row.clear()
                    while row.getprevious() is not None:
                        del row.getparent()[0]
                    continue
                audit["baris_valid"] += 1
                audit["baris_valid_tanpa_customer_id"] += int(not cid)
                baskets[invoice].add(stock)
                if description:
                    product_names[stock][description] += 1
                row.clear()
                while row.getprevious() is not None:
                    del row.getparent()[0]

    outdir.mkdir(parents=True, exist_ok=True)
    with gzip.open(outdir / "transactions.csv.gz", "wt", newline="", encoding="utf-8", compresslevel=6) as out:
        writer = csv.writer(out)
        writer.writerow(["InvoiceNo", "ItemsJson"])
        for invoice, basket in sorted(baskets.items()):
            writer.writerow([invoice, json.dumps(sorted(basket), ensure_ascii=False)])
    with (outdir / "products.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["StockCode", "Description", "FrekuensiDeskripsi"])
        for sku in sorted(product_names):
            descr, count = product_names[sku].most_common(1)[0]
            writer.writerow([sku, descr, count])
    result = {
        **dict(audit),
        "jumlah_invoice": len(baskets),
        "jumlah_produk_unik": len({sku for tx in baskets.values() for sku in tx}),
        "total_kemunculan_produk_unik_dalam_invoice": sum(map(len, baskets.values())),
        "jumlah_invoice_kosong": sum(not tx for tx in baskets.values()),
        "sha256_sumber": sha256(xlsx),
        "berkas_sumber": xlsx.name,
        "aturan_filter": "InvoiceNo tidak kosong dan bukan pembatalan awalan C, StockCode tidak kosong, Quantity > 0, UnitPrice > 0.",
        "unit_analisis": "Satu InvoiceNo = satu transaksi; StockCode unik per invoice; CustomerID kosong tetap disertakan.",
        "catatan": "Jumlah kondisi gagal filter bisa bertumpang tindih; baris valid berbeda dari jumlah kemunculan unik.",
    }
    (outdir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "Online_Retail.xlsx"
    print(json.dumps(build(src, ROOT / "data"), ensure_ascii=False, indent=2))
