"""Audit streaming dataset UCI XLSX tanpa memuat 541.909 baris ke RAM.

Pemakaian:
 python scripts/audit_dataset.py 'data/Online_Retail.xlsx'
Output: data/customer_rfm.csv dan data/audit.json
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import csv

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
COLUMNS = "ABCDEFGH"


def as_text(cell, shared):
    kind = cell.get("t")
    v = cell.find(NS + "v")
    if v is not None and v.text is not None:
        if kind == "s":
            return shared[int(v.text)]
        return v.text
    if kind == "inlineStr":
        return "".join(cell.itertext())
    return ""


def audit_xlsx(source: Path, output: Path) -> dict:
    from src.preprocessing import file_sha256
    result = defaultdict(int)
    invoices_assoc = set()
    products_assoc = set()
    invoices_rfm = set()
    customers = {}  # customer -> [last_date, set(invoice), money]
    date_min = None
    date_max = None
    with ZipFile(source) as archive:
        shared = []
        with archive.open("xl/sharedStrings.xml") as f:
            for _, elem in etree.iterparse(f, events=("end",), tag=NS + "si"):
                shared.append("".join(elem.itertext()))
                elem.clear()
        with archive.open("xl/worksheets/sheet1.xml") as f:
            for _, row in etree.iterparse(f, events=("end",), tag=NS + "row"):
                row_num = int(row.get("r", 0))
                if row_num <= 1:
                    row.clear()
                    continue
                result["baris_awal"] += 1
                cells = {}
                for cell in row:
                    ref = cell.get("r", "")
                    if ref and ref[0] in COLUMNS:
                        cells[ref[0]] = as_text(cell, shared)
                invoice = str(cells.get("A", "")).strip()
                stock = str(cells.get("B", "")).strip()
                qty_s = cells.get("D", "")
                date_s = cells.get("E", "")
                price_s = cells.get("F", "")
                customer_s = str(cells.get("G", "")).strip()
                if invoice.upper().startswith("C"):
                    result["baris_invoice_dibatalkan"] += 1
                try:
                    qty = float(qty_s)
                except (ValueError, TypeError):
                    qty = 0
                try:
                    price = float(price_s)
                except (ValueError, TypeError):
                    price = 0
                if qty <= 0:
                    result["baris_quantity_nonpositif"] += 1
                if price <= 0:
                    result["baris_harga_nonpositif"] += 1
                if not invoice or invoice.upper().startswith("C") or not stock or qty <= 0 or price <= 0:
                    row.clear()
                    continue
                result["baris_valid_aturan_asosiasi"] += 1
                invoices_assoc.add(invoice)
                products_assoc.add(stock)
                if not customer_s:
                    result["baris_tanpa_customer_id_setelah_filter_dasar"] += 1
                    row.clear()
                    continue
                try:
                    customer = str(int(float(customer_s)))
                    date = datetime(1899, 12, 30) + timedelta(days=float(date_s))
                except (ValueError, OverflowError, TypeError):
                    result["baris_gagal_parse_customer_atau_tanggal"] += 1
                    row.clear()
                    continue
                if date_min is None or date < date_min:
                    date_min = date
                if date_max is None or date > date_max:
                    date_max = date
                result["baris_valid_rfm"] += 1
                invoices_rfm.add(invoice)
                if customer not in customers:
                    customers[customer] = [date, set(), 0.0]
                person = customers[customer]
                if date > person[0]:
                    person[0] = date
                person[1].add(invoice)
                person[2] += qty * price
                row.clear()
                # Drop processed sibling nodes to bound streaming XML memory.
                while row.getprevious() is not None:
                    del row.getparent()[0]
    assert date_max is not None
    reference = datetime(date_max.year, date_max.month, date_max.day) + timedelta(days=1)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "customer_rfm.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CustomerID", "Recency", "Frequency", "Monetary"])
        for customer, (last, invoices, monetary) in sorted(customers.items(), key=lambda kv: int(kv[0])):
            recency = (reference.date() - last.date()).days
            writer.writerow([customer, recency, len(invoices), f"{monetary:.2f}"])
    result.update({
        "jumlah_invoice_valid_dasar": len(invoices_assoc),
        "jumlah_stockcode_unik_valid_dasar": len(products_assoc),
        "jumlah_invoice_valid_rfm": len(invoices_rfm),
        "jumlah_pelanggan": len(customers),
        "tanggal_minimum_rfm": str(date_min.date()),
        "tanggal_maksimum_rfm": str(date_max.date()),
        "tanggal_referensi": str(reference.date()),
        "sumber": source.name,
        "sha256_xlsx": file_sha256(source),
        "unit_clustering": "pelanggan (CustomerID), bukan invoice atau StockCode",
        "filter_rfm": "Invoice bukan C*, Invoice/StockCode terisi, Quantity>0, UnitPrice>0, CustomerID+InvoiceDate terisi",
        "definisi": {
            "Recency": "hari dari pembelian terakhir hingga 2011-12-10 (tanggal maksimum +1 hari)",
            "Frequency": "jumlah InvoiceNo unik valid per CustomerID",
            "Monetary": "jumlah Quantity*UnitPrice dari baris positif; bukan laba bersih dan tidak mencakup retur",
        },
        "catatan": "Hitungan filter dapat bertumpang tindih; baris valid asosiasi memasukkan CustomerID yang hilang.",
    })
    (output / "audit.json").write_text(json.dumps(dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(result)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    source = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "Online_Retail.xlsx")
    output = ROOT / "data"
    print(json.dumps(audit_xlsx(source, output), ensure_ascii=False, indent=2))
