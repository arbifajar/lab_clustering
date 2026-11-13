"""Streamlit Topik 3 - Association Rule Mining Online Retail (Kelompok 2).

Start: streamlit run app.py
Implementasi ini tidak melakukan clustering.
"""
from __future__ import annotations

import io
import json
from collections import Counter
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import plotly.express as px
import streamlit as st

from src.benchmark import BenchmarkResult, run_benchmark
from src.data import ROOT, baskets_subset, counts_frame, load_audit, load_baskets, load_products
from src.mining import min_count_for
from src.rules import make_rules

st.set_page_config(
    page_title="Lab Association Rule Mining | Kelompok 2",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2.8rem;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    div[data-testid="stMetric"] {background: rgba(31, 146, 184, .08); border: 1px solid rgba(36, 138, 180, .20);
      padding: .85rem 1rem; border-radius: .75rem;}
    section[data-testid="stSidebar"] {border-right: 1px solid rgba(90, 120, 160, .18);}
    .hero {background: linear-gradient(115deg, #061a3d, #105a86); color: white; padding: 24px 28px;
       border-radius: 16px; margin-bottom: 15px;}
    .hero h1 {color:#f4f9ff; margin:0 0 6px 0; font-size:2.05rem;}
    .hero p {color:#dbf5ff; margin:0; font-size:1.02rem;}
    .hero .eyebrow {color:#58e4ef; font-size:.82rem; font-weight:700; letter-spacing:.1em; margin-bottom:6px;}
    .mini {color:#657991; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Memuat 19.960 basket dari dataset pra-pemrosesan…")
def get_data():
    invoice_ids, baskets = load_baskets()
    products, audit = load_products(), load_audit()
    return invoice_ids, baskets, products, audit


invoice_ids, baskets, products, audit = get_data()

st.markdown("""
<div class="hero">
  <div class="eyebrow">KELOMPOK 2 · TOPIK 3 · ANALISIS DAN DESAIN ALGORITMA</div>
  <h1>Lab Association Rule Mining</h1>
  <p>Perbandingan Apriori, FP-Growth, dan Eclat pada dataset UCI Online Retail</p>
</div>
""", unsafe_allow_html=True)
st.caption("Objek penelitian: transaksi per InvoiceNo dan kemunculan StockCode. **Bukan clustering dan bukan RFM pelanggan.**")

with st.sidebar:
    st.header("⚙️ Pengaturan eksperimen")
    mode = st.radio(
        "Paket skenario",
        ["Demonstrasi cepat", "Replikasi 9 skenario slide", "Atur sendiri"],
        help="Skenario slide = 25/50/100% invoice × support 3/5/8%. Tiga algoritma dijalankan untuk setiap kondisi.",
    )
    if mode == "Demonstrasi cepat":
        fractions, supports = [25], [3]
        repeats = st.number_input("Pengulangan waktu", min_value=1, max_value=10, value=3, step=1)
        st.info("Mode cepat: 4.990 invoice × support 3% × 3 algoritma (aturan asosiasi muncul).")
    elif mode == "Replikasi 9 skenario slide":
        fractions, supports = [25, 50, 100], [3, 5, 8]
        repeats = st.number_input("Pengulangan waktu", min_value=1, max_value=10, value=10, step=1)
        st.warning("Mode lengkap: 9 kondisi × 3 algoritma × 10 ulangan = 270 pengukuran waktu. Aktifkan checkbox memori bila ingin 27 run memori tambahan (lebih lambat).")
    else:
        fractions = st.multiselect("Proporsi invoice (%)", [25, 50, 100], default=[25, 50, 100])
        supports = st.multiselect("Min support (%)", [3, 5, 8], default=[3, 5, 8])
        repeats = st.number_input("Pengulangan waktu", min_value=1, max_value=10, value=3, step=1)
    seed = st.number_input("Seed pengambilan invoice", value=42, step=1)
    max_len = st.selectbox("Batas panjang itemset (L)", [3, 2, 1], index=0)
    memory_enabled = st.checkbox("Ukur memori Python (run terpisah)", value=(mode != "Replikasi 9 skenario slide"), key=f"ukur_memori_{mode}",
                                 help="Pada mode lengkap, pengukuran tracemalloc untuk 27 kombinasi dapat berlangsung lama. Aktifkan setelah benchmark waktu selesai bila memori diperlukan.")
    st.divider()
    st.caption("**Rekomendasi presentasi:** demonstrasi cepat terlebih dahulu, lalu jalankan mode lengkap untuk hasil laporan. Gunakan seed 42 dan L=3 agar identik dengan rancangan slide.")
    st.caption("Aplikasi ini versi Association Rule Mining. Untuk mengganti deployment lama di Streamlit Cloud, unggah proyek ini ke repositori yang digunakan aplikasi.")

signature = (tuple(fractions), tuple(supports), int(repeats), int(seed), int(max_len), bool(memory_enabled))
if "run" not in st.session_state:
    st.session_state.run = None
if "run_signature" not in st.session_state:
    st.session_state.run_signature = None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Baris sumber", f"{audit['baris_awal']:,}".replace(",", "."))
m2.metric("Baris valid", f"{audit['baris_valid']:,}".replace(",", "."))
m3.metric("Transaksi invoice", f"{len(baskets):,}".replace(",", "."))
m4.metric("Kode produk unik", f"{audit['jumlah_produk_unik']:,}".replace(",", "."))

TAB_NAMES = ["1 · Dataset & Pembersihan", "2 · Benchmark", "3 · Itemset & Validasi",
             "4 · Aturan Asosiasi", "5 · Metodologi & Unduhan"]
tab_data, tab_bench, tab_item, tab_rule, tab_method = st.tabs(TAB_NAMES)


def pretty_int(number: int) -> str:
    return f"{number:,}".replace(",", ".")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def result_if_available() -> BenchmarkResult | None:
    return st.session_state.run


with tab_data:
    st.subheader("Satuan data yang dianalisis")
    st.markdown("""
Dataset UCI Online Retail berisi baris barang dalam transaksi. **Satu InvoiceNo dianggap satu transaksi**,
sementara item adalah **StockCode**. Jika kode barang sama muncul berkali-kali di invoice yang sama,
frekuensinya tetap dihitung **satu kali**. `CustomerID` boleh kosong karena tujuan analisis adalah
menemukan *ko-kemunculan barang*, bukan mengenali pelanggan.
""")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Audit pembersihan")
        audit_table = pd.DataFrame([
            ("Baris input", audit["baris_awal"]),
            ("Baris valid setelah filter", audit["baris_valid"]),
            ("Invoice unik valid", audit["jumlah_invoice"]),
            ("StockCode unik valid", audit["jumlah_produk_unik"]),
            ("Kehadiran item unik per invoice", audit["total_kemunculan_produk_unik_dalam_invoice"]),
        ], columns=["Ukuran", "Jumlah"])
        st.dataframe(audit_table, hide_index=True, use_container_width=True)
        st.markdown("""**Filter:** hapus InvoiceNo kosong atau berawalan `C` (pembatalan),
StockCode kosong, `Quantity ≤ 0`, serta `UnitPrice ≤ 0`.
Hitungan alasan penolakan dapat bertumpang tindih dan **tidak** boleh dijumlahkan.
""")
        st.caption(f"SHA-256 Excel sumber: `{audit['sha256_sumber']}`")
    with right:
        st.markdown("#### Contoh lima basket")
        preview = pd.DataFrame({
            "InvoiceNo": invoice_ids[:5],
            "Jumlah barang unik": [len(b) for b in baskets[:5]],
            "Contoh StockCode": [", ".join(sorted(x)[:7]) for x in baskets[:5]],
        })
        st.dataframe(preview, hide_index=True, use_container_width=True)
        st.caption("Basket berasal dari CSV pra-pemrosesan yang disertakan; tidak perlu membaca ulang Excel 23 MB pada setiap klik.")
    st.markdown("#### Kode barang paling sering muncul (jumlah invoice, bukan jumlah unit terjual)")
    item_counter = Counter(item for basket in baskets for item in basket)
    top = pd.DataFrame([{"StockCode": k, "Nama": products.get(k, ""), "Invoice": v,
                         "Support (%)": 100 * v / len(baskets)} for k, v in item_counter.most_common(15)])
    st.bar_chart(top.set_index("StockCode")["Invoice"], height=240)
    st.dataframe(top, hide_index=True, use_container_width=True)
    st.download_button("Unduh dataset basket pra-pemrosesan (.csv.gz)",
                       (ROOT / "data" / "transactions.csv.gz").read_bytes(),
                       file_name="online_retail_basket.csv.gz", mime="application/gzip")

with tab_bench:
    st.subheader("Jalankan perbandingan Apriori, FP-Growth, Eclat")
    st.markdown("""
Setiap kondisi menggunakan **invoice dan ambang support yang sama**. Waktu diulang,
median dipakai sebagai ringkasan utama, dan `tracemalloc` dijalankan **satu kali terpisah**
untuk menghindari tambahan overhead pada pengukuran waktu.
""")
    n_cases = len(fractions) * len(supports)
    st.info(f"Konfigurasi aktif: **{n_cases} skenario**, **{n_cases * 3} kombinasi algoritma**, "
            f"**{n_cases * 3 * int(repeats)}** pengukuran waktu. L={max_len}, seed={seed}.")
    if max_len != 3 or seed != 42:
        st.warning("Seed atau panjang itemset berbeda dari protokol PowerPoint. Hasil jumlah itemset dapat berubah.")
    ready = bool(fractions and supports)
    if st.button("▶ Jalankan benchmark sesuai pengaturan", type="primary", disabled=not ready):
        bar = st.progress(0.0)
        status = st.empty()
        def progress(done, total, message):
            bar.progress(min(1.0, done / max(1, total)))
            status.caption(f"Proses {done}/{total}: {message}")
        try:
            with st.spinner("Menjalankan perhitungan. Jangan refresh halaman saat benchmark berlangsung."):
                result = run_benchmark(
                    baskets, fractions=fractions, supports=supports, repeats=int(repeats),
                    seed=int(seed), max_len=int(max_len), measure_memory=memory_enabled,
                    progress=progress,
                )
            st.session_state.run = result
            st.session_state.run_signature = signature
            st.success("Benchmark selesai. Semua frequent itemset beserta support count identik antaralgoritma untuk setiap skenario.")
        except Exception as exc:
            st.session_state.run = None
            st.session_state.run_signature = None
            st.error(f"Pengujian dihentikan: {exc}")

    result = result_if_available()
    if result is None:
        st.caption("Belum ada hasil pada sesi browser ini. Jalankan benchmark untuk melihat metrik dan grafik.")
    else:
        if signature != st.session_state.run_signature:
            st.warning("Pengaturan sidebar telah berubah setelah benchmark. Hasil di bawah tetap menggunakan konfigurasi saat pengujian terakhir. Jalankan ulang untuk memperbarui.")
        st.markdown("#### Ringkasan seluruh skenario")
        show_cols = ["Proporsi (%)", "Support minimum (%)", "Jumlah invoice", "Min count", "Algoritma",
                     "Itemset frequent", "Median (s)", "Q1 (s)", "Q3 (s)", "CV (%)", "Peak Python (MiB)", "Kesetaraan output"]
        st.dataframe(result.summary[show_cols], hide_index=True, use_container_width=True)
        st.caption("CV dihitung dari SD sampel/rata-rata. Jika hanya 1 ulangan, SD dan CV tidak terdefinisi. Q1–Q3 bukan interval kepercayaan.")
        available_support = sorted(result.summary["Support minimum (%)"].unique().tolist())
        s_chart = st.selectbox("Pilih support untuk grafik benchmark", available_support, key="benchmark_support")
        graph = result.summary[result.summary["Support minimum (%)"] == s_chart]
        c1, c2 = st.columns(2)
        with c1:
            time_chart = px.line(graph, x="Jumlah invoice", y="Median (s)", color="Algoritma",
                                 markers=True, title=f"Skalabilitas empiris | support {s_chart}%",
                                 hover_data=["Proporsi (%)", "Itemset frequent"])
            time_chart.update_layout(legend_title_text="Algoritma", xaxis_title="Jumlah invoice")
            st.plotly_chart(time_chart, use_container_width=True)
        with c2:
            time_bar = px.bar(graph, x="Proporsi (%)", y="Median (s)", color="Algoritma",
                              barmode="group", title="Perbandingan median waktu")
            st.plotly_chart(time_bar, use_container_width=True)
        st.markdown("#### Penggunaan memori")
        if result.memory["Run memori independen"].max() == 0:
            st.info("Pengukuran memori dimatikan pada benchmark ini.")
        else:
            mem_graph = graph.dropna(subset=["Peak Python (MiB)"])
            m1, m2 = st.columns(2)
            with m1:
                fig_mem = px.bar(mem_graph, x="Proporsi (%)", y="Peak Python (MiB)", color="Algoritma",
                                 barmode="group", title="Peak alokasi Python selama mining")
                st.plotly_chart(fig_mem, use_container_width=True)
            with m2:
                fig_rss = px.bar(mem_graph, x="Proporsi (%)", y="Delta RSS setelah (MiB)", color="Algoritma",
                                 barmode="group", title="Perubahan RSS (bukan peak RAM)")
                st.plotly_chart(fig_rss, use_container_width=True)
            st.caption("Peak tracemalloc hanya alokasi Python yang dilacak pada run terpisah. Delta RSS adalah perubahan sebelum/sesudah, dapat negatif, dan bukan maksimum RAM proses.")
        st.markdown("#### Waktu setiap ulangan")
        st.dataframe(result.durations, hide_index=True, use_container_width=True)
        st.download_button("Unduh seluruh waktu mentah CSV", csv_bytes(result.durations),
                           file_name="benchmark_ulangan_waktu.csv", mime="text/csv")

with tab_item:
    st.subheader("Frequent itemset dan validasi kesetaraan hasil")
    st.markdown("Frequent itemset adalah kumpulan 1–3 barang yang kemunculannya memenuhi minimal support. "
                "Apriori, FP-Growth, dan Eclat **harus menghasilkan daftar itemset dan count yang sama**, bukan sekadar jumlahnya sama.")
    result = result_if_available()
    if result is None:
        st.info("Jalankan benchmark terlebih dahulu pada tab 2.")
    else:
        selections = sorted(result.counts_by_scenario)
        chosen = st.selectbox("Pilih skenario (proporsi data, support)", selections,
                              format_func=lambda x: f"{x[0]}% invoice — min support {x[1]}%", key="items_scenario")
        itemcounts = result.counts_by_scenario[chosen]
        frac, sup = chosen
        n = int(result.summary.loc[(result.summary["Proporsi (%)"] == frac) &
                                   (result.summary["Support minimum (%)"] == sup), "Jumlah invoice"].iloc[0])
        mc = min_count_for(n, sup / 100)
        ia, ib, ic, id_ = st.columns(4)
        ia.metric("Itemset frequent", pretty_int(len(itemcounts)))
        ib.metric("Itemset 1 barang", pretty_int(sum(len(x) == 1 for x in itemcounts)))
        ic.metric("Itemset 2 barang", pretty_int(sum(len(x) == 2 for x in itemcounts)))
        id_.metric("Itemset 3 barang", pretty_int(sum(len(x) == 3 for x in itemcounts)))
        st.success(f"Validasi cocok: hasil Apriori = FP-Growth = Eclat | {pretty_int(n)} invoice | support minimal {mc} invoice.")
        table = counts_frame(itemcounts, n, products)
        if not table.empty:
            table = table.sort_values(["Support count", "Panjang", "Kode barang"],
                                      ascending=[False, True, True])
            selected_length = st.multiselect("Tampilkan panjang itemset", [1, 2, 3], default=[1, 2, 3])
            table = table[table["Panjang"].isin(selected_length)]
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button("Unduh frequent itemset CSV", csv_bytes(table),
                               file_name=f"itemset_{frac}persen_support{sup}.csv", mime="text/csv")
        else:
            st.warning("Tidak ditemukan itemset frequent pada ambang ini.")
        st.caption("'Count' adalah jumlah invoice yang memuat seluruh barang dalam itemset, bukan jumlah unit atau nilai penjualan.")

with tab_rule:
    st.subheader("Aturan asosiasi: support, confidence, lift")
    st.markdown("Aturan **A → B** menyatakan peluang B hadir pada invoice yang memuat A. "
                "Arah aturan penting: confidence(A→B) umumnya **berbeda** dari confidence(B→A). "
                "Lift adalah rasio peluang bersama terhadap baseline kemunculan B; bukan bukti sebab-akibat.")
    result = result_if_available()
    if result is None:
        st.info("Jalankan benchmark di tab 2 untuk membangkitkan aturan dari itemset yang telah divalidasi.")
    else:
        choices = sorted(result.counts_by_scenario)
        selected = st.selectbox("Skenario untuk aturan asosiasi", choices,
                                format_func=lambda x: f"{x[0]}% invoice — support {x[1]}%", key="rule_scenario")
        fra, sup = selected
        row = result.summary.loc[(result.summary["Proporsi (%)"] == fra) &
                                 (result.summary["Support minimum (%)"] == sup)].iloc[0]
        count_invoices = int(row["Jumlah invoice"])
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            confidence = st.slider("Confidence minimal (%)", 0, 100, 30, 5,
                                   help="30% adalah ambang ekspor aturan pada PowerPoint.")
        with col_b:
            format_rules = st.radio("Bentuk aturan", ["Pasangan 1 → 1 (seperti slide)", "Semua aturan sampai L"],
                                    help="Untuk membandingkan 18 aturan di slide pilih pasangan 1→1.")
        with col_c:
            min_lift = st.number_input("Lift minimal", 0.0, 50.0, 0.0, 0.5)
        rules = make_rules(result.counts_by_scenario[selected], count_invoices,
                           min_confidence=confidence / 100,
                           pair_only=format_rules.startswith("Pasangan"),
                           min_lift=float(min_lift), products=products)
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah aturan", pretty_int(len(rules)))
        col2.metric("Confidence minimum", f"{confidence}%")
        col3.metric("Support minimum", f"{sup}%")
        if rules.empty:
            st.warning("Tidak ada aturan yang memenuhi parameter sekarang. Coba turunkan confidence atau lift.")
        else:
            show = rules.copy()
            numeric = ["Support (%)", "Confidence (%)", "Lift"]
            show[numeric] = show[numeric].round(3)
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.download_button("Unduh aturan CSV", csv_bytes(rules),
                               file_name=f"aturan_{fra}persen_support{sup}_conf{confidence}.csv", mime="text/csv")
            chart = px.scatter(rules, x="Support (%)", y="Confidence (%)", size="Lift",
                               color="Lift", hover_name="Antecedent", hover_data=["Consequent", "Count A∩B"],
                               title="Sebaran support–confidence (ukuran titik = lift)")
            st.plotly_chart(chart, use_container_width=True)
        if selected == (100, 3) and confidence <= 30 and min_lift <= 16:
            st.markdown("#### Contoh rujukan dari PowerPoint: cangkir Regency")
            st.info("Kombinasi **22697 → 22698** ada pada **632 dari 19.960 invoice** (support ≈3,17%). "
                    "Confidence arah 22697→22698 ≈62,39%; arah sebaliknya ≈82,61%; lift ≈16,28. "
                    "Berarti ko-kemunculan jauh lebih sering daripada baseline independen, **bukan** hubungan sebab-akibat.")

with tab_method:
    st.subheader("Landasan metodologi dan hubungan dengan slide")
    st.markdown("""
**Topik tugas yang benar:** Kelompok 2 — Topik 3, *Association Rule Mining: Penemuan Pola Asosiasi Transaksi*.

- **Apriori:** pencarian per tingkat; membuat kandidat itemset, memangkas kandidat yang subsetnya tidak frequent, dan menghitung dukungan dengan memindai basket.
- **FP-Growth:** menyusun FP-tree dan basis pola kondisional; menggali frequent itemset tanpa enumerasi kandidat ala Apriori.
- **Eclat:** mengubah representasi menjadi *vertical tidset* (bitset), kemudian mencari itemset dengan irisan TID secara *depth-first*.

**Protokol yang diadaptasi persis:** dataset sama, filter sama, L=3, `seed=42`, sampel bersarang 25%, 50%, 100%, threshold support 3%, 5%, 8%, `min_count=ceil(s·n)`, serta validasi count identik.
**Pengukuran:** 10 ulangan waktu untuk meniru slide (median) dan satu run memori terpisah. Memori `tracemalloc` **bukan** total RAM, sehingga jangan mencampurnya dengan delta RSS.

**Batas interpretasi:** beda waktu antara aplikasi ini dan PowerPoint bisa berasal dari algoritma Python yang ditulis ulang, versi Python/library, jumlah proses, cache, dan komputer. Tiga ukuran sampel tidak membuktikan kelas kompleksitas Big-O.
""")
    with st.expander("Lihat kesembilan jumlah itemset pada PowerPoint (nilai referensi)", expanded=True):
        reference = pd.DataFrame([
            (25, 3, 4990, 150), (25, 5, 4990, 33), (25, 8, 4990, 4),
            (50, 3, 9980, 141), (50, 5, 9980, 33), (50, 8, 9980, 4),
            (100, 3, 19960, 142), (100, 5, 19960, 36), (100, 8, 19960, 4),
        ], columns=["Proporsi (%)", "Support (%)", "Invoice", "Jumlah itemset referensi PPT"])
        st.dataframe(reference, hide_index=True, use_container_width=True)
        st.caption("Angka ini diambil dari lampiran slide, BUKAN diukur ulang pada runtime saat ini. Bandingkan dengan tab Itemset & Validasi setelah benchmark.")
    with st.expander("Rumus support, confidence, dan lift"):
        st.latex(r"\mathrm{support}(A) = \frac{\#\{t: A \subseteq t\}}{N}")
        st.latex(r"\mathrm{confidence}(A\to B) = \frac{\mathrm{support}(A\cup B)}{\mathrm{support}(A)}")
        st.latex(r"\mathrm{lift}(A\to B) = \frac{\mathrm{confidence}(A\to B)}{\mathrm{support}(B)}")
        st.write("Support di UI ditampilkan dalam persen; support count adalah pembilangnya. "
                 "Lift >1 = asosiasi positif, =1 = mendekati independen, <1 = asosiasi negatif pada distribusi ini.")
    with st.expander("Pertanyaan dosen yang umum"):
        st.markdown("""
**Mengapa ini bukan clustering?** Outputnya itemset frequent dan aturan barang A→B, bukan label kelompok pelanggan.

**Mengapa CustomerID kosong tetap ada?** Karena unit analisis adalah InvoiceNo, bukan CustomerID.

**Mengapa aturan kedua arah bisa berbeda?** Confidence dibagi support antecedent; support gabungan serta lift sama untuk kedua arah.

**Mengapa Eclat/FP-Growth dan Apriori harus identik?** Untuk data, ambang support, dan batas panjang yang sama, ketiganya algoritma frequent itemset eksak.

**Apakah 18 aturan berarti 18 pasangan?** Tidak: pada support 3%, confidence 30%, dan filter 1→1 terdapat 18 **aturan berarah** dari 9 pasangan produk.

**Apakah lift besar berarti rekomendasi pasti menguntungkan?** Tidak. Lift bukan profit atau bukti kausalitas; margin, penjualan, musiman, dan validasi temporal belum diuji.

**Mengapa waktu benchmark tidak sama dengan slide?** Kode implementasi dan perangkat keras berbeda, meskipun output itemset bisa identik.

**Apakah 10 kali pengukuran memori?** Tidak. Sepuluh pengulangan hanya waktu, sementara memori satu eksekusi instrumentasi terpisah.
""")
    st.markdown("#### Ekspor hasil penelitian")
    result = result_if_available()
    if result is None:
        st.caption("Jalankan benchmark agar tautan ekspor hasil aktif.")
    else:
        bundle = io.BytesIO()
        with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
            z.writestr("ringkasan_benchmark.csv", csv_bytes(result.summary))
            z.writestr("ulangan_waktu.csv", csv_bytes(result.durations))
            z.writestr("pengukuran_memori.csv", csv_bytes(result.memory))
            metadata = {**result.metadata, "audit_dataset": audit}
            z.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            for (frac, support), count_map in result.counts_by_scenario.items():
                n = int(result.summary.loc[(result.summary["Proporsi (%)"] == frac) &
                                           (result.summary["Support minimum (%)"] == support), "Jumlah invoice"].iloc[0])
                z.writestr(f"itemset_{frac}p_support{support}p.csv", csv_bytes(counts_frame(count_map, n, products)))
                rules = make_rules(count_map, n, .30, pair_only=True, products=products)
                z.writestr(f"aturan_pasangan_{frac}p_support{support}p.csv", csv_bytes(rules))
        st.download_button("📦 Unduh paket seluruh hasil CSV + metadata JSON (.zip)",
                           bundle.getvalue(), file_name="hasil_benchmark_association_kelompok2.zip",
                           mime="application/zip", type="primary")
        st.json(result.metadata, expanded=False)
    st.caption("Acuan dataset: https://archive.ics.uci.edu/dataset/352/online+retail • Acuan rancangan: PowerPoint 'Perbandingan Apriori, FP-Growth, dan Eclat'.")
