"""Dasbor edukatif: komparasi K-Means, DBSCAN dan Agglomerative Clustering.

Eksekusi: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src.benchmark import ALGORITHMS, Parameters, environment_info, run_algorithm
from src.evaluation import cluster_details, evaluate_clustering
from src.preprocessing import (
    RFM_COLUMNS, build_customer_rfm, file_sha256,
    make_features, read_online_retail, validate_rfm,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
APP_VERSION = "1.0.0"

st.set_page_config(
    page_title="Lab Clustering | UCI Online Retail",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
      .block-container {padding-top: 1.9rem; max-width: 1420px}
      .big-note {font-size: 1.05rem; color: #475569}
      div[data-testid='stMetric'] {background: #F5F8FC; border:1px solid #E5EBF4;
        padding: 12px 16px; border-radius: 12px}
      div[data-testid='stMetricLabel'] {font-weight: 650}
      h1,h2,h3 {letter-spacing: -.025em}
      div.stButton > button[kind='primary'] {font-weight: 700}
    </style>""",
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Menyiapkan fitur pelanggan...")
def load_packaged_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    csv = pd.read_csv(DATA_DIR / "customer_rfm.csv", dtype={"CustomerID": str})
    audit = json.loads((DATA_DIR / "audit.json").read_text(encoding="utf-8"))
    return validate_rfm(csv), audit

@st.cache_data(show_spinner="Membaca dan memproses Excel; dataset besar dapat memerlukan waktu...")
def load_xlsx_bytes(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_online_retail(content)
    customer_rfm, audit = build_customer_rfm(raw)
    audit["sha256_xlsx"] = file_sha256(content)
    audit["sumber"] = "Excel unggahan pengguna"
    return validate_rfm(customer_rfm), audit

@st.cache_data(show_spinner=False)
def load_csv_bytes(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv = pd.read_csv(__import__("io").BytesIO(content), dtype={"CustomerID": str})
    df = validate_rfm(csv)
    return df, {
        "sumber": "CSV RFM unggahan pengguna", "sha256_csv": file_sha256(content),
        "jumlah_pelanggan": len(df),
        "catatan": "Data RFM telah dihitung oleh pengguna; audit transaksi mentah tidak tersedia.",
    }


def stable_signature(**kwargs: Any) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()


def number_metric(value: float, decimals: int = 3) -> str:
    return "—" if pd.isna(value) else f"{value:,.{decimals}f}"


def plot_clusters(X: np.ndarray, labels: np.ndarray, rfm: pd.DataFrame, mode: str, seed: int):
    if mode == "UMAP (opsional)":
        try:
            from umap import UMAP
        except ImportError:
            st.warning("Paket UMAP belum terpasang. Jalankan `pip install umap-learn` atau pilih PCA.")
            return None
        points = UMAP(n_components=2, random_state=seed, n_neighbors=min(15, len(X) - 1)).fit_transform(X)
        x_label, y_label = "UMAP 1", "UMAP 2"
        explanation = "UMAP menekankan kedekatan lokal; jarak pada grafik tidak mewakili jarak numerik asli."
    else:
        pca = PCA(n_components=2, random_state=seed)
        points = pca.fit_transform(X)
        ratios = pca.explained_variance_ratio_
        x_label = f"PC1 ({ratios[0]:.1%} variasi)"
        y_label = f"PC2 ({ratios[1]:.1%} variasi)"
        explanation = f"PCA menampilkan {ratios.sum():.1%} variasi fitur; skor kualitas tetap dihitung pada 3 dimensi fitur."
    plot_df = rfm.copy()
    plot_df["Dimensi 1"] = points[:, 0]
    plot_df["Dimensi 2"] = points[:, 1]
    plot_df["Label"] = ["Noise (-1)" if x == -1 else f"Klaster {x}" for x in labels]
    fig = px.scatter(
        plot_df,
        x="Dimensi 1", y="Dimensi 2", color="Label",
        hover_data={"CustomerID": True, "Recency": True, "Frequency": True, "Monetary": ":.2f"},
        opacity=0.73, render_mode="webgl",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title=f"Proyeksi {mode.split(' ')[0]} — label clustering",
    )
    fig.update_traces(marker={"size": 6})
    fig.update_layout(legend_title="Kelompok", height=540, margin=dict(l=20, r=20, t=55, b=20))
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    return fig, explanation


def run_experiment(
    rfm: pd.DataFrame, cap: int, ratios: list[int], p: Parameters,
    scaler: str, quantile: float, do_memory: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(p.seed)
    order = rng.permutation(len(rfm))[:cap]
    total_steps = len(ratios) * len(ALGORITHMS)
    progress = st.progress(0, text="Menyiapkan benchmark...")
    summary_rows, raw_rows, detail_rows = [], [], []
    labels_by_key, indices_by_ratio = {}, {}
    for ratio in sorted(ratios):
        size = max(1, int(np.floor(cap * ratio / 100)))
        indices = np.sort(order[:size])
        sample = rfm.iloc[indices].reset_index(drop=True)
        X, _ = make_features(sample, scaler, quantile)
        indices_by_ratio[ratio] = indices
        for algorithm in ALGORITHMS:
            progress.progress(
                len(summary_rows) / total_steps,
                text=f"Menghitung {algorithm} | {ratio}% ({len(X)} pelanggan)",
            )
            try:
                labels, stats, per_run = run_algorithm(X, algorithm, p, measure_memory=do_memory)
                scores = evaluate_clustering(X, labels, seed=p.seed)
                summary_rows.append({"Proporsi": f"{ratio}%", "Proporsi_angka": ratio,
                                     "Pelanggan": len(sample), "Algoritma": algorithm,
                                     **stats, **scores, "Status": "OK", "Galat": ""})
                raw_rows.extend({"Proporsi": f"{ratio}%", "Pelanggan": len(sample), **v} for v in per_run)
                details = cluster_details(sample, labels)
                details.insert(0, "Algoritma", algorithm)
                details.insert(0, "Proporsi", f"{ratio}%")
                detail_rows.append(details)
                labels_by_key[(ratio, algorithm)] = labels
            except (ValueError, MemoryError) as exc:
                summary_rows.append({"Proporsi": f"{ratio}%", "Proporsi_angka": ratio,
                                     "Pelanggan": len(sample), "Algoritma": algorithm,
                                     "Status": "Gagal", "Galat": str(exc)})
    progress.progress(1.0, text="Pengujian selesai.")
    return {
        "summary": pd.DataFrame(summary_rows),
        "raw": pd.DataFrame(raw_rows),
        "profiles": pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(),
        "labels": labels_by_key, "indices": indices_by_ratio,
        "metadata": {
            "versi": APP_VERSION, "cap_pelanggan": cap, "proporsi": sorted(ratios),
            "random_seed": p.seed, "k": p.k, "eps": p.eps,
            "min_samples": p.min_samples, "repetisi": p.repeats,
            "warmup": p.warmups, "scaler": scaler, "batas_kuantil": quantile,
            "fit_model": "fit_predict termasuk inisialisasi estimator, tanpa baca Excel/CSV, tanpa scaling, tanpa evaluasi",
            "silhouette": "non-noise, sampel terstratifikasi maksimal 1200 jika diperlukan",
            "memori": "satu eksekusi tambahan; tracemalloc Python peak dan RSS peak sampled relatif baseline (belum terisolasi)",
            "lingkungan": environment_info(),
        },
    }


st.title("Lab Clustering Online Retail")
st.markdown(
    '<p class="big-note">Komparasi <b>K-Means</b>, <b>DBSCAN</b>, dan '
    '<b>Agglomerative Clustering</b> untuk segmentasi pelanggan berbasis RFM.</p>',
    unsafe_allow_html=True,
)
st.info(
    "**Penting:** Tugas Kelompok 2 adalah clustering. PowerPoint lama membahas "
    "**association rule mining (Apriori, FP-Growth, Eclat)**; keduanya berbeda sasaran, "
    "parameter, hasil, dan metrik evaluasi."
)

with st.sidebar:
    st.header("1. Sumber data")
    source = st.radio("Pilih sumber", ["Dataset RFM bawaan", "Unggah Excel Online Retail", "Unggah CSV RFM"],
                      label_visibility="collapsed")
    uploaded = None
    if source != "Dataset RFM bawaan":
        uploaded = st.file_uploader(
            "Berkas sumber", type=["xlsx"] if "Excel" in source else ["csv"],
            help="Excel akan dibersihkan lalu diagregasi menjadi satu baris per CustomerID.",
        )
    st.divider()
    st.header("2. Pra-pemrosesan")
    scaler = st.selectbox("Skalasi fitur", ["StandardScaler", "RobustScaler"],
                          help="Semua fitur memakai log1p terlebih dahulu; scaler di-fit ulang per subset.")
    quantile_pct = st.selectbox("Batas nilai ekstrem", [100, 99, 95], index=0,
                                format_func=lambda x: "Tanpa pemotongan" if x == 100 else f"Kuantil {x}%")
    seed = int(st.number_input("Seed acak", min_value=0, max_value=9999999, value=42, step=1))

try:
    if source == "Dataset RFM bawaan":
        rfm, audit = load_packaged_data()
        input_hash = audit.get("sha256_xlsx", "")
    elif uploaded is None:
        st.warning("Unggah berkas terlebih dahulu atau gunakan dataset bawaan.")
        st.stop()
    elif "Excel" in source:
        content = uploaded.getvalue()
        rfm, audit = load_xlsx_bytes(content)
        input_hash = file_sha256(content)
    else:
        content = uploaded.getvalue()
        rfm, audit = load_csv_bytes(content)
        input_hash = file_sha256(content)
except (ValueError, OSError, ImportError) as exc:
    st.error(f"Tidak dapat menyiapkan data: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("3. Eksperimen")
    max_cap = min(3000, len(rfm))
    min_cap = min(50, max_cap)
    cap = st.slider("Batas pelanggan bersama", min_value=min_cap, max_value=max_cap,
                    value=min(1200, max_cap), step=1,
                    help="Batas ini berlaku sama untuk K-Means, DBSCAN, dan Agglomerative; "
                         "menghindari perbandingan tidak adil dan beban kuadratik.")
    ratios = st.multiselect("Proporsi dari batas bersama", [25, 50, 100], default=[25, 50, 100],
                            format_func=lambda v: f"{v}%")
    k_max = max(2, min(12, cap // 5))
    k = st.slider("Jumlah klaster K-Means & Agglomerative (k)", 2, k_max, min(4, k_max))
    eps = st.slider("DBSCAN: eps (jarak dalam ruang terskalakan)", 0.1, 4.0, 0.50, 0.05)
    min_samples = st.slider("DBSCAN: min_samples", 2, min(50, cap - 1), min(8, cap - 1))
    repeats = st.slider("Ulangan waktu", 1, 10, 3, help="Slide menggunakan 10 ulangan; default 3 lebih ringan.")
    warmups = st.selectbox("Warm-up sebelum timing", [0, 1], index=0)
    do_memory = st.checkbox("Ukur memori (1 fit tambahan/algoritma/kondisi)", value=True)
    st.caption("Batas atas 3.000 pelanggan sebagai perlindungan untuk DBSCAN dan Agglomerative. "
               "Untuk benchmark keras, jalankan ulang dengan isolasi proses/mesin.")

params = Parameters(k=k, eps=eps, min_samples=min_samples, seed=seed,
                    repeats=repeats, warmups=warmups)
signature = stable_signature(
    source=input_hash, n=len(rfm), cap=cap, ratios=sorted(ratios),
    params=vars(params), scaler=scaler, quantile=quantile_pct, memory=do_memory,
)

metrics = st.columns(4)
metrics[0].metric("Pelanggan siap cluster", f"{len(rfm):,}")
metrics[1].metric("Batas eksperimen", f"{cap:,}")
metrics[2].metric("Fitur model", "3 (RFM)")
metrics[3].metric("Skenario", f"{len(ratios)*3}" if ratios else "0")

tab_data, tab_experiment, tab_visual, tab_method = st.tabs(
    ["Data & Audit", "Benchmark & Evaluasi", "Visualisasi Klaster", "Metodologi & Unduhan"]
)
with tab_data:
    st.subheader("Audit dataset dan pembentukan RFM")
    st.write(
        "Sumber Online Retail berupa **baris item transaksi**. Analisis ini mengubah baris-baris "
        "valid menjadi **satu pengamatan per CustomerID**, lalu menghitung Recency, Frequency, Monetary."
    )
    a = audit
    if "baris_awal" in a:
        c = st.columns(4)
        c[0].metric("Baris awal Excel", f"{a['baris_awal']:,}")
        c[1].metric("Baris valid filter dasar", f"{a['baris_valid_aturan_asosiasi']:,}")
        c[2].metric("Baris valid untuk RFM", f"{a['baris_valid_rfm']:,}")
        c[3].metric("CustomerID tidak tersedia*", f"{a['baris_tanpa_customer_id_setelah_filter_dasar']:,}")
        st.caption("*Dihitung setelah filter transaksi dasar. Jumlah berbagai jenis baris yang tersaring "
                   "tidak boleh dijumlahkan karena kondisi dapat bertumpang tindih.")
        st.write(f"Invoice valid pada pembersihan dasar: **{a['jumlah_invoice_valid_dasar']:,}**; "
                 f"invoice valid dengan identitas pelanggan: **{a['jumlah_invoice_valid_rfm']:,}**. "
                 f"Tanggal referensi Recency: **{a['tanggal_referensi']}**.")
    else:
        st.caption("CSV yang diunggah sudah berupa RFM; hanya validasi struktur dan nilai yang dilakukan.")
    st.markdown("**Cuplikan data RFM**")
    st.dataframe(rfm.head(25), use_container_width=True, hide_index=True)
    st.markdown("**Distribusi nilai asli (sebelum log1p dan scaling)**")
    dist = rfm[list(RFM_COLUMNS)].describe(percentiles=[.25, .5, .75, .95]).T
    st.dataframe(dist.round(2), use_container_width=True)
    st.caption("Unit: Recency = hari, Frequency = invoice unik, Monetary = nilai penjualan bruto (GBP pada sumber UCI). "
               "Nominal tidak ditafsirkan sebagai laba atau setelah dikurangi retur.")
    st.download_button("Unduh fitur pelanggan (CSV)",
                       data=rfm.to_csv(index=False).encode("utf-8-sig"),
                       file_name="online_retail_customer_rfm.csv", mime="text/csv")

with tab_experiment:
    st.subheader("Pengujian terkontrol dengan tiga algoritma")
    st.write("Setiap kombinasi proporsi menggunakan **pelanggan dan matriks fitur yang sama** "
             "untuk ketiga algoritma. Subset proporsi bersarang dengan seed tetap. "
             "Waktu merupakan median pengulangan `fit_predict` (preprocessing dan evaluasi tidak dihitung).")
    if not ratios:
        st.warning("Pilih minimal satu proporsi pada panel kiri.")
    if cap > 2000:
        st.warning("Batas pelanggan cukup tinggi. DBSCAN/Agglomerative memiliki risiko waktu dan memori "
                   "kuadratik; gunakan 1.200–2.000 bila perangkat terbatas.")
    if st.button("▶ Jalankan benchmark", type="primary", disabled=not bool(ratios), use_container_width=True):
        try:
            with st.spinner("Eksperimen berjalan di sesi Streamlit ini..."):
                output = run_experiment(
                    rfm, cap, ratios, params, scaler, quantile_pct / 100, do_memory
                )
            output["signature"] = signature
            output["metadata"]["sumber_data"] = audit
            output["metadata"]["input_sha256"] = input_hash
            st.session_state["hasil_klaster"] = output
        except Exception as exc:
            st.error(f"Eksperimen terhenti: {type(exc).__name__}: {exc}")
            st.exception(exc)

output = st.session_state.get("hasil_klaster")
ready = bool(output is not None and output.get("signature") == signature)
if output is not None and not ready:
    with tab_experiment:
        st.warning("Parameter/sumber data telah berubah. Jalankan benchmark lagi agar hasil sesuai pengaturan saat ini.")

if ready:
    summary = output["summary"]
    valid = summary[summary["Status"] == "OK"].copy()
    with tab_experiment:
        if valid.empty:
            st.error("Seluruh percobaan gagal. Periksa parameter dan berkas.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Pengujian berhasil", f"{len(valid)} / {len(summary)}")
            c2.metric("Ulangan waktu/sel", str(repeats))
            c3.metric("Skor internal tersedia", str(int(valid["valid_internal"].sum())))
            st.markdown("**Hasil utama** — silhouette lebih besar, Davies–Bouldin lebih kecil, "
                        "Calinski–Harabasz lebih besar. Skor DBSCAN menggunakan data non-noise.")
            cols = ["Proporsi", "Pelanggan", "Algoritma", "jumlah_klaster", "jumlah_noise",
                    "cakupan_persen", "silhouette", "davies_bouldin", "calinski_harabasz",
                    "median_detik", "cv_persen", "python_peak_mib", "rss_delta_peak_mib"]
            st.dataframe(valid[cols].round(4), use_container_width=True, hide_index=True)
            colA, colB = st.columns(2)
            with colA:
                f1 = px.bar(valid, x="Proporsi", y="median_detik", color="Algoritma", barmode="group",
                            category_orders={"Proporsi": [f"{p}%" for p in sorted(ratios)]},
                            title="Median waktu fit_predict (detik)", text_auto=".3f")
                st.plotly_chart(f1, use_container_width=True)
            with colB:
                f2 = px.line(valid, x="Pelanggan", y="median_detik", color="Algoritma", markers=True,
                             title="Skalabilitas empiris terhadap jumlah pelanggan")
                st.plotly_chart(f2, use_container_width=True)
            if do_memory:
                colC, colD = st.columns(2)
                with colC:
                    fm = px.bar(valid, x="Proporsi", y="python_peak_mib", color="Algoritma",
                                barmode="group", title="Peak alokasi Python (tracemalloc, MiB)")
                    st.plotly_chart(fm, use_container_width=True)
                with colD:
                    fr = px.bar(valid, x="Proporsi", y="rss_delta_peak_mib", color="Algoritma",
                                barmode="group", title="Perubahan peak RSS tersampel (MiB)")
                    st.plotly_chart(fr, use_container_width=True)
                st.caption("Peak tracemalloc bukan total RAM. Sampel RSS adalah pengamatan periodik "
                           "dalam proses Streamlit yang sama, bukan puncak memori native yang pasti atau hasil terisolasi.")
            st.markdown("**Kualitas dan penanganan noise**")
            colE, colF = st.columns(2)
            with colE:
                fq = px.bar(valid, x="Proporsi", y="silhouette", color="Algoritma", barmode="group",
                            title="Silhouette non-noise (jika terdefinisi)")
                st.plotly_chart(fq, use_container_width=True)
            with colF:
                fn = px.bar(valid, x="Proporsi", y="persen_noise", color="Algoritma", barmode="group",
                            title="Proporsi noise (% semua titik)")
                st.plotly_chart(fn, use_container_width=True)
            with st.expander("Lihat detail setiap pengukuran dan catatan validasi"):
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.dataframe(output["raw"], use_container_width=True, hide_index=True)

    with tab_visual:
        st.subheader("Eksplorasi klaster dan profil bisnis")
        ratios_available = sorted(output["indices"])
        r = st.selectbox("Skenario", ratios_available, index=len(ratios_available)-1,
                         format_func=lambda v: f"{v}% dari batas bersama")
        available_algorithms = [a for a in ALGORITHMS if (r, a) in output["labels"]]
        if available_algorithms:
            a = st.selectbox("Algoritma", available_algorithms)
            mode = st.radio("Metode proyeksi 2D", ["PCA", "UMAP (opsional)"], horizontal=True)
            idx = output["indices"][r]
            sample = rfm.iloc[idx].reset_index(drop=True)
            X, _ = make_features(sample, scaler, quantile_pct / 100)
            labels = output["labels"][(r, a)]
            figure = plot_clusters(X, labels, sample, mode, seed)
            if figure is not None:
                st.plotly_chart(figure[0], use_container_width=True)
                st.caption(figure[1])
            details = cluster_details(sample, labels)
            st.markdown("**Profil per klaster (median dalam satuan asli)**")
            st.dataframe(details.round(2), use_container_width=True, hide_index=True)
            fig_size = px.bar(details, x="Klaster", y="Pelanggan", title="Ukuran tiap klaster",
                              text_auto=True)
            st.plotly_chart(fig_size, use_container_width=True)
            st.caption("Noise DBSCAN (label -1) dijelaskan terpisah dan tidak dimasukkan "
                       "ke skor silhouette/DB/CH. Nomor klaster tidak menunjukkan peringkat.")
            export_labels = sample.copy()
            export_labels["Klaster"] = labels
            st.download_button("Unduh label pelanggan per skenario (CSV)",
                               data=export_labels.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"label_{a.lower().replace(' ', '_')}_{r}persen.csv", mime="text/csv")
        else:
            st.warning("Tidak ada hasil model yang berhasil untuk skenario terpilih.")

    with tab_method:
        st.subheader("Unduh hasil eksperimen")
        st.download_button("Unduh ringkasan pengujian (CSV)",
                           data=summary.to_csv(index=False).encode("utf-8-sig"),
                           file_name="ringkasan_clustering.csv", mime="text/csv")
        st.download_button("Unduh pengulangan waktu mentah (CSV)",
                           data=output["raw"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="waktu_mentah.csv", mime="text/csv")
        st.download_button("Unduh profil seluruh klaster (CSV)",
                           data=output["profiles"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="profil_klaster.csv", mime="text/csv")
        st.download_button("Unduh metadata eksperimen (JSON)",
                           data=json.dumps(output["metadata"], indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name="metadata_eksperimen.json", mime="application/json")
        st.json(output["metadata"], expanded=False)
else:
    with tab_visual:
        st.info("Jalankan benchmark pada tab **Benchmark & Evaluasi** untuk melihat plot dan profil klaster.")

with tab_method:
    st.markdown("""
### Keputusan metodologis dan keterbatasan

1. **Unit analisis:** CustomerID (pelanggan), bukan itemset, StockCode, atau invoice.
   Satu invoice dapat memiliki banyak baris barang; `Frequency` menghitung invoice unik.
2. **Pembersihan transaksi:** invoice berawalan `C`, invoice/kode produk kosong,
   Quantity ≤ 0, dan UnitPrice ≤ 0 dikeluarkan. Baris tanpa CustomerID juga
   dikeluarkan **untuk clustering pelanggan**. Retur tidak diperhitungkan dalam Monetary.
3. **Fitur:** `Recency = (tanggal maksimum dataset + satu hari) - tanggal belanja terakhir`;
   `Frequency = jumlah invoice unik`; `Monetary = Σ(Quantity × UnitPrice)`.
   Ketiganya di-`log1p` lalu diskalakan. Opsional pemotongan atas kuantil sebelum log.
4. **Skenario adil:** 25%, 50%, 100% dari **batas pelanggan bersama**,
   subset bertingkat seed tetap. Semua algoritma menerima observasi serta fitur identik.
   Data sumber dapat lebih banyak dari batas, sehingga 100% bukan berarti 100% Excel.
5. **Kualitas klaster:** Silhouette (↑), Davies–Bouldin (↓), Calinski–Harabasz (↑)
   adalah metrik internal pada ruang RFM terskalakan. Tidak ada label kebenaran,
   sehingga akurasi/precision/recall tidak relevan.
6. **Noise DBSCAN:** label `-1` tidak masuk ketiga metrik kualitas; laporkan
   jumlah, persentase, dan cakupannya. Jika semua noise/hanya satu klaster,
   skor kualitas **tidak terdefinisi** dan dibiarkan kosong, bukan nol.
7. **Waktu:** `perf_counter` untuk tiap `fit_predict`, estimator baru per pengulangan;
   median utama dan CV, Q1/Q3 pelengkap. Pembacaan Excel, feature engineering,
   scaling, evaluasi, plotting, dan ekspor **di luar** stopwatch.
8. **Memori:** satu fit terpisah per kondisi. `tracemalloc` menghitung peak
   alokasi Python (bukan semua alokasi C/native). RSS dipantau periodik pada
   proses aplikasi yang sama: indikasi praktis, bukan peak eksak dan bukan
   eksperimen terisolasi. Jangan menjumlahkan keduanya.
9. **Mitigasi skala:** DBSCAN serta Agglomerative dapat mendekati O(n²) waktu/memori.
   Maksimum bersama 3.000 pelanggan (default 1.200), sampel silhouette maksimal
   1.200, dan pembuatan seluruh jarak pasangan tidak dilakukan secara eksplisit.
   Pengukuran skala merupakan observasi empiris, **bukan pembuktian Big-O**.
10. **Grafik PCA/UMAP:** hanya proyeksi untuk interpretasi. Metrik kualitas
    tidak dihitung dari koordinat 2D. Klaster pelanggan bukan aturan asosiasi.

**Yang bisa diadaptasi dari slide lama:** proporsi 25/50/100, seed, replikasi
waktu dan median, CV dan kuartil, pengukuran memori terpisah, grafik skala,
serta pelaporan lingkungan. **Yang tidak dibawa:** threshold support 3/5/8,
confidence, lift, frekuensi itemset, dan klaim kesetaraan output algoritma
Apriori/FP-Growth/Eclat.
""")
    st.caption("Kode, hasil audit, dataset mentah, CSV RFM siap pakai, dan referensi tugas tersedia dalam paket proyek ZIP.")
