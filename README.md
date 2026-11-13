# Lab Clustering Kelompok 2 — UCI Online Retail

Aplikasi **Streamlit berbahasa Indonesia** untuk menganalisis dan membandingkan **K-Means, DBSCAN, dan Agglomerative Clustering** pada segmentasi pelanggan menggunakan fitur **RFM** (Recency, Frequency, Monetary).

> **Beda tugas:** gambar instruksi Kelompok 2 berisi **clustering**, sedangkan PowerPoint yang diunggah berisi **association rule mining** (Apriori, FP-Growth, Eclat). Keduanya bukan algoritma sejenis. Hasil support, confidence, lift, dan frequent itemsets pada slide tidak ditafsirkan sebagai hasil clustering. Yang diadaptasi hanya desain eksperimennya: subset 25/50/100%, pengulangan waktu, median, profil memori, uji validitas, dan grafik skalabilitas.

## Jalankan aplikasi

**Prasyarat:** Python **3.11 atau 3.12**, minimal 4 GB RAM (disarankan 8 GB untuk Excel mentah), sistem Windows/macOS/Linux. Akses internet dibutuhkan **hanya sekali untuk mengunduh library** jika belum terpasang. Setelah itu aplikasi dan dataset bawaan bisa digunakan secara lokal tanpa internet.

### Windows PowerShell

```powershell
cd kelompok2_clustering_streamlit
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Jika `py -3.12` tidak tersedia, gunakan `python -m venv .venv`. PowerShell mungkin memerlukan pengaturan izin aktivasi; sebagai alternatif jalankan `.\.venv\Scripts\python.exe -m streamlit run app.py` tanpa mengaktifkan venv.

### Linux/macOS

```bash
cd kelompok2_clustering_streamlit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Buka URL lokal yang ditampilkan di terminal (umumnya `http://localhost:8501`). **Tidak perlu mengunggah dataset lagi:** aplikasi langsung membuka `data/customer_rfm.csv`, yang sudah dihitung dari berkas Excel asli dalam paket.

### Alur di antarmuka

1. Tab **Data & Audit:** cek jumlah baris dan statistik RFM. Pilih CSV bawaan, unggah `Online Retail.xlsx`, atau unggah CSV RFM sendiri.
2. Panel kiri **Pra-pemrosesan:** pilih scaler, pemotongan kuantil (opsional), dan seed acak.
3. Panel kiri **Eksperimen:** tentukan **batas pelanggan bersama** (awal 1.200; maksimal 3.000), proporsi 25/50/100%, k, eps, min_samples, jumlah pengulangan (1–10), warm-up, dan pengukuran memori.
4. Tab **Benchmark & Evaluasi:** klik **Jalankan benchmark**; lihat tabel, grafik kualitas, waktu, sebaran, noise, dan memori.
5. Tab **Visualisasi Klaster:** pilih proporsi dan algoritma; jelajahi grafik PCA dan profil RFM, lalu unduh label per pelanggan.
6. Tab **Metodologi & Unduhan:** baca keterbatasan dan unduh CSV pengujian, CSV timing mentah, profil klaster, dan JSON metadata.

**Opsional UMAP:** instal `pip install umap-learn`, lalu pilih UMAP di tab Visualisasi. PCA tidak membutuhkan paket tambahan.

## Inventaris proyek

```text
kelompok2_clustering_streamlit/
├── app.py                            # aplikasi Streamlit, UI dalam bahasa Indonesia
├── requirements.txt                   # dependency runtime
├── requirements-dev.txt               # dependency termasuk pytest
├── README.md                          # petunjuk ini
├── AUDIT_DAN_METODOLOGI.md            # audit data, slide, desain pengujian
├── HASIL_CONTOH.md                    # interpretasi contoh output benchmark
├── .streamlit/config.toml             # pengaturan tema dan kapasitas upload
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # filter transaksi, agregasi RFM, transformasi fitur
│   ├── benchmark.py                   # estimator, benchmark berulang, memori
│   └── evaluation.py                  # metrik tanpa ground truth, profil klaster
├── data/
│   ├── Online_Retail.xlsx             # dataset asli unggahan (541.909 baris)
│   ├── customer_rfm.csv               # fitur siap pakai (4.338 pelanggan)
│   └── audit.json                     # hasil audit dan checksum sumber
├── scripts/
│   ├── audit_dataset.py               # audit XLSX streaming dan regenerasi RFM
│   └── benchmark_cli.py               # benchmark tanpa antarmuka browser
├── contoh_hasil/                     # CSV+JSON benchmark contoh 1.200 pelanggan
├── tests/
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   ├── test_benchmark.py
│   └── test_precomputed.py
└── referensi/
    ├── instruksi_kelompok_2.png
    └── slide_association_rule_mining_asli.pptx
```

## Mengapa satu pelanggan, bukan satu invoice?

Dataset Online Retail merekam **baris produk per faktur**. Ada minimal tiga pilihan unit analisis yang tidak ekuivalen: produk, faktur, atau pelanggan. Proyek ini **secara eksplisit memilih pelanggan** karena tujuan segmentasi pembeli dan fitur RFM cukup transparan untuk tugas tiga algoritma clustering. Untuk dataset asli, `CustomerID` kosong sering ditemukan; baris itu dapat digunakan saat menghitung asosiasi produk, tetapi tidak bisa dipakai untuk membangun profil pelanggan secara benar.

**Aturan pembersihan:** hapus invoice yang diawali `C`, invoice/StockCode kosong, Quantity ≤ 0, UnitPrice ≤ 0, CustomerID kosong atau tanggal tidak valid. Untuk model pelanggan, baris return/cancel tidak dikompensasikan terhadap pembelian sebelumnya: Monetary menggambarkan **penjualan positif bruto**, **bukan** revenue bersih setelah retur, bukan laba, dan tidak mengandung margin. Satu invoice dengan banyak baris produk tetap dihitung **satu** frekuensi. Nilai negatif/zero tidak di-imputasi. Nilai di `CustomerID` dipakai sebagai ID, **bukan fitur numerik**.

**RFM:**

- `Recency = tanggal_ref − tanggal_pembelian_terakhir` dalam hari; tanggal_ref adalah tanggal transaksi terakhir dataset **+ 1 hari**.
- `Frequency = jumlah InvoiceNo unik` valid per CustomerID.
- `Monetary = Σ(Quantity × UnitPrice)` valid per CustomerID.

Fitur numerik diproses menggunakan `log1p` untuk mengurangi skew lalu `StandardScaler` (default) atau `RobustScaler`; pemotongan kuantil atas 99/95% opsional (default **tidak** dipotong). Transformasi **di-fit ulang per skenario** supaya tidak menggunakan distribusi sampel lebih besar untuk skenario kecil. Tidak memasukkan `CustomerID` sebagai fitur.

## Keberulangan pengujian dan keterbatasan

**Kontrol perbandingan:**

- Baseline 1.200 pelanggan dipilih dari total 4.338 dengan permutasi acak seed=42; subset 25%, 50%, dan 100% adalah awalan *permutasi yang sama*, sehingga saling bersarang.
- Satu skenario menggunakan **persis pelanggan dan representasi tiga fitur yang sama** untuk semua algoritma. Angka 100% di sini berarti **100% dari cap bersama** yang dipilih, **bukan** 100% dari 541.909 baris Excel.
- K-Means: `n_clusters=k`, `n_init=10`, `random_state=seed`. Agglomerative: `ward`, `n_clusters=k`. DBSCAN: `eps`/`min_samples` pada tiga fitur terskalakan, `n_jobs=1`. DBSCAN menentukan jumlah klaster sendiri dan boleh memberi label noise=-1.
- Pengukuran waktu adalah **fit_predict** model baru per pengulangan, mulai stopwatch setelah persiapan dataset. Laporan median, rata-rata, standar deviasi, Q1/Q3, dan CV. Hasil boleh berubah oleh CPU, beban perangkat, versi library, alokasi memori. Warm-up default nol dan opsional satu kali.
- **Memori:** satu run terpisah setelah pengukuran waktu. `tracemalloc` mengukur puncak alokasi *Python*; `psutil` memantau peningkatan peak RSS tersampel relatif sebelum fit. Dua besaran tersebut **tidak boleh dijumlahkan**. Proses tidak diisolasi per algoritma; RSS sampled bukan puncak yang dijamin dan bukan penggunaan RAM total dari suatu algoritma.
- **Silhouette** hanya untuk titik non-noise dan dibatasi maksimum 1.200 sampel dengan pemilihan terstratifikasi jika perlu. Davies–Bouldin dan Calinski–Harabasz pada seluruh titik non-noise. Bila kurang dari dua klaster atau tiap titik menjadi klaster sendiri, metrik tidak terdefinisi (kosong/NaN). **Skor DBSCAN tidak boleh dibandingkan tanpa mempertimbangkan jumlah noise/cakupan.**
- **PCA/UMAP hanya untuk tampilan**; metrik dihitung pada fitur asli tiga dimensi terskalakan. Tidak ada kelas target, sehingga accuracy/precision/recall dan pengukuran supervised lain tidak berlaku.
- Metrik internal tidak membuktikan kebermanfaatan bisnis; segmentasi perlu divalidasi secara substantif, misalnya interpretasi profil RFM dan evaluasi stabilitas pada periode lain.
- Skenario 25/50/100 menampilkan **skalabilitas empiris**, bukan estimasi kelas kompleksitas Big-O yang valid dari tiga titik.

**Mitigasi beban:** DBSCAN dan Agglomerative dapat memerlukan kerja/memori kuadratik pada banyak titik. Karena itu semua algoritma dibatasi **maksimal 3.000 pelanggan** (awal 1.200) agar benchmark tetap adil. Untuk penelitian formal, jalankan di proses terpisah, acak urutan algoritma, lakukan cukup banyak pengulangan, ukur RSS di luar proses (misalnya profiler/psrecord), simpan spesifikasi mesin, dan ulangi eksperimen lintas seed/hyperparameter.

## Pengujian & benchmark dari terminal

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Eksperimen cepat: setiap algoritma × proporsi 25/50/100
python scripts/benchmark_cli.py --limit 600 --repeats 2 --memory

# Eksperimen mendekati protokol slide (lebih berat)
python scripts/benchmark_cli.py --limit 1200 --repeats 10 --memory

# Regenerasi dataset siap pakai + audit dari XLSX asli
python scripts/audit_dataset.py data/Online_Retail.xlsx
```

Script CLI menulis `hasil/ringkasan.csv`, `hasil/ulangan_waktu.csv`, `hasil/profil_klaster.csv`, `hasil/metadata.json`. Paket juga memuat contoh pengujian 1.200 pelanggan (3 ulangan) di folder `contoh_hasil/`; nilai ini demonstrasi dari lingkungan uji, **bukan benchmark universal**.

## Relevansi terhadap PowerPoint dan sitasi data

- **Slide 15:** pembersihan faktur untuk asosiasi: 541.909 baris mentah → 530.104 baris valid → 19.960 invoice, 3.922 StockCode. Cocok dengan file Excel, tetapi **berbeda dari filter RFM** yang membuang pelanggan anonim.
- **Slide 16:** 25/50/100% *invoice* + ambang support 3/5/8%; yang diambil hanya pendekatan proporsi, bukan ambang support atau perhitungan itemset.
- **Slide 17, 19–22:** pengulangan pengukuran waktu, median, CV, memori, grafik skalabilitas, serta detail lingkungan diadaptasi dengan definisi biaya yang jelas.
- **Slide 23:** validasi kesetaraan **frequent itemsets** tidak bisa diulang untuk tiga algoritma clustering yang berbeda objektif. Proyek ini memakai validitas numerik metrik internal, pemeriksaan label, profil, dan pengujian fungsi.

Lihat [`AUDIT_DAN_METODOLOGI.md`](AUDIT_DAN_METODOLOGI.md) untuk tabel audit lengkap. Rujukan dataset publik: [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail). Dataset dan slide dalam ZIP disertakan sebagai **referensi pengguna**, tanpa mengklaim hak kepemilikan atas sumbernya.

## Kesalahan umum

- `ModuleNotFoundError: streamlit` → jalankan `python -m pip install -r requirements.txt` di environment yang sama.
- Aplikasi lambat saat unggah Excel → dataset berisi >500 ribu baris; gunakan opsi **Dataset RFM bawaan** untuk startup cepat.
- DBSCAN menghasilkan satu klaster atau semua noise → silhouette/DB/CH kosong **memang benar**; coba nilai `eps`/`min_samples` berbeda, lalu lihat persen noise.
- Nilai metrik berubah saat skala/quantile/seed diubah → normal: jarak antar pelanggan berubah; klik benchmark ulang.
- CPU/RAM terbatas → turunkan batas pelanggan menjadi 300–800, ulangan menjadi 1–3, atau nonaktifkan pengukuran memori.

**Proyek dapat dibuka dan digunakan tanpa mengganti kode.** Nilai hasil clustering dan waktu *tidak* di-hardcode: setiap tombol benchmark menghitung ulang berdasarkan pengaturan pengguna.
