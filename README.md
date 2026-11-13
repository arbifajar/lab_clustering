# Kelompok 2 — Topik 3: Association Rule Mining

**Judul tugas:** *Association Rule Mining: Penemuan Pola Asosiasi Transaksi*  
**Aplikasi:** Streamlit untuk perbandingan **Apriori, FP-Growth, dan Eclat** pada **UCI Online Retail**.  
**Status:** pengganti proyek clustering/RFM sebelumnya, sesuai penugasan kelompok dan PowerPoint *Perbandingan Apriori, FP-Growth, dan Eclat*.

## 1. Apa yang berubah?

| Sebelum (keliru) | Versi ini (sesuai penugasan) |
|---|---|
| Clustering pelanggan berbasis RFM | Frequent itemset mining dan aturan asosiasi antarbarang |
| K-Means, DBSCAN, Agglomerative | Apriori, FP-Growth, Eclat |
| Baris analisis: pelanggan/CustomerID | Baris analisis: **InvoiceNo (basket)** |
| Evaluasi Silhouette, Davies–Bouldin | **Validasi itemset/count**, support, confidence, lift |
| PCA dan klaster | Grafik benchmark, daftar frequent itemset, pola aturan |

**PowerPoint awal sekarang memang topik yang benar.** Tidak perlu mempresentasikan aplikasi clustering lama.

## 2. Menjalankan aplikasi

Gunakan Python 3.10–3.13 (disarankan 3.11/3.12). Dari folder hasil ekstraksi:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` jika browser tidak otomatis terbuka.

**Tidak perlu menjalankan ulang preprocessing**: paket sudah memuat `data/transactions.csv.gz` (~1,1 MB), `data/products.csv`, dan `data/audit.json`. Data Excel sumber asli (`data/Online_Retail.xlsx`, ~23 MB) disertakan agar eksperimen dapat diaudit. Sumber PowerPoint juga disertakan.

### Menayangkan versi baru pada Streamlit Cloud

Aplikasi yang saat ini berada di URL `https://labclustering-srtqddxud5wuxac3kmw7xz.streamlit.app/` **tidak otomatis berubah** hanya karena ZIP ini telah dibuat. Untuk memperbaruinya:

1. Salin berkas proyek versi baru ke repositori GitHub yang terhubung dengan aplikasi Streamlit Cloud (sebaiknya buat commit/cabang baru atau cadangkan versi clustering terlebih dahulu).
2. Pastikan `app.py`, `requirements.txt`, `src/`, `data/transactions.csv.gz`, `data/products.csv`, dan `data/audit.json` ikut ter-commit. Jangan memblokir file `.gz` di `.gitignore`.
3. Di Streamlit Cloud, pilih aplikasi yang ada lalu *Manage app* → pengaturan repository/branch dan jalur utama `app.py`, atau buat aplikasi baru jika ingin mempertahankan yang lama.
4. *Reboot* atau *redeploy* sesuai antarmuka Cloud, buka halaman dan pastikan judul **Lab Association Rule Mining** tampil.
5. Lakukan uji skenario cepat lalu skenario 100%, support 3% (atau penuh) untuk memverifikasi aturan slide.

Saran: `data/Online_Retail.xlsx` dan file PowerPoint **boleh tidak dimasukkan** ke repositori deployment untuk menghemat bandwidth karena tidak digunakan saat menjalankan Streamlit. Namun, pertahankan dalam arsip ZIP penelitian demi keterlacakan. Jika tidak memasukkan Excel pada repo, audit SHA-256 yang sudah tersimpan di `data/audit.json` tetap menjelaskan identitas sumber.

## 3. Cara memakai lima tab

1. **Dataset & Pembersihan.** Memperlihatkan angka audit, contoh basket, frekuensi StockCode, aturan filter, SHA sumber, serta unduhan basket pra-pemrosesan.
2. **Benchmark.** Pilih mode pada sidebar, klik **Jalankan benchmark**; lihat median waktu, Q1–Q3, CV, grafik skalabilitas, tracemalloc, dan RSS. Variasi waktu hanya dihitung dari ulangan; memori satu kali terpisah.
3. **Itemset & Validasi.** Pilih kombinasi ukuran data dan support; hasil tiga algoritma **harus identik dalam itemset dan support count**. Lihat 1-item, 2-item, 3-item dan unduh CSV.
4. **Aturan Asosiasi.** Pilih skenario, confidence minimal (default 30%), aturan pasangan 1→1 seperti slide, dan lift minimum. Grafik support–confidence serta daftar aturan dapat diunduh. Untuk memverifikasi 18 aturan, pilih **100% / support 3%**.
5. **Metodologi & Unduhan.** Rumus, rancangan eksperimen, FAQ, angka referensi slide yang dilabeli sebagai *referensi* (bukan hasil benchmark baru), dan paket hasil CSV + JSON metadata.

### Mode yang dianjurkan

- **Demonstrasi cepat:** 25% data (4.990 invoice), support 3%, L=3, seed 42, tiga algoritma, tiga ulangan; sudah bisa menunjukkan aturan.
- **Replikasi 9 skenario slide:** kombinasi 25/50/100% × support 3/5/8%, L=3, seed 42, 10 ulangan; menghasilkan **270 durasi individual** serta 27 pengukuran memori jika dicentang.
- **Atur sendiri:** pilihan beberapa proporsi/support dan jumlah ulangan.

### Perhatian bila browser atau Cloud lambat

- Jalankan demo cepat lebih dahulu; hindari menjalankan beberapa benchmark penuh secara simultan dari browser rekan kelompok.
- Mematikan pengukuran memori mengurangi beban karena instrumentation `tracemalloc` membuat proses lebih lambat. Pada mode 9 skenario opsi ini **default tidak aktif**; aktifkan secara eksplisit jika ingin memori pada semua kondisi.
- Gunakan 3 ulangan untuk gladi, 10 ulangan untuk laporan yang ingin konsisten dengan skema PPT.
- Jangan menyegarkan browser ketika proses sedang berlangsung; hasil berada di `st.session_state` tab/browser sehingga hilang saat sesi direstart.

## 4. Dataset dan langkah pembersihan

Berasal dari [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail). Data mentah memiliki 541.909 baris item barang, kurun 2010-12-01 s.d. 2011-12-09. Pembersihan:

- Hapus InvoiceNo kosong, invoice pembatalan (awalan `C`), StockCode kosong.
- Hapus baris dengan `Quantity ≤ 0` atau `UnitPrice ≤ 0`.
- **Pertahankan transaksi tanpa CustomerID** karena invoice sudah cukup sebagai unit analisis.
- Ubah tiap invoice menjadi **set StockCode unik** (satu barang muncul dua kali dalam invoice = satu kehadiran).
- Urutkan invoice secara konsisten; pilih subset dengan `random.Random(42).shuffle` sehingga 25% ⊂ 50% ⊂ 100%.

Hasil audit sumber:

| Tahap | Jumlah |
|---|---:|
| Baris mentah | 541.909 |
| Baris valid | 530.104 |
| Invoice unik | 19.960 |
| StockCode unik | 3.922 |

Regenerasi basket dari Excel (opsional):

```bash
python scripts/build_transactions.py
```

Script memakai parsing XLSX streaming `lxml` untuk menjaga pemakaian RAM. **Perhatian:** menjalankan ulang akan menimpa tiga file hasil praproses. Bandingkan `data/audit.json` sebelum/sesudah jika Anda melakukan perubahan filter.

## 5. Parameter eksperimen dan evaluasi

Pada setiap kombinasi input:

- **Support:** persentase invoice yang harus memuat semua barang dalam itemset. `min_count = ceil(s * n)`.
- **L = 3:** hanya itemset berukuran 1, 2, 3 produk.
- **Confidence min 30%:** filter *setelah* frequent itemset diperoleh. Tidak memengaruhi waktu mining.
- **Aturan 1→1:** dibatasi pada pasangan produk agar sebanding dengan tabel pasangan pada slide.
- **Waktu:** `time.perf_counter`; run ulang yang berbeda, **median** utama; menambahkan Q1, Q3, rata-rata, standar deviasi, CV.
- **Memori:** `tracemalloc` *peak* sekali per kondisi/algoritma di run **terpisah**, bukan total RAM atau 10 observasi memori independen. Delta RSS tidak menunjukkan peak.
- **Validasi:** membandingkan **seluruh dict `{itemset: count}`**, bukan hanya jumlah itemset. Pengujian unit menggunakan oracle enumerasi brute-force untuk dataset kecil.
- **Skalabilitas:** waktu median vs jumlah invoice pada tiga ukuran; bukan pembuktian Big-O.

### Perbedaan versi ini dengan implementasi pada slide

Aplikasi ini memakai **implementasi independen berbasis Python** (Apriori kandidat level-wise; FP-Growth FP-tree dengan conditional mining; Eclat vertical bitset) yang seluruhnya dibuat untuk aplikasi ini. **Implementasi benchmark PPT lama tidak disertakan.** Oleh karena itu, jangan menyatakan bahwa angka waktu dan memori yang muncul adalah reproduksi angka PPT atau berasal dari komputer Windows 11 yang sama. Kesamaan output itemset pada data dan pengaturan yang sama telah dibuktikan; kecepatan dan alokasi memori bisa berbeda.

**Yang sama dengan PPT:** dataset dan filter; urutan invoice; `random.Random(seed=42)`; sembilan ukuran/ambang; min count; L=3; aturan 1→1 confidence 30%. Hasil itemset pada sembilan kondisi dan 18 aturan pada data penuh support 3% cocok persis dengan tabel referensi slide.

## 6. Rumus yang dipresentasikan

Untuk total N invoice:

- `support(A) = invoice_yang_memuat_A / N`
- `confidence(A → B) = support(A∪B) / support(A)`
- `lift(A → B) = confidence(A → B) / support(B)`

**Contoh slide pada data penuh 3%:** `22697` (cangkir Regency hijau) dan `22698` (merah muda) muncul bersama pada **632 invoice**; support **3,17%**; confidence `22697→22698` **62,39%**, arah balik **82,61%**; lift **16,28**. Confidence bergantung arah; support dan lift simetris. Lift tinggi **tidak** menunjukkan sebab-akibat atau profit.

## 7. Pengujian

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tes mencakup tiga algoritma melawan brute-force pada dataset sintetis, satu set kecil terkontrol, rumus aturan, sampling bersarang, seluruh **9 jumlah itemset acuan**, hitungan 632 invoice pasangan produk contoh, 18 aturan 1→1, serta benchmark kecil.

## 8. Struktur proyek

```
app.py                        # Antarmuka Streamlit lima tab
requirements.txt              # Dependensi produksi
requirements-dev.txt          # Dependensi pengujian
.streamlit/config.toml        # Tema aplikasi
src/mining.py                 # Apriori / FP-Growth / Eclat
src/rules.py                  # Support / confidence / lift
src/benchmark.py              # Ulangan waktu, peak memori, validasi
src/data.py                   # Loader basket dan sampling
scripts/build_transactions.py # Preprocessing Excel streaming
scripts/benchmark_cli.py      # Opsional: jalankan benchmark tanpa UI
data/transactions.csv.gz      # Dataset basket siap pakai
```

Berkas sumber `data/Online_Retail.xlsx`, `data/Slide_Acuan_Association_Rule_Mining.pptx`, `data/audit.json`, `data/products.csv`, folder `contoh_hasil/` (9 skenario, 3 ulangan waktu, memori tidak diukur), dan `tests/` juga disertakan.

## 9. Rujukan utama

- Slide kelompok yang diunggah: *Perbandingan Apriori, FP-Growth, dan Eclat — Menemukan Pola Belanja pada Dataset UCI Online Retail*.
- UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail
- R. Agrawal & R. Srikant, *Fast Algorithms for Mining Association Rules*, 1994: https://www.vldb.org/conf/1994/P487.PDF
- Han dkk., *Mining Frequent Patterns without Candidate Generation*: https://hanj.cs.illinois.edu/pdf/dami04_fptree.pdf
- M. Zaki, *Scalable Algorithms for Association Mining*: https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2000%20-%20IEEETKDE%20-%20Zaki%20-%20ScalableAlgorithms%20for%20Association%20Mining%20.pdf

> **Catatan untuk dosen:** kesetaraan output itemset tidak sama dengan kesetaraan durasi. Waktu dan memori harus dibahas sebagai hasil spesifik implementasi dan lingkungan saat eksperimen dilakukan.
