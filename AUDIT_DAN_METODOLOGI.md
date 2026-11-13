# Audit sumber & desain eksperimen — Kelompok 2

## 1. Temuan dokumen: metode berbeda, tidak dapat disamakan

Gambar tugas menetapkan **Kelompok 2: Clustering**, dengan tiga algoritma **K-Means, DBSCAN, Agglomerative Clustering**, dan dataset disarankan **UCI Online Retail**. Sebaliknya PowerPoint `20261108 - Kelompok 2_Association Rule Mining(5).pptx` berjudul *Perbandingan Apriori, FP-Growth, dan Eclat* dan merumuskan target **frequent itemsets** dan **association rules**. Ini *dua masalah analitis berbeda*.

| Aspek | PowerPoint asal (asosiasi) | Proyek ini (clustering) |
|---|---|---|
| Unit analisis | Invoice → kumpulan StockCode | CustomerID → vektor Recency, Frequency, Monetary |
| Algoritma | Apriori, FP-Growth, Eclat | K-Means, DBSCAN, Agglomerative Clustering |
| Output | Itemset, support count, confidence, lift | Label klaster, label noise, profil RFM |
| Parameter | Minimum support 3%, 5%, 8%, panjang ≤3, confidence 30% | k, eps, min_samples, scaling, log1p, quantile |
| Validasi | Kesetaraan himpunan itemset dan count | Silhouette, Davies–Bouldin, Calinski–Harabasz, cakupan noise, karakteristik klaster |
| Waktu | Representasi transaksi + frequent itemset mining | `fit_predict` estimator clustering tanpa persiapan fitur |

**Angka benchmark lama (Apriori/Eclat/FP-Growth) tidak digunakan sebagai baseline waktu maupun klaim kecepatan K-Means/DBSCAN/Agglomerative.**

## 2. Audit Excel yang benar-benar diunggah

Berkas: `Online Retail(5).xlsx` (disimpan sebagai `data/Online_Retail.xlsx` dalam proyek), SHA-256: `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.

| Pengukuran | Hasil audit |
|---|---:|
| Baris asli (di luar header) | **541.909** |
| Baris lolos filter dasar transaksi (pembersihan asosiasi slide) | **530.104** |
| Invoice unik dari filter dasar | **19.960** |
| StockCode unik dari filter dasar | **3.922** |
| Baris valid dasar tanpa CustomerID | **132.220** |
| Baris valid dengan CustomerID dan tanggal untuk RFM | **397.884** |
| Invoice unik valid untuk RFM | **18.532** |
| CustomerID unik → sampel pelanggan | **4.338** |
| Baris invoice awalan `C` (sebelum filter) | 9.288 |
| Baris Quantity ≤0 | 10.624 |
| Baris UnitPrice ≤0 | 2.517 |
| Waktu transaksi RFM | 1 Desember 2010 – 9 Desember 2011 |
| Tanggal referensi Recency | **10 Desember 2011** |

Hitungan kategori pengecualian **tidak saling lepas** dan tidak boleh dijumlahkan menjadi jumlah baris terbuang. Audit dilakukan dengan membaca isi XML berkas XLSX secara streaming (`scripts/audit_dataset.py`), lalu **dicocokkan dengan jalur `pandas.read_excel` di aplikasi**: jumlah baris, invoice, pelanggan, dan nilai RFM per pelanggan konsisten.

**Perbedaan 530.104 vs 397.884 bukan error.** Angka pertama masih memuat pembelian tanpa CustomerID sehingga cocok untuk tujuan asosiasi berbasis invoice; angka kedua membuang data anonim karena objek clustering adalah pelanggan.

## 3. Desain eksperimen yang dapat dipertanggungjawabkan

**Fitur:** Recency (hari sejak kunjungan terakhir ke tanggal referensi +1), Frequency (invoice unik), Monetary (total harga × jumlah barang yang lolos filter). Dilakukan `log1p`, scaler Standard atau Robust. Pilihan cap kuantil 100%, 99%, 95%; default 100% (tanpa pemotongan).

**Data:** sampel acak bersarang (seed default 42). Default cap global 1.200 pelanggan dari 4.338; tiga proporsi menghasilkan 300, 600, dan 1.200 pelanggan. Ini *bukan* 25/50/100% dari 19.960 invoice seperti slide; seluruh algoritma menggunakan subsampel identik pada setiap skenario. Dengan cap maksimum 3.000, penggunaan waktu/memori DBSCAN dan Agglomerative dibatasi; namun bukan jaminan terhadap OOM pada mesin sangat terbatas.

**Model dan parameter default:** K-Means k=4, `n_init=10`; Agglomerative Ward k=4; DBSCAN `eps=0,50`, `min_samples=8`, `n_jobs=1`. Kedua algoritma dengan k menghasilkan empat klaster (kecuali data tidak memadai), sedangkan DBSCAN dapat menemukan 0, 1, atau banyak klaster dan noise. Konfigurasi ini adalah titik awal pembelajaran, **belum optimasi hyperparameter**.

**Metrik internal** dihitung pada matriks 3 fitur yang telah diskalakan. Jika noise=-1, evaluasi DBSCAN hanya pada data non-noise dengan coverage dilaporkan terpisah; hati-hati membandingkan silhouette DBSCAN dengan algoritma tanpa noise. Silhouette dibatasi 1.200 titik dengan sampling terstratifikasi untuk menjaga representasi klaster; Davies–Bouldin dan Calinski–Harabasz memakai seluruh titik non-noise. Kondisi tak memenuhi syarat tidak diubah menjadi skor nol atau klaim buruk secara otomatis.

**Benchmark:** jam monotonic `time.perf_counter`, objek estimator baru untuk setiap ulangan. Metrik utama median detik. Rata-rata, standar deviasi sampel, kuartil, dan CV tersedia. Preprocessing, evaluasi kualitas, proyeksi PCA, dan visualisasi di luar stopwatch. Memori diukur dengan satu fit tambahan: `tracemalloc` Python peak dan `psutil` RSS delta peak tersampel; **bukan** total native memory eksak, tidak terisolasi per proses, dan bukan 10 pengukuran memori independen.

**Skalabilitas:** durasi median dapat diplot terhadap jumlah pelanggan. Dengan tiga ukuran sampel, jangan menarik kesimpulan formal Big-O. Hasil juga berbeda jika CPU, sistem operasi, versi scikit-learn, seed, eps, dan tingkat noise berubah.

**Validasi teknis:** tes sintetis mencakup penghapusan baris tak valid, invoice unik sebagai Frequency, RFM, nested sampling, label noise, kondisi kualitas tidak terdefinisi, profil klaster, dan ketiga algoritma. Output per pelanggan dari Excel penuh telah dicocokkan dengan CSV RFM bawaan.

## 4. Relasi terhadap slide yang diunggah

| Referensi slide | Status dalam aplikasi |
|---|---|
| **Slide 15**, satu invoice per transaksi; valid 530.104 baris | **Diverifikasi**; berbeda dari satu pelanggan per baris RFM |
| **Slide 16**, sembilan skenario 25/50/100 × support 3/5/8 | **Sebagian diadopsi:** ukuran sampel 25/50/100; *support* tidak bermakna untuk clustering |
| **Slide 17**, 10 replikasi median dan puncak tracemalloc | **Diadaptasi:** 1–10 replikasi (3 default), metrik deskriptif, satu fit memori terpisah, RSS sampled tambahan |
| **Slide 19–22**, grafik time/memory/scale dan CV | **Diadaptasi:** grafik berdasarkan n pelanggan, median waktu, CV, metrik memori berlabel jelas |
| **Slide 23**, validasi output tiga frequent itemset miner identik | **Tidak bisa ditransfer:** keluaran clustering berbeda; validasi internal, kasus sintetis dan karakteristik noise |
| **Slide 24–31**, lift/confidence/pasangan item | **Tidak digunakan:** topik asosiasi, bukan validitas clustering |

## 5. Hal yang belum dibuktikan oleh proyek

- Tidak ada ground truth segmen pelanggan, jadi tidak bisa mengklaim akurasi klasifikasi.
- Skor internal bukan bukti kualitas bisnis, signifikansi statistik, maupun stabilitas antarperiode/antarseed.
- Tidak ada adjustment pengembalian barang/retur dalam Monetary bruto.
- Tidak ada eksplorasi hiperparameter luas maupun uji beda statistik antaralgoritma; parameter yang berbeda dapat mengubah ranking.
- Benchmark UI berjalan dalam satu proses Streamlit; untuk klaim kinerja akademik formal, ulangi dengan isolasi proses, pengacakan urutan, dan pencatatan perangkat.
- Dataset asli adalah dataset historis 2010–2011 dan tidak mencerminkan kondisi pasar 2026.

## 6. Bahan sumber

- Gambar tugas Kelompok 2: `referensi/instruksi_kelompok_2.png`.
- PowerPoint yang diunggah (18 halaman): `referensi/slide_association_rule_mining_asli.pptx`.
- Dataset yang diunggah: `data/Online_Retail.xlsx`.
- Halaman sumber dataset: https://archive.ics.uci.edu/dataset/352/online+retail
