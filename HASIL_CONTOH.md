# Contoh hasil benchmark — bukan hasil pada PowerPoint

Folder `contoh_hasil/` berisi keluaran **yang benar-benar dihitung** dari CSV RFM dataset unggahan (bukan angka contoh rekaan atau hasil Apriori/FP-Growth/Eclat): 1.200 pelanggan maksimum, subset bertingkat 300/600/1.200, seed 42, `log1p` + StandardScaler, k=4, DBSCAN eps=0,50, min_samples=8, **3 ulangan** per model dan kondisi; memori diukur satu fit tambahan.

| Pelanggan | Metode | Klaster | Noise | Silhouette (non-noise) |
|---:|---|---:|---:|---:|
| 300 | K-Means | 4 | 0,0% | 0,348 |
| 300 | DBSCAN | 3 | 24,0% | 0,276 |
| 300 | Agglomerative | 4 | 0,0% | 0,306 |
| 600 | K-Means | 4 | 0,0% | 0,341 |
| 600 | DBSCAN | 3 | 11,0% | 0,244 |
| 600 | Agglomerative | 4 | 0,0% | 0,282 |
| 1.200 | K-Means | 4 | 0,0% | 0,349 |
| 1.200 | DBSCAN | 2 | 4,5% | 0,311 |
| 1.200 | Agglomerative | 4 | 0,0% | 0,290 |

**Cara membaca:** pada pilihan parameter dan sampel ini, Silhouette K-Means paling tinggi untuk ukuran 1.200. Ini **bukan** bukti bahwa K-Means selalu lebih baik: jumlah klaster K-Means/Agglomerative dipaksa menjadi 4, DBSCAN menemukan 2 klaster dengan noise 4,5%, dan metrik DBSCAN **tidak** mengikutsertakan titik noise. Skor tidak boleh dilepaskan dari cakupan, jumlah klaster, tujuan segmentasi, dan kecocokan hyperparameter.

Persentase noise DBSCAN mengecil ketika n bertambah dalam eksperimen ini. Hal tersebut konsisten dengan sifat pendekatan berbasis kepadatan, tetapi tiga titik data belum cukup untuk menyimpulkan hubungan kausal umum.

Durasi dan memori lingkungan uji tidak perlu ditiru pada mesin lain. Data mentah per ulangan tersedia dalam `contoh_hasil/ulangan_waktu.csv`, ringkasan termasuk durasi median/CV serta dua metrik memori dalam `ringkasan.csv`, profil klaster dalam `profil_klaster.csv`, dan lingkungan pelaksanaan dalam `metadata.json`.

**Keterbatasan uji:** pengulangan hanya tiga kali, urutan algoritma belum diacak, pengukuran RAM tidak terisolasi, metode yang dibandingkan tidak mengoptimalkan objektif sama, dan tidak ada label kebenaran. Untuk laporan akademik, jalankan ulang pada perangkat presentasi dengan minimal 10 pengulangan bila waktu cukup; lakukan interpretasi dan dokumentasikan seluruh parameter.
