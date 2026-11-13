# Hasil benchmark contoh pada lingkungan pembuatan proyek

Data dalam folder ini **hasil eksekusi nyata**, bukan salinan angka waktu dari slide.

- Seluruh sembilan skenario dari PPT: proporsi 25%, 50%, 100% × minimum support 3%, 5%, 8%.
- Setiap skenario menguji Apriori, FP-Growth, Eclat.
- **3 ulangan waktu**, bukan 10 seperti sesi utama PowerPoint. Total 81 observasi waktu.
- **Pengukuran memori dinonaktifkan** untuk run contoh ini, sehingga kolom memori kosong (`NaN`). Bukan berarti pemakaian memorinya nol.
- Itemset dan count ketiga algoritma cocok pada seluruh skenario, jumlah itemset cocok pada tabel PowerPoint.
- Durasi dapat berbeda antar komputer/versi Python; **tidak boleh diklaim sebagai hasil Windows 11 dalam slide**.

Untuk mengukur memori pada sesi tersendiri, aktifkan kotak centang dalam aplikasi atau jalankan CLI tanpa `--no-memory`, misalnya `python scripts/benchmark_cli.py --repeats 3` untuk mode cepat.
