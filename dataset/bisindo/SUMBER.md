# Sumber Dataset BISINDO (Abjad A-Z)

- File `bisindo_features.csv` berasal dari repositori **realtime-bisindo-classification**
  oleh Krisna Santosa (lisensi MIT, lihat `LICENSE_realtime-bisindo-classification.txt`).
  https://github.com/KrisnaSantosa15/realtime-bisindo-classification
- Gambar aslinya: dataset Kaggle "Bahasa Isyarat Indonesia (BISINDO) Alphabets" (achmadnoer),
  312 foto (26 huruf x 12 foto), lalu di-augmentasi 50x per foto (flip, rotasi +-25 derajat,
  perubahan kecerahan, blur). Hasil akhirnya ~14.5 ribu baris.
- Format kolom: `label` + 126 angka = 2 tangan x 21 landmark x (x, y, z), koordinat MediaPipe
  yang dinormalisasi terhadap gambar. NaN pada blok tangan ke-2 berarti huruf itu hanya
  terdeteksi satu tangan. Nama kolom (hand_i_x/y/z) di file itu tidak sesuai urutan data
  sebenarnya; urutan data aslinya x,y,z per landmark. Kode kita membaca berdasarkan posisi.
- Catatan: semua foto berasal dari satu penanda (signer), jadi akurasi pada orang lain
  bisa lebih rendah daripada angka evaluasi.

## Dataset tambahan: foto huruf dari banyak orang
- "BISINDO Indonesian Sign Language: Alphabet Image Data", Samuel Ady Sanjaya, Universitas Multimedia Nusantara,
  Mendeley Data, V1, 18 Oktober 2024, DOI 10.17632/ywnjpbcz8m.1. Lisensi: CC BY 4.0 (wajib mencantumkan kredit).
  https://data.mendeley.com/datasets/ywnjpbcz8m
- Diubah menjadi fitur landmark oleh `ml/import_sanjaya_alphabet.py` -> `sanjaya_features.csv`. Fotonya sendiri tidak disimpan di proyek.

