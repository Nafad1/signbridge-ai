# SignBridge AI - Penerjemah Bahasa Isyarat Indonesia (BISINDO)

**Judul Proyek:** Pengembangan Website SignBridge AI sebagai Media Penerjemah Dua Arah Bahasa Isyarat Indonesia (BISINDO) Berbasis Artificial Intelligence untuk Mendukung Komunikasi Inklusif.

**Kelompok :** [Kelompok 7]
**Mata Kuliah:** [Metode Numerik]

## 📖 Tentang Proyek
SignBridge AI adalah website terintegrasi yang dirancang untuk membantu komunikasi antara pengguna bahasa isyarat dan orang yang belum memahami bahasa isyarat. Proyek ini memiliki dua fitur utama:
1. **Sign Language to Text:** Menerjemahkan gerakan tangan BISINDO menjadi teks menggunakan kamera dan Machine Learning.
2. **Text to Sign Language:** Menampilkan video/penjelasan gerakan BISINDO berdasarkan kata yang diketik.

## 🛠️ Teknologi yang Digunakan
- **Frontend:** HTML, CSS, JavaScript, Bootstrap 5
- **Backend:** Python, Flask
- **Computer Vision:** MediaPipe Hands (21 Landmark)
- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Database:** SQLite
- **Pengolahan Data:** Pandas, NumPy, Joblib

## 🧠 Cara Kerja AI (Alur Sistem)
1. Kamera menangkap gambar dan mengirimkannya ke backend Flask.
2. MediaPipe mendeteksi tangan dan mengekstrak 21 titik koordinat (relatif terhadap pergelangan tangan).
3. Koordinat tersebut dimasukkan ke dalam model Random Forest.
4. Model memprediksi kosakata BISINDO dan menampilkannya di layar.

## 🚀 Cara Menjalankan Proyek (Di Komputer Baru)
Jika Anda ingin menjalankan proyek ini di komputer lain, ikuti langkah berikut:

1. Buka terminal (PowerShell) dan masuk ke folder proyek.
2. Buat Virtual Environment: `python -m venv venv`
3. Aktifkan Virtual Environment: `.\venv\Scripts\activate`
4. Instal semua library: `pip install -r requirements.txt`
5. Inisialisasi Database (jika belum ada): `python database/init_db.py`
6. Jalankan server Flask: `python app.py`
7. Buka browser dan akses: `http://127.0.0.1:5000`

## 🆕 Pembaruan: Dataset BISINDO Abjad & Teks ke Isyarat
**1. Mode Huruf A-Z (Sign to Text).** Dataset `realtime-bisindo-classification` (Krisna Santosa, MIT) ditambahkan di `dataset/bisindo/`
(lihat `SUMBER.md`). Model baru `ml/model/alphabet_rf.pkl` mengenali 26 huruf dari 1 atau 2 tangan. Halaman Penerjemah punya pilihan mode: *Huruf A-Z* atau *Kata* (5 kosakata lama, tidak berubah).
Fitur dihitung di `ml/extract_features.py` (relatif ke pergelangan tangan, dinormalisasi ukuran telapak, urutan tangan kiri-ke-kanan) sehingga tidak bergantung posisi/jarak tangan.
Latih ulang: `python ml/train_alphabet.py`.

**2. Teks ke Isyarat.** Ketik kata/kalimat di halaman *Penerjemah Teks*: kata yang ada di kamus diisyaratkan langsung, kata lain dieja per huruf, lalu diputar berurutan (putar/jeda, kecepatan, klik huruf untuk melompat).
Tiap item memakai **video asli** bila file ada, kalau tidak memakai **animasi kerangka tangan** dari data landmark (`python ml/build_poses.py` membuat `static/data/poses.json`).
Penamaan video (huruf kecil): `static/videos/alphabet/a.mp4` ... `z.mp4`, dan `static/videos/words/halo.mp4`, `terima_kasih.mp4`, dst. Cukup taruh file, tanpa ubah kode.

**3. Klip video asli.** Dari *BISINDO Video Dataset* (lihat `dataset/bisindo_video/SUMBER.md`) sudah dipotong **34 klip**: huruf **A-Q**, angka **1-10**, dan hari **Senin-Minggu**, lengkap di `static/videos/` dan database. Huruf R-Z masih memakai animasi kerangka. Ketik "tujuh", "25", atau "senin" di Teks ke Isyarat untuk mencoba. Pemotongan bisa diulang dengan `python ml/import_videos.py --dataset "<folder>"` (butuh ffmpeg). **Urutan hari belum terverifikasi, mohon dicek di Kamus.**

**4. 32 kata dari WL-BISINDO** (BINUS University, **lisensi CC BY-NC 4.0: non-komersial, wajib kredit**; lihat `dataset/bisindo_video/SUMBER.md`). Kata seperti *air, belajar, makan, rumah, teman, apa, siapa, di mana, merah, pagi, malam* kini punya video asli dan masuk Kamus. Varian BISINDO Banten dan penandanya berbeda dari sumber lain. Pembuatan ulang: `python ml/import_wl_bisindo.py --source <folder> --signer 3 --sample 2`.

**Hasil model huruf:** akurasi 97.5% (split acak) dan 97.5% (holdout berurutan). Huruf paling lemah: M, N, B.
> Angka ini kemungkinan **terlalu optimis**: semua foto berasal dari satu penanda dan tiap foto diaugmentasi 50x. Uji langsung dengan kamera dan beberapa orang berbeda sebelum dilaporkan.

## 🧠 Melatih Kata Bergerak dari WL-BISINDO (langkah 1-2)
Dataset WL berisi 1.600 video (32 kata, 5 penanda). Karena isyaratnya bergerak, modelnya belajar dari **urutan frame**, bukan satu foto.
1. Ekstraksi landmark (sekali saja, bisa dilanjutkan jika terhenti): `python ml/extract_wl_landmarks.py --source C:\data\wl`
2. Latih + evaluasi: `python ml/train_wl_words.py` (pembagian resmi signer-independent: latih penanda 0-3, uji penanda 4).
Hasil: `ml/model/wl_words_rf.pkl` dan `wl_words_metrics.json`.

### Memakai di kamera (Penerjemah -> mode "Kata gerak WL")
Nyalakan kamera, pilih mode **Kata gerak WL**, tekan **Rekam kata**, tunggu hitungan 3-2-1, lalu lakukan satu isyarat selama sekitar 2,5 detik. Browser mengirim 15 frame sekaligus ke `/api/predict-clip`; hasilnya 3 tebakan teratas. Kode: `ml/predict.py` (`predict_clip`), `static/js/script.js`.
- Akurasi pada penanda baru (penanda 4): top-1 **76%**, top-3 **88%**; dengan 15 frame seperti rekaman browser: top-1 72%, top-3 88%. Kata lemah: *merah*, *berangkat*, *apa*, *kapan*.
- Ambang ukuran tangan sekarang terpisah: `MIN_PALM_SIZE = 0.06` (huruf & 5 kata) dan `WL_MIN_PALM_SIZE = 0.03` (mode WL dan ekstraksi video WL) di `config.py`.
- Uji tanpa kamera: `python tests/test_wl_clip.py`.
Uji alur: `python tests/test_wl_pipeline.py`. **Lisensi CC BY-NC 4.0 juga berlaku untuk model turunan (non-komersial).**

## 🎬 Kamus Berbasis Video
Semua entri Kamus dan Teks ke Isyarat kini diputar sebagai **video**. Urutan prioritas: (1) video asli di `static/videos/alphabet/` dan `static/videos/words/`, (2) klip **animasi kerangka tangan** di `static/videos/generated/` untuk entri yang belum punya video asli (huruf R-Z, HALO, YA, TIDAK). Klip kerangka diberi tanda di Kamus, dibuat dari data landmark (bukan rekaman penanda) dan dibuat ulang dengan `python ml/render_pose_videos.py`. **Untuk menggantinya dengan video asli, cukup taruh file ber-nama sama**, misalnya `static/videos/alphabet/r.mp4` atau `static/videos/words/halo.mp4`; tanpa ubah kode.

## 🎯 Pelacakan Tangan (diperbarui)
- **Deteksi per frame mandiri** (`static_image_mode=True`): tidak ada sisa pelacakan dari frame/pengguna lain.
- **Pemilihan tangan** (`ml/hand_filter.py`): tangan latar/orang lain (jauh lebih kecil), deteksi ganda, dan tangan terpotong bingkai tidak ikut diprediksi. Mode kata selalu memakai tangan terbesar (terdekat). Ambang di `config.py` (`MIN_PALM_SIZE`, `MIN_REL_PALM`, `DUPLICATE_IOU`, `EDGE_MARGIN`).
- **Penghalus hasil** (`static/js/tracking.js`): huruf baru tampil jika menang 3 dari 5 frame terakhir; hasil dikosongkan saat tangan hilang. Hasil lama tidak terbawa saat ganti mode.
- **Landmark ditampilkan di kamera**: hijau/kuning = tangan dipakai, abu-abu putus-putus = diabaikan.
- Gambar dikirim dengan rasio asli kamera (dulu dipaksa 640x480 sehingga bisa gepeng).
- Uji: `python tests/test_tracking.py` dan `node tests/test_smoother.js`.

## ⚠️ Batasan Sistem (Penting)
- **Kosakata Terbatas:** Mode Kata hanya mengenali 5 kata awal (HALO, TERIMA KASIH, MAAF, YA, TIDAK) sesuai dataset yang dilatih.
- **Gerakan Statis:** Model Random Forest versi awal ini lebih optimal untuk mengenali gerakan/pose tangan yang relatif diam. Gerakan yang sangat dinamis membutuhkan model LSTM (pengembangan lanjutan).
- **Video Belum Lengkap:** Huruf R-Z serta kata HALO, YA, TIDAK belum punya video; Teks ke Isyarat memakai animasi kerangka tangan sebagai cadangan (bisa tampak tertukar kiri-kanan karena dataset berisi gambar yang di-flip). Angka belum didukung. Deskripsi gerakan huruf di kamus masih netral.
- **Bukan Pengganti Juru Bahasa Isyarat:** Proyek ini adalah prototipe media bantu untuk perkuliahan, bukan pengganti penerjemah profesional.

## 📊 Hasil Pengujian AI
- **Akurasi Model:** [Tulis persentase akurasi dari terminal, misal: 100%]
- **Jumlah Kosakata:** 5 kata (dapat ditambah dengan merekam dataset baru).

---
*Dibuat untuk memenuhi tugas perkuliahan.*

## Menambah data huruf dari banyak orang (dataset Sanjaya, CC BY 4.0)
Model huruf lama hanya belajar dari satu penanda. Dataset "BISINDO Indonesian Sign Language: Alphabet Image Data" (Samuel Ady Sanjaya, Universitas Multimedia Nusantara, DOI 10.17632/ywnjpbcz8m.1) berisi sekitar 4.400 foto A-Z dari banyak orang dan HP.
1. Ekstrak zip "02. Resized Images", misalnya ke `C:\data`.
2. `python ml/import_sanjaya_alphabet.py --source "C:\data\02. Resized Images"` (hanya folder foto BERWARNA; "Binary Images" tidak bisa dipakai karena berupa topeng hitam-putih). Hasilnya `dataset/bisindo/sanjaya_features.csv` dan tabel gambar terpakai per huruf. Opsi `--lenient` menerima 1 tangan untuk huruf dua tangan (lebih banyak foto terpakai); `--dup-iou 0.85` mencegah tangan kedua yang bertumpuk (M, N, Q) dibuang sebagai "kembaran".
3. `python ml/train_alphabet.py` (otomatis memakai file itu; mencetak akurasi pada perangkat/sesi yang tidak ikut dilatih).
4. Restart `python app.py`.
Uji logika tanpa kamera: `python tests/test_sanjaya_import.py`. Kredit untuk laporan: Sanjaya, S. A. (2024), Mendeley Data, V1, doi:10.17632/ywnjpbcz8m.1, lisensi CC BY 4.0.

## Mengganti animasi kerangka tangan dengan video orang
Animasi kerangka hanya dipakai untuk isyarat yang belum punya video asli (saat ini kata HALO, YA, TIDAK; semua huruf A-Z memakai video dari dataset Trunojoyo, `python ml/import_trunojoyo_alphabet.py --source <folder> [<folder> ...]`). Rekam sendiri dengan HP (2-4 detik, latar polos, badan bagian atas dan tangan terlihat), beri nama sesuai isyaratnya (`R.mp4`, `halo.mov`, ...), lalu:
`python ml/import_my_videos.py --source "C:\rekaman"` (butuh ffmpeg, atau `pip install imageio-ffmpeg`). Opsi: `--start`, `--duration`, `--mirror`, `--force`, `--remove-skeleton`. Restart `python app.py` dan tekan Ctrl+F5.

## Data huruf dari video orang (Trunojoyo) untuk model huruf
Foto Sanjaya diambil dari atas, sedangkan webcam melihat orang dari depan. Video Trunojoyo lebih mirip webcam:
`python ml/import_trunojoyo_frames.py --source "<folder A-E>" "<folder F-K>" "<folder L-Q>" "<folder R-Z>"` (sekitar 15 menit; hasilnya `dataset/bisindo/trunojoyo_features.csv`), lalu `python ml/train_alphabet.py`.
Pelatihan mencetak akurasi pada dataset yang tidak ikut dilatih ("Dilatih TANPA ...") dan huruf yang paling sering tertukar. Uji tanpa kamera: `python tests/test_trunojoyo_frames.py`.

