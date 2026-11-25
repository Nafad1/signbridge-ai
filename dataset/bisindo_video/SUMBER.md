# Sumber Klip Video

Klip di `static/videos/` dipotong dari **BISINDO Video Dataset** (Tito Sugiharto, Mendeley Data):
https://data.mendeley.com/datasets/f33k9w86wr
Cek lisensi/ketentuan penggunaan di halaman tersebut sebelum mempublikasikan aplikasi atau laporan.

Dipotong dengan `python ml/import_videos.py --dataset "<folder BISINDO Video Dataset>"`:
- Huruf A-Q  -> static/videos/alphabet/a.mp4 ... q.mp4
- Angka 1-10 -> static/videos/words/1.mp4 ... 10.mp4
- Hari       -> static/videos/words/senin.mp4 ... minggu.mp4

Catatan keandalan:
- Label diberikan berdasarkan URUTAN kemunculan di video. Huruf A, I, P sudah dicocokkan dengan foto
  referensi dataset lain. Angka mengikuti urutan jari yang jelas (1..10). Urutan HARI (Senin-Minggu)
  adalah ASUMSI dan belum bisa diverifikasi: periksa di halaman Kamus; jika ada yang tertukar, ubah
  daftar 'labels' di ml/import_videos.py lalu jalankan ulang.
- Huruf R-Z tidak diambil (tangan hampir keluar bingkai di bagian akhir video, deteksi tidak andal).
- Video Introductory, Family, dan Short Story belum dipakai (tidak ada daftar label per klip).
- Folder "Alphabets Image" hanya frame dari video abjad yang sama, tanpa label; tidak dipakai.
- Paket WL-BISINDO-main.zip hanya berisi kode & metadata; 1.600 videonya ada di Kaggle
  (glennleonali/wl-bisindo, varian Banten) dan belum dimasukkan.

---
# WL-BISINDO (32 kata)

Klip kata di `static/videos/words/` (air, belajar, ..., malam; total 32) berasal dari **WL-BISINDO**:
- Grace Oktaviani Kindy, Glenn Leonali, Henry Lucky (BINUS University).
  "Word-Level BISINDO: A Novel Video Indonesian Sign Language Dataset and Baseline Methods",
  Procedia Computer Science, 2025. DOI: 10.1016/j.procs.2025.08.277
- Dataset: https://www.kaggle.com/datasets/glennleonali/wl-bisindo
- Kode: https://github.com/AceKinnn/WL-BISINDO

**LISENSI: CC BY-NC 4.0 (Attribution-NonCommercial).** Boleh untuk tugas kuliah, penelitian, dan demo
dengan mencantumkan kredit di atas. TIDAK boleh dipakai untuk produk/layanan komersial.

Pembuatan klip: `python ml/import_wl_bisindo.py --source <folder> --signer 3 --sample 2`
(1 klip per kata, penanda nomor 3, sampel 2, dipotong ke bagian tubuh atas, 480 px).
Catatan: ini varian BISINDO daerah Banten dan penandanya berbeda dari video abjad/angka/hari
(Mendeley), sehingga isyarat dan tampilan antar sumber bisa tidak seragam.

## Huruf A-Z: Indonesian Sign Language Dataset: Alphabet Video
- Indah Siradjuddin (Universitas Trunojoyo Madura), Mendeley Data, V2, 26 Februari 2024, DOI 10.17632/p7j5jrsbbb.2.
  Lisensi: CC BY 4.0 (wajib mencantumkan kredit). https://data.mendeley.com/datasets/p7j5jrsbbb/2
- Dipakai satu klip per huruf (nomor klip ada di `PICKS` pada `ml/import_trunojoyo_alphabet.py`), dipotong ke area orang
  dan diubah ke H.264 tanpa suara oleh skrip itu. Wajah dalam sumber sudah dikaburkan oleh pembuat dataset.
- Penampilan: F-Z berlatar putih/abu-abu dan beresolusi 1080p; huruf A-E berlatar hijau-biru dan sebagian
  hanya 640x480, jadi A-E lebih buram. Penandanya berbeda-beda antar huruf, jadi tampilan tidak seragam.
- Video huruf A-Q dari dataset Tito Sugiharto (di atas) digantikan oleh klip ini; angka dan hari tetap dari dataset Tito Sugiharto.
