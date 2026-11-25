"""
Fungsi ekstraksi fitur dari landmark tangan MediaPipe.

Dipakai bersama oleh:
  - ml/train_alphabet.py  (melatih model huruf dari dataset BISINDO)
  - ml/predict.py         (prediksi saat aplikasi berjalan)
Agar fitur saat training dan saat prediksi PASTI sama, logikanya hanya ditulis di sini.

Input berupa array (21, 3) berisi koordinat x, y, z yang dinormalisasi MediaPipe (0-1).
"""
import numpy as np

NUM_LANDMARKS = 21
MIDDLE_MCP = 9                 # pangkal jari tengah, dipakai sebagai ukuran telapak tangan
WORD_FEATURE_COUNT = 42        # 21 titik x (x, y)
ALPHABET_FEATURE_COUNT = 130   # 2 tangan x 63 + selisih posisi 3 + penanda "2 tangan" 1


# ---------------------------------------------------------------------------
# Model KATA (5 kosakata, 1 tangan) -- sama persis dengan logika lama di app.py
# ---------------------------------------------------------------------------
def word_features(landmarks):
    """Koordinat x,y relatif terhadap pergelangan tangan (titik 0) -> 42 angka."""
    pts = np.asarray(landmarks, dtype=float)
    return (pts[:, :2] - pts[0, :2]).ravel().tolist()


# ---------------------------------------------------------------------------
# Model HURUF (A-Z, 1 atau 2 tangan)
# ---------------------------------------------------------------------------
def _normalize_hand(pts):
    """Geser agar pergelangan = (0,0,0), lalu bagi dengan panjang telapak tangan.
    Hasilnya tidak bergantung pada posisi tangan di layar ataupun jarak ke kamera."""
    rel = pts - pts[0]
    scale = np.linalg.norm(rel[MIDDLE_MCP, :2])
    if scale < 1e-6:
        scale = 1.0
    return rel / scale, scale


def alphabet_features(hands, width=1.0, height=1.0):
    """
    hands  : daftar 1-2 array (21, 3) hasil MediaPipe (koordinat ternormalisasi).
    width, height : ukuran gambar dalam piksel. Dipakai agar x dan y punya skala yang
                    sama (webcam 640x480 tidak berbentuk persegi). Dataset BISINDO berupa
                    gambar persegi 1:1, jadi untuk data training cukup 1.0 / 1.0.

    Urutan tangan dibuat KONSISTEN dengan mengurutkan dari kiri ke kanan di gambar,
    sebab MediaPipe tidak menjamin urutan tangan pertama/kedua.

    Return: vektor 130 angka
      [0:63]    tangan kiri-gambar   (21 titik x,y,z, sudah dinormalisasi)
      [63:126]  tangan kanan-gambar  (nol jika hanya ada 1 tangan)
      [126:129] posisi pergelangan tangan ke-2 relatif ke ke-1 (dibagi ukuran tangan ke-1)
      [129]     1.0 jika ada 2 tangan, selain itu 0.0
    """
    prepared = []
    for h in list(hands)[:2]:
        p = np.asarray(h, dtype=float).copy()
        p[:, 0] *= width
        p[:, 1] *= height
        p[:, 2] *= width        # MediaPipe: z memakai skala yang sama dengan x
        prepared.append(p)
    prepared.sort(key=lambda p: p[0, 0])

    feat = np.zeros(ALPHABET_FEATURE_COUNT)
    if not prepared:
        return feat

    first, scale1 = _normalize_hand(prepared[0])
    feat[0:63] = first.ravel()
    if len(prepared) == 2:
        second, _ = _normalize_hand(prepared[1])
        feat[63:126] = second.ravel()
        feat[126:129] = (prepared[1][0] - prepared[0][0]) / scale1
        feat[129] = 1.0
    return feat


def hands_from_dataset_row(row):
    """Ubah 1 baris dataset BISINDO (126 angka: x,y,z per landmark, 2 tangan) menjadi
    daftar array (21, 3). Blok tangan ke-2 yang kosong (NaN/0) dianggap tidak ada."""
    row = np.nan_to_num(np.asarray(row, dtype=float))
    hands = [row[:63].reshape(NUM_LANDMARKS, 3)]
    if np.any(row[63:126]):
        hands.append(row[63:126].reshape(NUM_LANDMARKS, 3))
    return hands
