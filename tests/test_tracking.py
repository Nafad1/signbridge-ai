"""Uji filter tangan (tanpa MediaPipe).   Jalankan: python tests/test_tracking.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml'))
from hand_filter import is_cut, select_hands  # noqa: E402


def hand(cx, cy, palm=0.2, open_hand=True):
    """Tangan sintetis: 21 titik; pergelangan di (cx,cy), telapak ke atas sepanjang `palm` (tinggi gambar=1)."""
    pts = np.zeros((21, 3))
    pts[0] = [cx, cy, 0]
    pts[9] = [cx, cy - palm, 0]
    reach = palm * (1.8 if open_hand else 0.9)       # jari terbuka vs mengepal
    for i in range(1, 21):
        if i != 9:
            pts[i] = [cx + (i % 5 - 2) * palm * 0.15, cy - palm - (reach - palm) * (i % 4) / 3, 0]
    return pts


def check(name, cond):
    print(('OK   ' if cond else 'GAGAL'), name)
    assert cond, name


W, H = 640, 480
big = hand(0.4, 0.8, 0.20)

# 1. tangan latar yang jauh lebih kecil dibuang
kept, dropped = select_hands([big, hand(0.8, 0.5, 0.05)], W, H)
check('tangan kecil di latar dibuang', len(kept) == 1 and dropped[0]['reason'] == 'terlalu_kecil')

# 2. tangan ke-2 yang sah (ukuran mirip) dipertahankan
kept, _ = select_hands([big, hand(0.7, 0.8, 0.18)], W, H)
check('dua tangan sah dipertahankan', len(kept) == 2)

# 3. kepalan tidak dianggap kecil (ukuran pakai telapak, bukan luas kotak)
kept, _ = select_hands([hand(0.3, 0.8, 0.2, True), hand(0.7, 0.8, 0.2, False)], W, H)
check('kepalan + tangan terbuka tetap 2 tangan', len(kept) == 2)

# 4. tangan lebih kecil dari 50% tangan utama dibuang
kept, dropped = select_hands([big, hand(0.8, 0.8, 0.08)], W, H)
check('tangan <50% ukuran utama dibuang', len(kept) == 1 and dropped[0]['reason'] == 'lebih_kecil_dari_tangan_utama')

# 5. deteksi ganda untuk tangan yang sama
kept, dropped = select_hands([big, big + np.array([0.005, 0.0, 0.0])], W, H)
check('deteksi ganda dibuang', len(kept) == 1 and dropped[0]['reason'] == 'duplikat')

# 6. mode 1 tangan memilih yang TERBESAR, apa pun urutan masukannya
a, b = hand(0.3, 0.8, 0.15), hand(0.7, 0.8, 0.22)
for order in ([a, b], [b, a]):
    kept, _ = select_hands(order, W, H, max_hands=1)
    check('mode 1 tangan memilih tangan terbesar', abs(kept[0]['palm'] - 0.22) < 1e-6)

# 7. tangan terpotong bingkai
check('tangan terpotong terdeteksi', is_cut(hand(0.05, 0.8, 0.2) + np.array([-0.2, 0, 0])))
check('tangan utuh tidak dianggap terpotong', not is_cut(big))

# 8. tidak ada tangan
kept, dropped = select_hands([], W, H)
check('tanpa deteksi -> kosong', kept == [] and dropped == [])
print('\nSemua tes lulus.')
