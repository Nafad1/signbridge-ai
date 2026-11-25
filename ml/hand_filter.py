"""
Memilih tangan yang BENAR-BENAR dipakai sebagai isyarat dari semua deteksi MediaPipe.
Murni numpy (tidak butuh MediaPipe) sehingga mudah diuji: python tests/test_tracking.py

Masalah yang diatasi:
  - tangan orang lain / tangan di latar (jauh lebih kecil)      -> dibuang
  - satu tangan terdeteksi dua kali (kotak saling menimpa)      -> dibuang salah satu
  - tangan yang terpotong tepi bingkai (titik di luar gambar)   -> ditandai, tidak diprediksi
  - mode 1 tangan: selalu pilih tangan TERBESAR (terdekat kamera), bukan sembarang yang pertama

Ukuran tangan memakai panjang telapak (pergelangan -> pangkal jari tengah), bukan luas kotak:
telapak hampir tidak berubah saat tangan mengepal atau terbuka, jadi kepalan tidak keliru dianggap "kecil".
"""
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import MIDDLE_MCP  # noqa: E402


def palm_size(pts, width=1.0, height=1.0):
    """Panjang telapak dalam piksel, dibagi tinggi gambar (agar tidak bergantung resolusi)."""
    d = (np.asarray(pts)[MIDDLE_MCP, :2] - np.asarray(pts)[0, :2]) * np.array([width, height])
    return float(np.linalg.norm(d)) / float(height)


def bbox(pts):
    p = np.asarray(pts)[:, :2]
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_cut(pts, margin=0.005):
    """True jika ada titik tangan di luar bingkai (koordinat ternormalisasi 0-1)."""
    p = np.asarray(pts)[:, :2]
    return bool((p < margin).any() or (p > 1 - margin).any())


def select_hands(hands, width, height, max_hands=2, min_palm=0.06, min_rel_palm=0.5, dup_iou=0.5):
    """
    hands : daftar array (21, 3) hasil MediaPipe.
    Return (kept, dropped): daftar dict {'pts', 'palm', 'reason'} ; kept urut dari terbesar.
    """
    items = [{'pts': np.asarray(h, float), 'palm': palm_size(h, width, height), 'box': bbox(h)} for h in hands]
    items.sort(key=lambda d: -d['palm'])
    kept, dropped = [], []
    for it in items:
        if it['palm'] < min_palm:
            it['reason'] = 'terlalu_kecil'
        elif kept and it['palm'] < min_rel_palm * kept[0]['palm']:
            it['reason'] = 'lebih_kecil_dari_tangan_utama'
        elif any(iou(it['box'], k['box']) > dup_iou for k in kept):
            it['reason'] = 'duplikat'
        elif len(kept) >= max_hands:
            it['reason'] = 'kelebihan'
        else:
            kept.append(it)
            continue
        dropped.append(it)
    return kept, dropped
