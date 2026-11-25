"""
Fitur URUTAN FRAME untuk mengenali kata bergerak (dataset WL-BISINDO).
Dipakai bersama oleh ekstraksi, pelatihan, dan (nanti) prediksi di aplikasi.

Satu frame  -> vektor 134 angka  : 2 slot tangan (urut kiri-ke-kanan di gambar) x 67
              [0:63]  bentuk tangan (21 titik x,y,z relatif pergelangan, dibagi panjang telapak)
              [63:65] posisi pergelangan (x,y) di gambar 0-1   <- penting: letak tangan di tubuh
              [65]    ukuran telapak (perkiraan jarak ke kamera)
              [66]    1.0 jika tangan ada
Satu video  -> vektor 1608 angka : 10 titik waktu (diresample) + rata-rata + simpangan baku.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import _normalize_hand  # noqa: E402
from hand_filter import palm_size  # noqa: E402

PER_HAND = 67
FRAME_DIM = 2 * PER_HAND          # 134
STEPS = 10
SEQ_DIM = STEPS * FRAME_DIM + 2 * FRAME_DIM   # 1608


def frame_vector(hands, width, height):
    """hands: daftar 0-2 array (21,3) koordinat ternormalisasi MediaPipe (sudah difilter)."""
    vec = np.zeros(FRAME_DIM)
    prepared = sorted([np.asarray(h, float) for h in list(hands)[:2]], key=lambda p: p[0, 0])
    for slot, h in enumerate(prepared):
        p = h.copy()
        p[:, 0] *= width
        p[:, 1] *= height
        p[:, 2] *= width
        shape, _ = _normalize_hand(p)
        b = slot * PER_HAND
        vec[b:b + 63] = shape.ravel()
        vec[b + 63], vec[b + 64] = h[0, 0], h[0, 1]
        vec[b + 65] = palm_size(h, width, height)
        vec[b + 66] = 1.0
    return vec


def mirror_frames(frames):
    """Cerminkan urutan frame (untuk augmentasi penanda kidal / arah kamera berbeda)."""
    out = np.array(frames, dtype=float, copy=True)
    for t in range(len(out)):
        slots = []
        for s in (0, 1):
            h = out[t, s * PER_HAND:(s + 1) * PER_HAND].copy()
            if h[66] > 0:
                h[0:63:3] *= -1            # x bentuk dibalik
                h[63] = 1.0 - h[63]        # posisi x dicerminkan
            slots.append(h)
        if slots[1][66] > 0:               # dua tangan: urutan kiri-kanan ikut tertukar
            slots = slots[::-1]
        out[t] = np.concatenate(slots)
    return out


def sequence_features(frames, steps=STEPS):
    """frames: array (T,134). Return vektor (SEQ_DIM,) atau None jika tidak ada tangan sama sekali."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2 or len(frames) == 0:
        return None
    present = (frames[:, 66] + frames[:, 133]) > 0
    idx = np.where(present)[0]
    if len(idx) == 0:
        return None
    frames = frames[idx[0]:idx[-1] + 1].copy()           # buang frame kosong di awal/akhir
    for t in range(1, len(frames)):                       # frame kosong di tengah: pakai frame sebelumnya
        if frames[t, 66] + frames[t, 133] == 0:
            frames[t] = frames[t - 1]
    T = len(frames)
    grid = np.linspace(0, T - 1, steps)
    resampled = np.stack([np.interp(grid, np.arange(T), frames[:, d]) for d in range(FRAME_DIM)], axis=1)
    return np.concatenate([resampled.ravel(), frames.mean(0), frames.std(0)])
