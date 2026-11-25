"""
Membuat static/data/poses.json: pose tangan (21 titik) untuk tiap huruf & kata.
Dipakai fitur Teks ke Isyarat sebagai animasi kerangka tangan CADANGAN
ketika video asli belum tersedia di static/videos/.

    python ml/build_poses.py

Huruf : dari dataset BISINDO (pose "tengah" = medoid dari grup arah-tangan terbesar).
Kata  : dari dataset/raw/data.csv hasil collect_data.py (median per kata).
Catatan: dataset huruf berisi gambar yang di-flip acak, jadi pose bisa tampil
tertukar kiri-kanan dibanding foto aslinya.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import _normalize_hand, hands_from_dataset_row  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'static', 'data', 'poses.json')


def _hand_sign(p):
    """Tanda arah-tangan (cermin membalik tandanya): silang jari telunjuk x kelingking."""
    a, b = p[5, :2] - p[0, :2], p[17, :2] - p[0, :2]
    return 1 if a[0] * b[1] - a[1] * b[0] >= 0 else -1


def _pack(hands):
    """hands: list (21,3) -> posisi relatif ke pergelangan tangan pertama, skala = telapak tangan 1."""
    hands = sorted(hands, key=lambda p: p[0, 0])
    _, scale = _normalize_hand(hands[0])
    origin = hands[0][0]
    return [[[round(float(v), 3) for v in ((pt[:2] - origin[:2]) / scale)] for pt in h] for h in hands]


def alphabet_poses():
    df = pd.read_csv(os.path.join(BASE, 'dataset', 'bisindo', 'bisindo_features.csv'))
    out = {}
    for letter in sorted(df['label'].unique()):
        rows = df[df['label'] == letter].drop(columns='label').values
        samples = [sorted(hands_from_dataset_row(r), key=lambda p: p[0, 0]) for r in rows]
        signs = np.array([_hand_sign(s[0]) for s in samples])
        keep = [s for s, sg in zip(samples, signs) if sg == (1 if (signs == 1).sum() >= (signs == -1).sum() else -1)]
        keep = keep[:400]
        # fitur untuk memilih medoid: tangan ternormalisasi + posisi tangan ke-2
        def vec(s):
            parts = [_normalize_hand(s[0])[0].ravel()]
            parts.append(((s[1][0] - s[0][0]) / _normalize_hand(s[0])[1]) if len(s) > 1 else np.zeros(3))
            parts.append(_normalize_hand(s[1])[0].ravel() if len(s) > 1 else np.zeros(63))
            return np.concatenate(parts)
        two = [s for s in keep if len(s) == 2]
        pool = two if len(two) > len(keep) / 2 else [s for s in keep if len(s) == 1]
        pool = pool or keep
        V = np.array([vec(s) for s in pool])
        medoid = int(np.argmin(np.linalg.norm(V[:, None] - V[None], axis=2).sum(1)))
        out[letter] = _pack(pool[medoid])
    return out


def word_poses():
    path = os.path.join(BASE, 'dataset', 'raw', 'data.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for label, g in df.groupby('label'):
        xy = np.median(g.drop(columns='label').values, axis=0).reshape(21, 2)
        pts = np.zeros((21, 3))
        pts[:, 0] = xy[:, 0] * 640   # asumsi webcam 640x480 saat merekam
        pts[:, 1] = xy[:, 1] * 480
        out[label] = _pack([pts])
    return out


if __name__ == '__main__':
    data = {'alphabet': alphabet_poses(), 'words': word_poses()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f"Tersimpan: {OUT} | huruf: {len(data['alphabet'])} | kata: {len(data['words'])}")
