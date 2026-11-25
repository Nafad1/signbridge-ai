"""
Memotong video dari 'BISINDO Video Dataset' (Mendeley, Tito Sugiharto) menjadi klip per
huruf/angka/hari dan menaruhnya di static/videos/ untuk fitur Teks ke Isyarat.

    python ml/import_videos.py --dataset "C:\\path\\ke\\BISINDO Video Dataset"

Syarat: ffmpeg terpasang (perintah `ffmpeg` bisa dipanggil), plus opencv-python, numpy, scipy.

Cara kerja: penanda berdiri di latar biru dan menurunkan tangan di antara isyarat. Skrip
mendeteksi kapan tangan bergerak (selisih terhadap baseline lokal), memotong tiap kegiatan,
lalu memberi label BERDASARKAN URUTAN. Jumlah segmen harus cocok dengan jumlah label; kalau
tidak, skrip berhenti agar tidak salah memberi label.

Cakupan:
  - Huruf A-Q saja. R-Z tidak diambil: di video bagian akhir tangan hampir keluar bingkai dan
    pemotongan otomatis tidak andal. Video ditutup kartu CapCut.
  - Angka 1-10 dan hari Senin-Minggu (urutan label ASUMSI: urutan umum; periksa hasilnya!).
  - Video Introductory / Family / Short Story tidak diambil (tidak ada daftar labelnya).
Jika ada klip yang tertukar, ubah daftar 'labels' di JOBS lalu jalankan ulang.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import (binary_closing, binary_opening, label,
                           minimum_filter1d, uniform_filter1d)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(BASE, 'static', 'videos')

JOBS = [
    {'video': 'Alphabets Video Dataset/Alphabets.mp4', 'out': 'alphabet', 'thr': 0.03,
     'labels': list('ABCDEFGHIJKLMNOPQ'), 'take_first': True},
    {'video': 'Numbers Video Dataset/Numbers.mp4', 'out': 'words', 'thr': 0.015,
     'labels': [str(i) for i in range(1, 11)]},
    {'video': 'Name-of-the-Day Video Dataset/Days.mp4', 'out': 'words', 'thr': 0.015,
     'labels': ['SENIN', 'SELASA', 'RABU', 'KAMIS', 'JUMAT', 'SABTU', 'MINGGU']},
]


def activity(path, step=2):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ref, vals, idx, i = None, [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (192, 208)), cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
            ref = g if ref is None else ref
            d = np.abs(g - ref)[int(g.shape[0] * 0.38):]      # abaikan kepala
            vals.append((d > 35).mean())
            idx.append(i)
        i += 1
    return np.array(idx) / fps, np.array(vals)


def find_segments(t, v, thr, win_s=8.0, min_len=0.5, gap=0.3):
    dt = np.median(np.diff(t))
    sm = uniform_filter1d(v, 5)
    base = uniform_filter1d(minimum_filter1d(sm, int(win_s / dt)), int(win_s / dt / 2) + 1)
    on = (sm - base) > thr
    on = binary_closing(on, np.ones(int(gap / dt) + 1))
    on = binary_opening(on, np.ones(max(1, int(min_len / dt))))
    lab, n = label(on)
    segs = []
    for k in range(1, n + 1):
        w = np.where(lab == k)[0]
        segs.append((float(t[w[0]]), float(t[w[-1]])))
    return segs


def person_box(path, segs):
    """Kotak yang memuat penanda di seluruh segmen. Memakai komponen terhubung terbesar
    (bukan piksel bukan-biru mentah) supaya noise/strip tepi video tidak melebarkan kotak."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    boxes = []
    for a, b in segs:
        for tt in np.linspace(a, b, 4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tt * fps))
            ok, f = cap.read()
            if not ok:
                continue
            H, W = f.shape[:2]
            bg = np.median(np.vstack([f[:20].reshape(-1, 3), f[:, :20].reshape(-1, 3)]), axis=0)
            fg = (np.linalg.norm(f.astype(float) - bg, axis=2) > 70).astype(np.uint8)
            fg[:, :10] = 0; fg[:, -10:] = 0; fg[:10] = 0
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(fg)
            if n > 1:
                k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h, _ = stats[k]
                boxes.append((x, y, x + w, y + h))
    b = np.array(boxes)
    pad = 0.04 * W
    x0, x1 = max(0, b[:, 0].min() - pad), min(W, b[:, 2].max() + pad)
    y0 = max(0, b[:, 1].min() - pad)
    return int(x0) // 2 * 2, int(y0) // 2 * 2, int(x1 - x0) // 2 * 2, int(H - y0) // 2 * 2


def cut(src, dst, start, end, box):
    x, y, w, h = box
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{start:.2f}', '-to', f'{end:.2f}', '-i', src,
           '-vf', f'crop={w}:{h}:{x}:{y},scale=480:-2', '-an', '-c:v', 'libx264', '-crf', '28',
           '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='folder "BISINDO Video Dataset"')
    ap.add_argument('--pad', type=float, default=0.4, help='detik tambahan sebelum/sesudah isyarat')
    args = ap.parse_args()

    for job in JOBS:
        src = os.path.join(args.dataset, job['video'])
        print(f"\n== {job['video']}")
        t, v = activity(src)
        segs = find_segments(t, v, job['thr'])
        labels = job['labels']
        print(f"segmen terdeteksi: {len(segs)} | label: {len(labels)}")
        if job.get('take_first'):
            segs = segs[:len(labels)]
        if len(segs) != len(labels):
            sys.exit("Jumlah segmen tidak cocok dengan jumlah label; berhenti agar tidak salah label.")
        box = person_box(src, segs)
        dur = len(t) and t[-1]
        for (a, b), lab in zip(segs, labels):
            name = lab.lower().replace(' ', '_') + '.mp4'
            cut(src, os.path.join(OUT_ROOT, job['out'], name), max(0, a - args.pad), min(dur, b + args.pad), box)
            print(f"  {lab:8s} {a:6.1f}-{b:6.1f}s -> {job['out']}/{name}")


if __name__ == '__main__':
    main()
