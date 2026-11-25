"""
Mengubah video huruf Trunojoyo (A-Z, orang dari depan seperti webcam) menjadi data latih huruf.

Mengapa: foto Sanjaya diambil dari atas dan dekat, sedangkan webcam melihat orang dari depan. Video Trunojoyo
(Indah Siradjuddin, Universitas Trunojoyo Madura, CC BY 4.0, DOI 10.17632/p7j5jrsbbb.2) lebih mirip kondisi webcam.

    python ml/import_trunojoyo_frames.py --source "C:\\data\\trunojoyo_A-E" "C:\\data\\trunojoyo_F-K" "C:\\data\\trunojoyo_L-Q" "C:\\data\\trunojoyo_R-Z"
    python ml/train_alphabet.py

Hasil: dataset/bisindo/trunojoyo_features.csv (label, grup, 130 fitur). train_alphabet.py memakainya otomatis.

Cara kerja per klip:
  1. orang dipotong dari bingkai (supaya tangan cukup besar bagi MediaPipe, seperti saat orang mengisi bingkai webcam);
  2. hanya bagian tengah klip yang dipakai (30%-95% durasi): awal klip biasanya tangan masih turun/belum membentuk huruf;
  3. maksimal 15 frame per klip, merata;
  4. jumlah tangan harus sesuai huruf (aturan yang sama dengan impor Sanjaya).
Grup evaluasi = blok 5 klip berurutan per huruf (perkiraan 'penanda yang sama').
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE_DIR = os.path.dirname(HERE)
from extract_features import ALPHABET_FEATURE_COUNT, alphabet_features  # noqa: E402
from hand_filter import select_hands  # noqa: E402
from import_sanjaya_alphabet import expected_hand_counts, make_detector  # noqa: E402
from import_trunojoyo_alphabet import crop_rect, find_root, person_box  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, 'dataset', 'bisindo', 'trunojoyo_features.csv')
VIDEO_EXT = ('.mov', '.mp4')
WINDOW = (0.30, 0.95)
MAX_FRAMES = 15
MIN_ROWS_TO_SAVE = 500


def clip_group(letter, filename):
    m = re.search(r'_(\d+)', filename)
    n = int(m.group(1)) if m else 1
    return f'tj-{letter}-{(n - 1) // 5}'


def process_clip(path, letter, detect, allowed, dup_iou=0.85, min_rel_palm=0.5, min_palm=0.03,
                 window=WINDOW, max_frames=MAX_FRAMES, box=None):
    """Mengembalikan (baris, statistik). detect: fungsi gambar RGB -> daftar tangan (21,3); dipisah agar bisa diuji."""
    stats = Counter()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        stats['klip_rusak'] += 1
        return [], stats
    rect = crop_rect(box if box is not None else person_box(path))
    x, y, cw, ch = rect
    lo, hi = int(n * window[0]), max(int(n * window[1]), int(n * window[0]) + 1)
    idxs = sorted(set(np.linspace(lo, hi - 1, min(max_frames, hi - lo)).astype(int)))
    group = clip_group(letter, os.path.basename(path))
    rows = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[y:y + ch, x:x + cw]
        h, w = crop.shape[:2]
        hands = detect(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        kept, _ = select_hands(hands, w, h, max_hands=2, min_palm=min_palm,
                               min_rel_palm=min_rel_palm, dup_iou=dup_iou)
        if not kept:
            stats['tanpa_tangan'] += 1
            continue
        if len(kept) not in allowed.get(letter, {1, 2}):
            stats['jumlah_tangan_beda'] += 1
            continue
        rows.append([letter, group] + list(alphabet_features([k['pts'] for k in kept], w, h)))
        stats['dipakai'] += 1
    cap.release()
    return rows, stats


def list_clips(roots):
    clips = []
    for root in roots:
        for letter in sorted(d for d in os.listdir(root) if len(d) == 1 and d.isalpha()):
            folder = os.path.join(root, letter)
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(VIDEO_EXT):
                    clips.append((os.path.join(folder, f), letter.upper()))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, nargs='+', help='satu atau beberapa folder sumber (berisi A, B, ... Z)')
    ap.add_argument('--lenient', action='store_true', help='terima 1 atau 2 tangan untuk semua huruf')
    args = ap.parse_args()

    roots = [r for r in (find_root(s) for s in args.source) if r]
    if not roots:
        sys.exit('Folder huruf tidak ditemukan. Cek jalur --source.')
    print('Folder sumber:', *roots, sep='\n  ')
    clips = list_clips(roots)
    print(f'{len(clips)} klip di {len(set(l for _, l in clips))} huruf')
    allowed = expected_hand_counts()
    if args.lenient:
        allowed = {k: {1, 2} for k in allowed}
    detect = make_detector()

    all_rows, per_letter = [], defaultdict(Counter)
    for k, (path, letter) in enumerate(clips, 1):
        rows, st = process_clip(path, letter, detect, allowed)
        all_rows += rows
        per_letter[letter].update(st)
        per_letter[letter]['klip'] += 1
        if k % 50 == 0 or k == len(clips):
            print(f'  {k}/{len(clips)} klip, {len(all_rows)} baris sejauh ini')

    print('\nHuruf  klip  dipakai  tanpa_tangan  jumlah_tangan_beda')
    for letter in sorted(per_letter):
        s = per_letter[letter]
        print(f"  {letter}   {s['klip']:4d}  {s['dipakai']:6d}  {s['tanpa_tangan']:10d}  {s['jumlah_tangan_beda']:14d}")
    seen = sum(s['dipakai'] + s['tanpa_tangan'] + s['jumlah_tangan_beda'] for s in per_letter.values())
    print(f'\nTotal dipakai: {len(all_rows)} dari {seen} frame ({100 * len(all_rows) / max(1, seen):.0f}%)')
    low = [l for l, s in per_letter.items() if s['dipakai'] < 60]
    if low:
        print('PERINGATAN: huruf dengan kurang dari 60 frame terpakai:', ', '.join(sorted(low)))
    if len(all_rows) < MIN_ROWS_TO_SAVE:
        sys.exit(f'\nHanya {len(all_rows)} baris (minimal {MIN_ROWS_TO_SAVE}). File TIDAK disimpan; file lama aman.')
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['label', 'group'] + [f'f{i}' for i in range(ALPHABET_FEATURE_COUNT)])
        w.writerows(all_rows)
    print('Disimpan:', OUT_CSV)


if __name__ == '__main__':
    main()
