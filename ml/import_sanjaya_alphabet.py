"""
Mengubah dataset "BISINDO Indonesian Sign Language: Alphabet Image Data" (Samuel Ady Sanjaya,
Universitas Multimedia Nusantara, CC BY 4.0, DOI 10.17632/ywnjpbcz8m.1) menjadi fitur huruf.

Jalankan dari folder utama proyek (venv aktif), sesudah mengekstrak zip "02. Resized Images":
    python ml/import_sanjaya_alphabet.py --source "C:\\data\\02. Resized Images"

Hasil: dataset/bisindo/sanjaya_features.csv  (label, grup, 130 fitur)
Lalu latih ulang:  python ml/train_alphabet.py   (otomatis memakai file itu jika ada)

Aturan penyaringan (agar data yang salah deteksi tidak merusak model):
  - gambar tanpa tangan terdeteksi dibuang;
  - jumlah tangan harus masuk akal untuk huruf itu (aturan dari dataset lama, lihat expected_hand_counts()).
    Contoh: huruf yang jelas dua tangan tetapi hanya terdeteksi satu tangan dibuang.
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE_DIR = os.path.dirname(HERE)

from extract_features import ALPHABET_FEATURE_COUNT, alphabet_features, hands_from_dataset_row  # noqa: E402
from hand_filter import select_hands  # noqa: E402

OLD_CSV = os.path.join(BASE_DIR, 'dataset', 'bisindo', 'bisindo_features.csv')
OUT_CSV = os.path.join(BASE_DIR, 'dataset', 'bisindo', 'sanjaya_features.csv')
IMG_EXT = ('.jpg', '.jpeg', '.png')
MIN_ROWS_TO_SAVE = 500   # di bawah ini hasil dianggap salah folder (mis. gambar biner/hitam-putih) dan TIDAK disimpan


def enough_rows(n_rows):
    return n_rows >= MIN_ROWS_TO_SAVE


def parse_group(filename):
    """Grup untuk evaluasi = perangkat (HP) yang tertulis di nama file, mis.
    'B_Baju_Apple_Iphone 12.JPG' -> 'iphone 12'. Foto bernama cap waktu (20240509_182656.jpg)
    -> 'sesi-0509'. Ini hanya PERKIRAAN: nama file tidak memuat ID orang."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r'(\.jpe?g)+$', '', name, flags=re.I)
    if re.fullmatch(r'\d{8}_\d{6}', name):
        return 'sesi-' + name[4:8]
    parts = [p for p in re.split(r'_', name) if p.strip()]
    if len(parts) < 2:
        return 'lain'
    dev = parts[-1]
    if (dev[0].isdigit() or len(dev) <= 3) and len(parts) >= 3:   # '14Pro' / 'A' -> gabung dengan bagian sebelumnya
        dev = parts[-2] + dev
    dev = re.sub(r'[^a-z0-9+]', '', dev.lower())
    dev = re.sub(r'^(apple|samsung|oppo|vivo|xiaomi|realme|infinix)', '', dev)
    return dev or 'lain'


def find_letter_root(source, max_depth=4):
    """Cari folder yang berisi subfolder huruf (A, B, ... Z) di dalam 'source', sampai max_depth tingkat.
    Windows sering mengekstrak zip ke folder tambahan, jadi jalur yang diketik belum tentu tepat."""
    source = os.path.abspath(source)
    if not os.path.isdir(source):
        return None
    def n_letters(folder):
        return sum(1 for d in os.listdir(folder)
                   if len(d) == 1 and d.isalpha() and os.path.isdir(os.path.join(folder, d)))
    stack = [(source, 0)]
    while stack:
        folder, depth = stack.pop(0)
        if n_letters(folder) >= 20:
            return folder
        if depth < max_depth:
            for d in sorted(os.listdir(folder)):
                p = os.path.join(folder, d)
                if os.path.isdir(p):
                    stack.append((p, depth + 1))
    return None


def expected_hand_counts():
    """Jumlah tangan yang DIIZINKAN untuk tiap huruf (himpunan), dari dataset lama.
    Di dataset lama banyak huruf dua tangan hanya terdeteksi satu tangan (kira-kira separuh baris),
    jadi 'jumlah terbanyak' menyesatkan. Aturannya memakai porsi baris dua tangan:
        < 15%   -> huruf satu tangan : hanya 1 tangan yang diterima
        > 30%   -> huruf dua tangan  : hanya 2 tangan yang diterima
        lainnya -> ragu              : 1 atau 2 tangan diterima"""
    import pandas as pd
    df = pd.read_csv(OLD_CSV)
    allowed = {}
    for letter, g in df.groupby('label'):
        counts = Counter(len(hands_from_dataset_row(r)) for r in g.drop(columns='label').values)
        share2 = counts[2] / max(1, sum(counts.values()))
        allowed[letter] = {1} if share2 < 0.15 else ({2} if share2 > 0.30 else {1, 2})
    return allowed


def make_detector():
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                     min_detection_confidence=0.5)

    def detect(rgb):
        res = hands.process(rgb)
        return [np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark])
                for hl in (res.multi_hand_landmarks or [])]
    return detect


def build_rows(files, detect, expected, mirror=False, lenient=False, dup_iou=0.5, min_rel_palm=0.5):
    """files: daftar (jalur, huruf). detect: fungsi gambar RGB -> daftar tangan (21,3). Dipisah supaya bisa diuji."""
    import cv2
    rows, stats = [], defaultdict(Counter)
    for path, letter in files:
        img = cv2.imread(path)
        if img is None:
            stats[letter]['gambar_rusak'] += 1
            continue
        h, w = img.shape[:2]
        hands = detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        kept, dropped = select_hands(hands, w, h, max_hands=2, min_palm=0.05,
                                     min_rel_palm=min_rel_palm, dup_iou=dup_iou)
        if not kept:
            stats[letter]['tanpa_tangan'] += 1
            continue
        if not lenient and len(kept) not in expected.get(letter, {1, 2}):
            stats[letter]['jumlah_tangan_beda'] += 1
            # Pisahkan penyebabnya: MediaPipe memang hanya melihat 1 tangan, atau filter saya yang membuang tangan ke-2
            stats[letter]['beda_mp_1_tangan' if len(hands) == 1 else 'beda_dibuang_filter'] += 1
            continue
        pts = [k['pts'] for k in kept]
        group = parse_group(path)
        rows.append([letter, group] + list(alphabet_features(pts, w, h)))
        stats[letter]['dipakai'] += 1
        if mirror:   # cermin: x -> 1-x (tangan kiri/kanan tertukar)
            flipped = [p * np.array([-1, 1, 1]) + np.array([1, 0, 0]) for p in pts]
            rows.append([letter, group] + list(alphabet_features(flipped, w, h)))
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='folder berisi A, B, ... Z')
    ap.add_argument('--mirror', action='store_true', help='tambah salinan dicerminkan (opsional)')
    ap.add_argument('--lenient', action='store_true',
                    help='terima 1 atau 2 tangan untuk semua huruf (lebih banyak foto terpakai; '
                         'termasuk huruf dua tangan yang hanya terdeteksi satu tangan saat tangan bertumpuk)')
    ap.add_argument('--dup-iou', type=float, default=0.5,
                    help='tangan ke-2 dianggap kembaran (dibuang) jika kotaknya tumpang tindih lebih dari ini. '
                         'Huruf dua tangan yang bertumpuk (M, N, Q) mungkin perlu 0.85')
    ap.add_argument('--min-rel-palm', type=float, default=0.5,
                    help='tangan ke-2 dibuang jika telapaknya lebih kecil dari ini x tangan utama (turunkan ke 0.3 jika perlu)')
    args = ap.parse_args()

    root = find_letter_root(args.source)
    if root is None:
        sys.exit(f'Folder huruf (A, B, ... Z) tidak ditemukan di "{args.source}".\n'
                 'Cek jalurnya, misalnya:  dir "C:\\data" /s /b | findstr /i "\\A\\" | more')
    print('Folder huruf:', root)
    files = []
    for letter in sorted(d for d in os.listdir(root) if len(d) == 1 and d.isalpha()):
        folder = os.path.join(root, letter)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXT):
                files.append((os.path.join(folder, f), letter.upper()))
    print(f'{len(files)} gambar di {len(set(l for _, l in files))} huruf')
    expected = expected_hand_counts()
    print('Jumlah tangan yang diterima per huruf:', {k: '/'.join(map(str, sorted(v))) for k, v in sorted(expected.items())})

    rows, stats = build_rows(files, make_detector(), expected, mirror=args.mirror, lenient=args.lenient,
                              dup_iou=args.dup_iou, min_rel_palm=args.min_rel_palm)

    print('\nHuruf  dipakai  tanpa_tangan  jumlah_tangan_beda (MediaPipe lihat 1 | dibuang filter)  rusak')
    for letter in sorted(stats):
        s = stats[letter]
        print(f"  {letter}    {s['dipakai']:5d}   {s['tanpa_tangan']:8d}   {s['jumlah_tangan_beda']:12d}"
              f"  ({s['beda_mp_1_tangan']:4d} | {s['beda_dibuang_filter']:4d})   {s['gambar_rusak']:5d}")
    tot = lambda k: sum(s[k] for s in stats.values())  # noqa: E731
    print(f"\nJumlah tangan beda: {tot('jumlah_tangan_beda')} foto -> MediaPipe hanya melihat 1 tangan: "
          f"{tot('beda_mp_1_tangan')} | tangan ke-2 dibuang oleh filter saya: {tot('beda_dibuang_filter')}")
    used = sum(s['dipakai'] for s in stats.values())
    print(f'\nTotal dipakai: {used} dari {len(files)} gambar ({100 * used / max(1, len(files)):.0f}%)')
    low = [l for l, s in stats.items() if s['dipakai'] < 60]
    if low:
        print('PERINGATAN: huruf dengan kurang dari 60 gambar terpakai:', ', '.join(sorted(low)))

    if not enough_rows(len(rows)):
        sys.exit(f'\nHanya {len(rows)} baris yang terpakai (minimal {MIN_ROWS_TO_SAVE}). File TIDAK disimpan, '
                 'file lama (jika ada) tetap aman.\nBiasanya ini salah folder: pakai folder foto berwarna '
                 '("02. Resized Images"), bukan "Binary Images" atau "Background Removed".')
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['label', 'group'] + [f'f{i}' for i in range(ALPHABET_FEATURE_COUNT)])
        w.writerows(rows)
    print('Disimpan:', OUT_CSV)


if __name__ == '__main__':
    main()
