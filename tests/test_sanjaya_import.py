"""Uji logika impor dataset Sanjaya tanpa MediaPipe (detektor tiruan).
Jalankan:  python tests/test_sanjaya_import.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'ml'))
import import_sanjaya_alphabet as m  # noqa: E402


def check(name, cond):
    print(('OK   ' if cond else 'GAGAL'), name)
    assert cond, name


# --- nama file -> grup perangkat
check('cap waktu -> sesi', m.parse_group('20240509_182656.jpg') == 'sesi-0509')
check('iPhone dengan spasi', m.parse_group('B_Baju_Apple_Iphone 12.JPG') == 'iphone12')
check('ekstensi ganda', m.parse_group('B_Baju_Oppo_Reno 11f.jpg.jpg') == 'reno11f')
check('model pendek digabung dengan merek', m.parse_group('B_Baju_Samsung_A.jpg') == 'a')
check('nama aneh tidak membuat error', m.parse_group('_.jpg') == 'lain')


def hand(cx=0.5, cy=0.6, palm=0.2):
    p = np.zeros((21, 3)); p[0] = [cx, cy, 0]; p[9] = [cx, cy - palm, 0]
    for i in range(1, 21):
        if i != 9:
            p[i] = [cx + (i % 5 - 2) * 0.03, cy - palm - 0.02 * (i % 4), 0.01 * i]
    return p


tmp = tempfile.mkdtemp()
img_path = os.path.join(tmp, 'A_Baju_Iphone 12.jpg')
cv2.imwrite(img_path, np.full((224, 224, 3), 128, np.uint8))
allowed = {'A': {2}, 'C': {1}, 'G': {1, 2}}

cases = [
    ('A dengan 2 tangan dipakai', 'A', [hand(0.3), hand(0.7)], 1),
    ('A dengan 1 tangan dibuang', 'A', [hand(0.5)], 0),
    ('C dengan 1 tangan dipakai', 'C', [hand(0.5)], 1),
    ('C dengan 2 tangan dibuang', 'C', [hand(0.3), hand(0.7)], 0),
    ('G (ragu) menerima 1 tangan', 'G', [hand(0.5)], 1),
    ('tanpa tangan dibuang', 'A', [], 0),
]
for name, letter, hands, want in cases:
    rows, stats = m.build_rows([(img_path, letter)], lambda rgb, h=hands: h, allowed)
    check(name, len(rows) == want)
    if want:
        check('  baris = label + grup + 130 fitur', len(rows[0]) == 2 + 130 and rows[0][0] == letter and rows[0][1] == 'iphone12')

rows, _ = m.build_rows([(img_path, 'C')], lambda rgb: [hand(0.5)], allowed, mirror=True)
check('opsi --mirror menambah salinan cermin', len(rows) == 2 and rows[0][2:] != rows[1][2:])

rows, stats = m.build_rows([(os.path.join(tmp, 'tidak_ada.jpg'), 'A')], lambda rgb: [], allowed)
check('gambar rusak dicatat, bukan error', stats['A']['gambar_rusak'] == 1 and not rows)
# --- mode lenient & pengaman agar hasil salah folder tidak menimpa file bagus
rows, _ = m.build_rows([(img_path, 'A')], lambda rgb: [hand(0.5)], allowed, lenient=True)
check('--lenient menerima 1 tangan untuk huruf dua tangan', len(rows) == 1)
rows, _ = m.build_rows([(img_path, 'A')], lambda rgb: [], allowed, lenient=True)
check('--lenient tetap membuang gambar tanpa tangan', len(rows) == 0)
check('1 baris (folder biner) TIDAK disimpan', not m.enough_rows(1))
check('2458 baris disimpan', m.enough_rows(2458))

# --- pencarian folder huruf (jalur bersarang akibat ekstraksi zip)
base = tempfile.mkdtemp()
deep = os.path.join(base, '02__Resized_Images', '02. Resized Images')
for L in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    os.makedirs(os.path.join(deep, L))
os.makedirs(os.path.join(base, 'lain', 'foto'))
check('folder huruf ditemukan walau jalurnya bersarang', m.find_letter_root(base) == deep)
check('jalur yang sudah tepat dikenali', m.find_letter_root(deep) == deep)
check('jalur tidak ada -> None (bukan error)', m.find_letter_root(os.path.join(base, 'tidak-ada')) is None)
check('folder tanpa huruf -> None', m.find_letter_root(os.path.join(base, 'lain')) is None)
print('\nSemua tes lulus.')
