"""Uji impor frame Trunojoyo tanpa MediaPipe (detektor tiruan). Jalankan:  python tests/test_trunojoyo_frames.py"""
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'ml'))
import import_trunojoyo_frames as tf  # noqa: E402


def check(name, cond):
    print(('OK   ' if cond else 'GAGAL'), name)
    assert cond, name


def hand(cx, cy=0.6, palm=0.2):
    p = np.zeros((21, 3)); p[0] = [cx, cy, 0]; p[9] = [cx, cy - palm, 0]
    for i in range(1, 21):
        if i != 9:
            p[i] = [cx + (i % 5 - 2) * 0.03, cy - palm - 0.02 * (i % 4), 0.01 * i]
    return p


# klip sintetis 60 frame, latar putih, "orang" = kotak gelap di tengah
tmp = tempfile.mkdtemp()
folder = os.path.join(tmp, 'A'); os.makedirs(folder)
path = os.path.join(folder, 'A_007.mp4')
vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 480))
for i in range(60):
    f = np.full((480, 640, 3), 255, np.uint8)
    cv2.rectangle(f, (220, 100), (420, 480), (40, 40, 40), -1)
    vw.write(f)
vw.release()

calls = []
def det2(rgb):
    calls.append(1); return [hand(0.3), hand(0.7)]
def det1(rgb):
    return [hand(0.5)]
def det0(rgb):
    return []

allowed = {'A': {2}, 'C': {1}}
rows, st = tf.process_clip(path, 'A', det2, allowed)
check('A (dua tangan) dengan 2 tangan dipakai', len(rows) > 0 and st['dipakai'] == len(rows))
check('maksimal 15 frame per klip', len(calls) <= tf.MAX_FRAMES)
check('baris = label + grup + 130 fitur', len(rows[0]) == 132 and rows[0][0] == 'A')
check('grup = blok 5 klip (A_007 -> blok 1)', rows[0][1] == 'tj-A-1')
rows, st = tf.process_clip(path, 'A', det1, allowed)
check('A dengan 1 tangan dibuang', not rows and st['jumlah_tangan_beda'] > 0)
rows, st = tf.process_clip(path, 'C', det1, allowed)
check('C (satu tangan) dengan 1 tangan dipakai', len(rows) > 0)
rows, st = tf.process_clip(path, 'A', det0, allowed)
check('tanpa tangan dibuang', not rows and st['tanpa_tangan'] > 0)
check('nama klip -> grup', tf.clip_group('B', 'B_001.MOV') == 'tj-B-0' and tf.clip_group('B', 'B_006.MOV') == 'tj-B-1'
      and tf.clip_group('B', 'B_073.MOV') == 'tj-B-14')
check('klip ditemukan dari folder huruf', len(tf.list_clips([tmp])) == 1)
rows, st = tf.process_clip(os.path.join(tmp, 'tidak_ada.mp4'), 'A', det2, allowed)
check('klip rusak tidak membuat error', not rows and st['klip_rusak'] == 1)
print('\nSemua tes lulus.')
