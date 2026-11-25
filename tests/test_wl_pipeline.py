"""Uji alur WL tanpa MediaPipe (detektor tiruan + video asli).   Jalankan: python tests/test_wl_pipeline.py"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'ml'))
sys.path.insert(0, ROOT)
from extract_wl_landmarks import PATTERN, extract_video  # noqa: E402
from wl_features import FRAME_DIM, SEQ_DIM, frame_vector, mirror_frames, sequence_features  # noqa: E402


def check(name, cond):
    print(('OK   ' if cond else 'GAGAL'), name)
    assert cond, name


def fake_hand(cx, cy, palm=0.2):
    p = np.zeros((21, 3)); p[0] = [cx, cy, 0]; p[9] = [cx, cy - palm, 0]
    for i in range(1, 21):
        if i != 9:
            p[i] = [cx + (i % 5 - 2) * 0.03, cy - palm - 0.02 * (i % 4), 0.01 * i]
    return p


# 1. dimensi
check('dimensi frame & urutan sesuai', len(frame_vector([fake_hand(.5, .8)], 640, 480)) == FRAME_DIM == 134 and SEQ_DIM == 1608)

# 2. cermin dua kali = asli (satu & dua tangan)
fr = np.array([frame_vector([fake_hand(.3, .8)], 640, 480), frame_vector([fake_hand(.3, .8), fake_hand(.7, .6, .15)], 640, 480)])
check('cermin dua kali mengembalikan data asli', np.allclose(mirror_frames(mirror_frames(fr)), fr, atol=1e-9))
check('cermin memindahkan tangan kiri ke sisi kanan', abs(mirror_frames(fr)[0, 63] - 0.7) < 1e-9)

# 3. fitur urutan
check('video tanpa tangan -> None', sequence_features(np.zeros((10, FRAME_DIM))) is None)
seq = np.array([frame_vector([fake_hand(.3 + .02 * t, .8)], 640, 480) for t in range(25)])
seq[:3] = 0; seq[10:12] = 0                      # kosong di awal dan di tengah
f = sequence_features(seq)
check('urutan dengan frame kosong tetap valid (1608 angka, tanpa NaN)', f.shape == (SEQ_DIM,) and np.isfinite(f).all())
check('panjang video berbeda -> panjang fitur sama', sequence_features(seq[:15]).shape == f.shape)

# 4. ekstraksi dari video asli dengan detektor tiruan (menguji baca video, stride, flip, filter, nama file)
clips = sorted(glob.glob(os.path.join(ROOT, 'static', 'videos', 'words', '*.mp4')))
n = {'c': 0}
def fake_detect(frame):
    n['c'] += 1
    return [fake_hand(.4 + .001 * n['c'], .8)]
frames, total, found = extract_video(clips[0], fake_detect, stride=2, flip=True)
check(f'ekstraksi video: {total} frame diproses, semua berisi tangan', frames.shape == (total, FRAME_DIM) and found == total > 5)
check('nama file Kaggle dikenali', PATTERN.search('signer3_label23_sample4.mp4').groups() == ('3', '23', '4'))

# 5. pelatihan: kelas sintetis yang berbeda lintasan harus bisa dipisahkan oleh model
from sklearn.ensemble import RandomForestClassifier
rng = np.random.default_rng(0)
def synth(cls, signer):
    shift = 0.03 * signer
    return np.array([frame_vector([fake_hand(.3 + .1 * cls * t / 20 + shift + rng.normal(0, .005), .8 - .02 * cls)], 640, 480) for t in range(20)])
X, y, sg = [], [], []
for cls in range(5):
    for signer in range(5):
        for _ in range(6):
            X.append(sequence_features(synth(cls, signer))); y.append(cls); sg.append(signer)
X, y, sg = np.array(X), np.array(y), np.array(sg)
m = RandomForestClassifier(n_estimators=100, random_state=0).fit(X[sg < 4], y[sg < 4])
acc = (m.predict(X[sg == 4]) == y[sg == 4]).mean()
check(f'model memisahkan kelas sintetis pada penanda baru ({acc * 100:.0f}% > acak 20%)', acc > 0.8)
# 6. deteksi pada potongan dipetakan kembali ke koordinat bingkai penuh
from hand_detect import detect_in_box, square_box
from hand_filter import palm_size
frame = np.zeros((1080, 1920, 3), np.uint8)
box = square_box(1920, 1080, 0.85, 0.5, 0.08)
mapped = detect_in_box(lambda sub: [fake_hand(0.5, 0.5, 0.25)], frame, box)[0]
cx, cy = (box[0] + box[2]) / 2 / 1920, (box[1] + box[3]) / 2 / 1080
check('pusat potongan -> pusat yang sama di bingkai penuh', abs(mapped[0, 0] - cx) < 1e-9 and abs(mapped[0, 1] - cy) < 1e-9)
side = box[2] - box[0]
check('ukuran telapak konsisten dengan bingkai penuh', abs(palm_size(mapped, 1920, 1080) - 0.25 * side / 1080) < 1e-9)
check('tanpa kotak = bingkai penuh tidak diubah', np.allclose(detect_in_box(lambda f: [fake_hand(.3, .6)], frame, None)[0][0], [.3, .6, 0]))

print('\nSemua tes lulus.')
