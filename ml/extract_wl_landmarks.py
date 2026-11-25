"""
Langkah 1: ubah video WL-BISINDO menjadi urutan landmark tangan.

    python ml/extract_wl_landmarks.py --source C:\\data\\wl

Hasil: dataset/wl/wl_frames.joblib  (+ cache per video di dataset/wl/cache/, jadi bisa dihentikan
dan dilanjutkan; video yang sudah diproses dilewati).

--stride 2   ambil tiap frame ke-2 (lebih cepat; 1 = semua frame)
--workers N  jumlah proses paralel (default: setengah jumlah CPU)
--no-flip    jangan cerminkan frame (default: dicerminkan, sama seperti aplikasi)

Waktu proses tergantung CPU; ada penanda kemajuan. Kebanyakan waktu habis di deteksi tangan.
"""
import argparse
import os
import re
import sys
from multiprocessing import Pool, cpu_count

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from hand_filter import select_hands  # noqa: E402
from wl_features import frame_vector  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'dataset', 'wl')
CACHE = os.path.join(OUT_DIR, 'cache')
PATTERN = re.compile(r'signer(\d+)_label(\d+)_sample(\d+)\.mp4$', re.I)

_hands = None


def make_detector():
    """Detektor tangan MediaPipe, pengaturan sama dengan aplikasi (per frame mandiri)."""
    import mediapipe as mp
    h = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                 min_detection_confidence=config.DETECTION_CONFIDENCE['alphabet'])

    def detect(frame_bgr):
        res = h.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return [np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark]) for hl in (res.multi_hand_landmarks or [])]
    return detect


def extract_video(path, detect, stride=2, flip=True):
    """Return (frames array (T,134) float32, jumlah frame diproses, jumlah frame yang ada tangannya)."""
    cap = cv2.VideoCapture(path)
    frames, total, found, i = [], 0, 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if flip:
                f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            kept, _ = select_hands(detect(f), w, h, max_hands=2, min_palm=config.MIN_PALM_SIZE,
                                   min_rel_palm=config.MIN_REL_PALM, dup_iou=config.DUPLICATE_IOU)
            frames.append(frame_vector([k['pts'] for k in kept], w, h))
            total += 1
            found += bool(kept)
        i += 1
    cap.release()
    return np.array(frames, np.float32).reshape(-1, 134), total, found


def _work(job):
    global _hands
    path, stride, flip = job
    if _hands is None:
        _hands = make_detector()
    name = os.path.basename(path)
    cache = os.path.join(CACHE, name + '.npy')
    if os.path.isfile(cache):
        return name, np.load(cache), -1, -1
    frames, total, found = extract_video(path, _hands, stride, flip)
    np.save(cache, frames)
    return name, frames, total, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='folder berisi signerX_labelN_sampleM.mp4 (boleh bersarang)')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--workers', type=int, default=max(1, cpu_count() // 2))
    ap.add_argument('--no-flip', action='store_true')
    args = ap.parse_args()

    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.source) for f in fs if PATTERN.search(f))
    if not files:
        sys.exit('Tidak ada file signerX_labelN_sampleM.mp4 di folder itu.')
    os.makedirs(CACHE, exist_ok=True)
    print(f'{len(files)} video | stride {args.stride} | proses paralel {args.workers}')

    jobs = [(p, args.stride, not args.no_flip) for p in files]
    records, no_hand = [], []
    with Pool(args.workers) as pool:
        for n, (name, frames, total, found) in enumerate(pool.imap(_work, jobs), 1):
            signer, label, sample = map(int, PATTERN.search(name).groups())
            records.append({'name': name, 'signer': signer, 'label': label, 'sample': sample, 'frames': frames})
            if total > 0 and found / total < 0.5:
                no_hand.append((name, found, total))
            if n % 25 == 0 or n == len(files):
                print(f'  {n}/{len(files)} selesai')

    out = os.path.join(OUT_DIR, 'wl_frames.joblib')
    joblib.dump({'records': records, 'stride': args.stride, 'flip': not args.no_flip}, out)
    print(f'\nTersimpan: {out}')
    if no_hand:
        print(f'Peringatan: {len(no_hand)} video tangannya terdeteksi di <50% frame (mis. {no_hand[0][0]}).')


if __name__ == '__main__':
    main()
