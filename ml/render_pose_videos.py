"""
Membuat VIDEO animasi kerangka tangan (mp4) untuk entri kamus yang belum punya video asli,
supaya seluruh Kamus dan Teks ke Isyarat berbasis video.

    python ml/render_pose_videos.py            # hanya yang belum punya video asli
    python ml/render_pose_videos.py --all      # semua yang punya pose

Hasil: static/videos/generated/alphabet/r.mp4, static/videos/generated/words/halo.mp4, ...
Video asli SELALU didahulukan: taruh static/videos/alphabet/r.mp4 (atau words/halo.mp4) dan
klip kerangka otomatis tidak dipakai lagi (boleh dihapus).

Syarat: ffmpeg terpasang, opencv-python, numpy. Pose berasal dari static/data/poses.json
(python ml/build_poses.py). Ini BUKAN rekaman penanda asli, hanya visualisasi data landmark.
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSES = os.path.join(BASE, 'static', 'data', 'poses.json')
VIDEOS = os.path.join(BASE, 'static', 'videos')

W, H, FPS = 480, 408, 24
FADE_S, HOLD_S = 0.4, 2.0
BG = (41, 37, 33)                                   # sama dengan bg-dark Bootstrap (BGR)
COLORS = [(216, 180, 0), (3, 183, 255)]             # BGR dari #00b4d8 dan #ffb703
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def fit(hands):
    pts = np.array([p for h in hands for p in h], float)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    pad = min(W, H) * 0.14
    s = min((W - 2 * pad) / max(x1 - x0, 0.01), (H - 2 * pad) / max(y1 - y0, 0.01))
    return s, (W - (x1 - x0) * s) / 2 - x0 * s, (H - (y1 - y0) * s) / 2 - y0 * s


def frame(hands, alpha, zoom):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = BG
    s, ox, oy = fit(hands)
    cx, cy = W / 2, H / 2
    T = lambda p: (int((p[0] * s + ox - cx) * zoom + cx), int((p[1] * s + oy - cy) * zoom + cy))
    layer = img.copy()
    for hi, hand in enumerate(hands):
        col = COLORS[hi % 2]
        for a, b in CONN:
            cv2.line(layer, T(hand[a]), T(hand[b]), col, 3, cv2.LINE_AA)
        for p in hand:
            cv2.circle(layer, T(p), 4, col, -1, cv2.LINE_AA)
    cv2.putText(layer, 'animasi kerangka tangan', (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def render(hands, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                          '-crf', '28', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                          '-movflags', '+faststart', dst], stdin=subprocess.PIPE)
    n_fade, n_hold = int(FADE_S * FPS), int(HOLD_S * FPS)
    for i in range(n_fade + n_hold):
        t = min(1.0, i / n_fade)
        ease = 1 - (1 - t) ** 3                                  # muncul halus lalu diam
        p.stdin.write(frame(hands, ease, 0.92 + 0.08 * ease).tobytes())
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    poses = json.load(open(POSES, encoding='utf-8'))
    jobs = []
    for letter, hands in poses['alphabet'].items():
        jobs.append((hands, f'alphabet/{letter.lower()}.mp4'))
    for word, hands in poses['words'].items():
        jobs.append((hands, f'words/{word.lower().replace(" ", "_")}.mp4'))
    made = 0
    for hands, rel in jobs:
        if not args.all and os.path.isfile(os.path.join(VIDEOS, rel)):
            continue                                              # sudah ada video asli
        render(hands, os.path.join(VIDEOS, 'generated', rel))
        print('dibuat: generated/' + rel)
        made += 1
    print(f'Selesai: {made} video.')


if __name__ == '__main__':
    main()
