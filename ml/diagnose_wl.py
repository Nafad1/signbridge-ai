"""
Diagnosis: kenapa tangan di video WL banyak yang tidak terdeteksi, dan strategi apa yang paling membantu?

    python ml/diagnose_wl.py --source C:\\data\\wl

Mengambil contoh video dari tiap penanda, menjalankan deteksi tangan dengan beberapa strategi potongan
(bingkai penuh vs zoom ke tubuh bagian atas), lalu mencetak tabel dan menyimpan gambar contoh:
    dataset/wl/diagnose.jpg
Kirim TEKS tabel dan gambar itu. Hanya butuh beberapa menit, tidak mengubah data apa pun.
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from hand_detect import STRATEGIES, detect_in_box  # noqa: E402
from hand_filter import palm_size, select_hands  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = re.compile(r'signer(\d+)_label(\d+)_sample(\d+)\.mp4$', re.I)
COLORS = {'full': (0, 200, 0), 'zoom_lebar': (0, 220, 255), 'zoom_ketat': (255, 180, 0)}


def sample_videos(files, per_signer):
    by = {}
    for f in files:
        s, l, m = map(int, PATTERN.search(os.path.basename(f)).groups())
        by.setdefault(s, []).append((l, m, f))
    picked = {}
    for s, lst in sorted(by.items()):
        lst.sort()
        step = max(1, len(lst) // per_signer)
        picked[s] = [f for _, _, f in lst[::step][:per_signer]]
    return picked


def analyse(files_by_signer, detect, frames_per_video=12):
    stats, sheet_rows = {}, {}
    for signer, files in files_by_signer.items():
        st = {k: {'frames': 0, 'raw': 0, 'kept': 0, 'palms': []} for k in STRATEGIES}
        sheet = []
        for vi, path in enumerate(files):
            cap = cv2.VideoCapture(path)
            N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for fr in np.linspace(0, max(N - 1, 0), frames_per_video).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)                    # sama seperti aplikasi
                H, W = frame.shape[:2]
                draw = frame.copy() if (vi < 3 and fr == np.linspace(0, max(N - 1, 0), frames_per_video).astype(int)[frames_per_video // 2]) else None
                for name, boxfn in STRATEGIES.items():
                    box = boxfn(W, H) if boxfn else None
                    hands = detect_in_box(detect, frame, box)
                    kept, _ = select_hands(hands, W, H, max_hands=2, min_palm=config.MIN_PALM_SIZE,
                                           min_rel_palm=config.MIN_REL_PALM, dup_iou=config.DUPLICATE_IOU)
                    s = st[name]
                    s['frames'] += 1
                    s['raw'] += bool(hands)
                    s['kept'] += bool(kept)
                    s['palms'] += [palm_size(h, W, H) for h in hands]
                    if draw is not None:
                        if box:
                            cv2.rectangle(draw, box[:2], box[2:], COLORS[name], 3)
                        for h in hands:
                            pts = (h[:, :2] * [W, H]).astype(int)
                            for p in pts:
                                cv2.circle(draw, tuple(p), 5, COLORS[name], -1)
                if draw is not None:
                    cv2.putText(draw, f'signer{signer} {W}x{H}', (20, 50), 0, 1.4, (0, 0, 255), 3)
                    sheet.append(cv2.resize(draw, (480, int(480 * H / W))))
            cap.release()
        stats[signer], sheet_rows[signer] = st, sheet
    return stats, sheet_rows


def report(stats):
    print(f"\n{'penanda':8s}{'strategi':12s}{'frame':>7s}{'tangan terdeteksi':>20s}{'lolos filter':>14s}{'telapak median':>16s}")
    for signer, st in sorted(stats.items()):
        for name, s in st.items():
            n = max(s['frames'], 1)
            med = f"{np.median(s['palms']):.3f}" if s['palms'] else '-'
            print(f"{signer:<8d}{name:12s}{s['frames']:>7d}{s['raw'] / n * 100:>19.0f}%{s['kept'] / n * 100:>13.0f}%{med:>16s}")
    print(f"\n(ambang filter MIN_PALM_SIZE = {config.MIN_PALM_SIZE}; 'telapak median' = panjang telapak / tinggi bingkai)")


def save_sheet(sheet_rows, path):
    rows = []
    for signer, tiles in sorted(sheet_rows.items()):
        if not tiles:
            continue
        h = min(t.shape[0] for t in tiles)
        tiles = [t[:h] for t in tiles][:3]
        while len(tiles) < 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(tiles))
    if rows:
        w = min(r.shape[1] for r in rows)
        cv2.imwrite(path, np.vstack([r[:, :w] for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main(argv=None, detector=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--per-signer', type=int, default=8)
    args = ap.parse_args(argv)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.source) for f in fs if PATTERN.search(f))
    if not files:
        sys.exit('Tidak ada file signerX_labelN_sampleM.mp4 di folder itu.')
    if detector is None:
        from extract_wl_landmarks import make_detector
        detector = make_detector()
    stats, sheet_rows = analyse(sample_videos(files, args.per_signer), detector)
    report(stats)
    out = os.path.join(BASE, 'dataset', 'wl', 'diagnose.jpg')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_sheet(sheet_rows, out)
    print(f'\nGambar contoh: {out}')
    return stats


if __name__ == '__main__':
    main()
