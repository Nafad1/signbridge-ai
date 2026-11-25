"""
Mengambil satu klip per huruf (A-Z) dari "Indonesian Sign Language Dataset: Alphabet Video"
(Indah Siradjuddin, Universitas Trunojoyo Madura, Mendeley Data V2, DOI 10.17632/p7j5jrsbbb.2, CC BY 4.0)
dan menjadikannya video kamus: static/videos/alphabet/r.mp4 ... z.mp4

    python ml/import_trunojoyo_alphabet.py --source "C:\\data\\trunojoyo_A-E" "C:\\data\\trunojoyo_F-K" "C:\\data\\trunojoyo_L-Q" "C:\\data\\trunojoyo_R-Z"

Boleh satu atau beberapa folder sumber (dataset dibagikan terpisah-pisah). Strukturnya <source>/R/R_001.MP4 ... atau
<source>/A/A_001.MOV (folder huruf dicari otomatis sampai 3 tingkat). Opsi --only R S T membatasi huruf yang diproses.
Tiap klip dipotong (crop) ke area orang agar isyaratnya besar, lalu diubah ke H.264 tanpa suara.
Klip yang dipilih ditentukan di PICKS (hasil pemeriksaan visual); ganti angkanya jika ingin klip lain.
Syarat: ffmpeg di PATH (atau pip install imageio-ffmpeg), opencv-python, numpy.
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'static', 'videos', 'alphabet')
PICKS = {'A': 1, 'B': 30, 'C': 7, 'D': 4, 'E': 4, 'F': 11, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 15,
         'L': 1, 'M': 1, 'N': 1, 'O': 4, 'P': 7, 'Q': 1,
         'R': 3, 'S': 1, 'T': 3, 'U': 3, 'V': 3, 'W': 3, 'X': 3, 'Y': 1, 'Z': 3}   # nomor klip per huruf (hasil pemeriksaan visual)
TARGET_ASPECT = 3 / 4          # lebar : tinggi hasil crop
OUT_HEIGHT = 720
PAD = 0.12                     # ruang tambahan di sekeliling orang


def find_ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_root(source, depth=3):
    source = os.path.abspath(source)
    stack = [(source, 0)]
    while stack:
        folder, d = stack.pop(0)
        if not os.path.isdir(folder):
            continue
        subs = [s for s in os.listdir(folder) if len(s) == 1 and s.isalpha() and os.path.isdir(os.path.join(folder, s))]
        if len(subs) >= 5:
            return folder
        if d < depth:
            stack += [(os.path.join(folder, s), d + 1) for s in sorted(os.listdir(folder))]
    return None


def person_box(path, samples=10, dist_thr=38.0, edge=0.03):
    """Kotak (x0, y0, x1, y1, w, h) yang memuat orang di seluruh klip.
    Warna latar diperkirakan dari dua sudut atas bingkai (latar putih, hijau-biru, atau abu-abu), lalu piksel yang
    warnanya jauh dari latar dianggap orang. Pinggir bingkai diabaikan (sebagian klip punya garis hitam di tepi)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask = np.zeros((h, w), bool)
    ph, pw = int(h * 0.12), int(w * 0.10)
    ex, ey = int(w * edge), int(h * edge)
    for i in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        corners = np.concatenate([lab[ey:ey + ph, ex:ex + pw].reshape(-1, 3),
                                  lab[ey:ey + ph, w - ex - pw:w - ex].reshape(-1, 3)])
        bg = np.median(corners, axis=0)
        mask |= np.linalg.norm(lab - bg, axis=2) > dist_thr
    cap.release()
    mask[:, :ex] = mask[:, -ex:] = False
    mask[:ey] = mask[-ey:] = False
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n_lab, lab_img, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_lab <= 1:
        return 0, 0, w, h, w, h
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[big, :4]
    return int(x), int(y), int(x + bw), int(y + bh), w, h


def crop_rect(box, aspect=TARGET_ASPECT, pad=PAD):
    x0, y0, x1, y1, w, h = box
    bw, bh = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    cw = max(bw, bh * aspect)
    ch = cw / aspect
    if ch > h:                       # tidak boleh lebih tinggi dari bingkai
        ch, cw = h, h * aspect
    if cw > w:
        cw, ch = w, w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x = min(max(cx - cw / 2, 0), w - cw)
    y = min(max(cy - ch / 2, 0), h - ch)
    return int(x) // 2 * 2, int(y) // 2 * 2, int(cw) // 2 * 2, int(ch) // 2 * 2


def convert(ffmpeg, src, dst, rect):
    x, y, cw, ch = rect
    out_h = min(OUT_HEIGHT, ch) // 2 * 2          # jangan diperbesar melebihi resolusi asli
    vf = f'crop={cw}:{ch}:{x}:{y},scale=-2:{out_h}'
    cmd = [ffmpeg, '-y', '-v', 'error', '-i', src, '-vf', vf, '-an', '-c:v', 'libx264', '-preset', 'medium',
           '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        if os.path.exists(dst):
            os.remove(dst)
        return False, r.stderr.strip()[-300:]
    return True, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, nargs='+', help='satu atau beberapa folder sumber')
    ap.add_argument('--only', nargs='+', help='proses hanya huruf ini, mis. --only R S T')
    ap.add_argument('--force', action='store_true', help='timpa video yang sudah ada')
    args = ap.parse_args()
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        sys.exit('ffmpeg tidak ditemukan. Pasang:  pip install imageio-ffmpeg')
    roots = []
    for src in args.source:
        r = find_root(src)
        if r is None:
            print(f'PERINGATAN: folder huruf tidak ditemukan di "{src}"')
        else:
            roots.append(r)
    if not roots:
        sys.exit('Tidak ada folder sumber yang valid.')
    print('Folder sumber:', *roots, sep='\n  ')
    os.makedirs(OUT, exist_ok=True)
    wanted = {x.upper() for x in args.only} if args.only else None
    for letter, num in PICKS.items():
        if wanted and letter not in wanted:
            continue
        src = None
        for root in roots:
            for ext in ('.MP4', '.mp4', '.MOV', '.mov'):
                p = os.path.join(root, letter, f'{letter}_{num:03d}{ext}')
                if os.path.exists(p):
                    src = p
                    break
            if src:
                break
        dst = os.path.join(OUT, letter.lower() + '.mp4')
        if src is None:
            print(f'{letter}: klip {num} tidak ditemukan, dilewati')
            continue
        if os.path.exists(dst) and not args.force:
            print(f'{letter}: {os.path.relpath(dst, BASE)} sudah ada, dilewati (pakai --force untuk menimpa)')
            continue
        ok, err = convert(ffmpeg, src, dst, crop_rect(person_box(src)))
        print(f'{letter}: {os.path.basename(src)} -> {os.path.relpath(dst, BASE)}  ' +
              (f'OK {os.path.getsize(dst) / 1024:.0f} KB' if ok else 'GAGAL ' + err))


if __name__ == '__main__':
    main()
