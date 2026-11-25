"""
Mengambil 1 klip per kata dari dataset WL-BISINDO (Kaggle: glennleonali/wl-bisindo) dan
menaruhnya di static/videos/words/ untuk fitur Teks ke Isyarat.

    python ml/import_wl_bisindo.py --source C:\\data\\wl-sample --signer 3 --sample 2

--source : folder berisi file bernama signer{S}_label{N}_sample{M}.mp4
Syarat   : ffmpeg terpasang.

Dataset: 32 kata, varian BISINDO Banten, lisensi CC BY-NC 4.0 (non-komersial, wajib kredit).
"""
import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'static', 'videos', 'words')

# nomor label -> kata (dari data_structuring/SD_split_metadata.json di repo WL-BISINDO)
LABELS = {0: 'air', 1: 'belajar', 2: 'cari', 3: 'hari', 4: 'ingat', 5: 'lagi', 6: 'maaf', 7: 'makan',
          8: 'motor', 9: 'saya', 10: 'terima kasih', 11: 'tuli', 12: 'apa', 13: 'siapa', 14: 'kapan',
          15: 'di mana', 16: 'mengapa', 17: 'bagaimana', 18: 'merah', 19: 'kuning', 20: 'hijau',
          21: 'hitam', 22: 'dengar', 23: 'berangkat', 24: 'datang', 25: 'teman', 26: 'keluarga',
          27: 'rumah', 28: 'pagi', 29: 'siang', 30: 'sore', 31: 'malam'}

# Kotak potong (video 1920x1080, penanda di tengah-kiri): lebar, tinggi, x, y
CROP = (1000, 980, 340, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--signer', type=int, default=3)
    ap.add_argument('--sample', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    w, h, x, y = CROP
    missing = []
    for label, word in LABELS.items():
        src = os.path.join(args.source, f'signer{args.signer}_label{label}_sample{args.sample}.mp4')
        if not os.path.isfile(src):
            missing.append(src)
            continue
        dst = os.path.join(OUT, word.replace(' ', '_') + '.mp4')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src,
                        '-vf', f'crop={w}:{h}:{x}:{y},scale=480:-2', '-an', '-c:v', 'libx264',
                        '-crf', '28', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', dst], check=True)
        print(f'{label:2d} {word:14s} -> words/{os.path.basename(dst)}')
    if missing:
        sys.exit('File tidak ditemukan:\n' + '\n'.join(missing))


if __name__ == '__main__':
    main()
