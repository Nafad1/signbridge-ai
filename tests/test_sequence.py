"""Uji Teks ke Isyarat memakai database & video sungguhan.   Jalankan dari folder proyek: python tests/test_sequence.py"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config  # noqa: E402
import sign_sequence as S  # noqa: E402

rows = sqlite3.connect(config.DB_PATH).execute("SELECT word, category, description, video_filename FROM vocabulary").fetchall()
vocab = {w: {'category': c, 'description': d, 'video_filename': v} for w, c, d, v in rows}
poses = S.load_poses()


def seq(text):
    items, _ = S.build_sequence(text, vocab, poses)
    return items


def check(name, cond):
    print(('OK   ' if cond else 'GAGAL'), name)
    assert cond, name


# SEMUA item harus punya video (asli atau animasi kerangka) -> tidak ada layar kosong
for text in ['a', 'r', 'a b c', 'r s t', 'halo', 'ya tidak', 'terima kasih', 'senin', 'tujuh', '25',
             'di mana rumah teman', 'budi', 'ruswy', 'zebra']:
    items = seq(text)
    check(f'"{text}": {len(items)} item semua punya video', items and all(i['video'] for i in items))

# huruf tunggal diperlakukan sebagai huruf (video di alphabet/), bukan kata
it = seq('r')[0]
check('huruf tunggal bertipe letter dan memakai folder alphabet', it['type'] == 'letter' and 'alphabet/r.mp4' in it['video'])
check('klip kerangka ditandai skeleton (HALO), video asli ditandai real (A dan R)',
      seq('halo')[0]['video_kind'] == 'skeleton' and seq('a')[0]['video_kind'] == 'real' and seq('r')[0]['video_kind'] == 'real')
print('\nSemua tes lulus.')
