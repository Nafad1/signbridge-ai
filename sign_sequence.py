"""
Mengubah teks menjadi urutan isyarat untuk fitur Teks ke Isyarat.

Aturan:
  1. Kata/frasa yang ada di kamus (tabel vocabulary) -> 1 isyarat kata.
  2. Kata lain -> dieja huruf demi huruf (abjad BISINDO).
  3. Tiap item memakai VIDEO jika filenya ada di static/videos/, kalau tidak
     memakai pose kerangka tangan dari poses.json (animasi cadangan).

Penamaan file video (huruf kecil, spasi -> garis bawah):
  static/videos/alphabet/a.mp4 ... z.mp4
  static/videos/words/halo.mp4, terima_kasih.mp4, ...
  (atau nama file yang tertulis di kolom video_filename database, di static/videos/)
"""
import json
import os
import re

import config

MAX_PHRASE_WORDS = 3

# "satu" -> angka 1, dst. (angka 1-10 punya video; lihat ml/import_videos.py)
NUMBER_WORDS = {'SATU': '1', 'DUA': '2', 'TIGA': '3', 'EMPAT': '4', 'LIMA': '5',
                'ENAM': '6', 'TUJUH': '7', 'DELAPAN': '8', 'SEMBILAN': '9', 'SEPULUH': '10'}


def load_poses():
    try:
        with open(config.POSES_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'alphabet': {}, 'words': {}}


def _video_url(rel_path, allow_generated=True):
    """Video asli didahulukan; jika tidak ada, pakai klip animasi kerangka di static/videos/generated/
    (dibuat oleh ml/render_pose_videos.py)."""
    if not rel_path:
        return None
    for prefix in ('', 'generated/') if allow_generated else ('',):
        if os.path.isfile(os.path.join(config.VIDEO_DIR, prefix + rel_path)):
            return '/static/videos/' + (prefix + rel_path).replace(os.sep, '/')
    return None


def video_kind(url):
    """'real' = rekaman penanda, 'skeleton' = animasi kerangka tangan, None = tidak ada video."""
    if not url:
        return None
    return 'skeleton' if '/static/videos/generated/' in url else 'real'


def letter_video(letter):
    return _video_url(f'alphabet/{letter.lower()}.mp4')


def word_video(word, db_filename=None):
    return _video_url(db_filename, allow_generated=False) or _video_url(f'words/{word.lower().replace(" ", "_")}.mp4')


def tokenize(text):
    """Huruf besar, buang tanda baca, pisah per kata."""
    return re.findall(r"[A-Z0-9]+", text.upper())


def build_sequence(text, vocabulary, poses):
    """
    vocabulary : dict {KATA: {'category':..., 'description':..., 'video_filename':...}}
    Return (items, skipped)
    """
    tokens = [NUMBER_WORDS.get(t, t) for t in tokenize(text)]
    items, skipped = [], []
    group = 0
    i = 0
    while i < len(tokens):
        matched = None
        for n in range(min(MAX_PHRASE_WORDS, len(tokens) - i), 0, -1):
            phrase = ' '.join(tokens[i:i + n])
            if phrase in vocabulary:
                matched = (phrase, n)
                break
        if matched:
            phrase, n = matched
            entry = vocabulary[phrase]
            if entry.get('category') == 'Abjad':
                # Huruf tunggal ("r", "a b c") ada di tabel vocabulary, tetapi videonya di alphabet/, bukan words/
                items.append({
                    'type': 'letter', 'label': phrase, 'group': group, 'word': phrase,
                    'video': (v := letter_video(phrase)), 'video_kind': video_kind(v),
                    'pose_key': phrase if phrase in poses.get('alphabet', {}) else None,
                })
            else:
                items.append({
                    'type': 'word', 'label': phrase, 'group': group,
                    'video': (v := word_video(phrase, entry.get('video_filename'))), 'video_kind': video_kind(v),
                    'pose_key': phrase if phrase in poses.get('words', {}) else None,
                    'description': entry.get('description'),
                })
            i += n
        else:
            word = tokens[i]
            for ch in word:
                if 'A' <= ch <= 'Z':
                    items.append({
                        'type': 'letter', 'label': ch, 'group': group, 'word': word,
                        'video': (v := letter_video(ch)), 'video_kind': video_kind(v),
                        'pose_key': ch if ch in poses.get('alphabet', {}) else None,
                    })
                elif ch in vocabulary:     # angka bertingkat, mis. "25" -> 2 lalu 5
                    items.append({
                        'type': 'word', 'label': ch, 'group': group,
                        'video': (v := word_video(ch, vocabulary[ch].get('video_filename'))), 'video_kind': video_kind(v),
                        'pose_key': None, 'description': vocabulary[ch].get('description'),
                    })
                else:
                    skipped.append(ch)   # angka 0 belum didukung
            i += 1
        group += 1
    return items, sorted(set(skipped))
