"""Uji mode kata gerak WL tanpa kamera & tanpa MediaPipe asli (detektor tiruan).
Jalankan:  python tests/test_wl_clip.py
"""
import base64
import os
import sys
import types

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'ml'))


def check(name, cond):
    print(('OK   ' if cond else 'GAGAL'), name)
    assert cond, name


def fake_hand(cx, cy, palm=0.2):
    p = np.zeros((21, 3)); p[0] = [cx, cy, 0]; p[9] = [cx, cy - palm, 0]
    for i in range(1, 21):
        if i != 9:
            p[i] = [cx + (i % 5 - 2) * 0.03, cy - palm - 0.02 * (i % 4), 0.01 * i]
    return p


# ---- MediaPipe tiruan: mengembalikan satu tangan yang bergerak, atau tidak ada tangan jika gambar gelap
state = {'n': 0}


class _LM:
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z


class FakeHands:
    def __init__(self, *a, **k): pass

    def process(self, rgb):
        if rgb.mean() < 20:                       # gambar hitam = tidak ada tangan
            return types.SimpleNamespace(multi_hand_landmarks=None)
        state['n'] += 1
        pts = fake_hand(0.3 + 0.02 * (state['n'] % 20), 0.7)
        return types.SimpleNamespace(multi_hand_landmarks=[types.SimpleNamespace(landmark=[_LM(*p) for p in pts])])


import mediapipe as mp  # noqa: E402
mp.solutions = types.SimpleNamespace(hands=types.SimpleNamespace(Hands=FakeHands))

import config  # noqa: E402
import app as appmod  # noqa: E402

pred = appmod.predictor
check('model WL termuat', pred.available('wl'))
words = pred.wl_words()
check(f'32 kata dikenali ({len(words)})', len(words) == 32 and 'AIR' in words and 'DI MANA' in words)


def jpeg(brightness):
    img = np.full((480, 640, 3), brightness, np.uint8)
    img[::7] = 255 if brightness > 0 else 0       # sedikit tekstur agar JPEG valid
    ok, buf = cv2.imencode('.jpg', img)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


client = appmod.app.test_client()

# 1. rekaman normal -> 3 tebakan, urut turun, jumlah peluang <= 1
res = client.post('/api/predict-clip', json={'images': [jpeg(128)] * config.WL_CLIP_FRAMES}).get_json()
check('klip dengan tangan menghasilkan 3 tebakan', res.get('status') in ('success', 'uncertain') and len(res['top']) == 3)
c = [t['confidence'] for t in res['top']]
check('tebakan urut dari yang paling yakin & peluang masuk akal', c == sorted(c, reverse=True) and 0 < sum(c) <= 1.0001)
check('semua frame bertangan terhitung', res['hand_frames'] == res['total_frames'] == config.WL_CLIP_FRAMES)
check('kata hasil berupa huruf besar dari daftar model', all(t['word'] in words for t in res['top']))
if res['status'] == 'success':
    check('status success membawa word & confidence', res['word'] == res['top'][0]['word'])
else:
    check('status uncertain membawa pesan', 'message' in res)

# 2. tidak ada tangan sama sekali
res = client.post('/api/predict-clip', json={'images': [jpeg(0)] * 10}).get_json()
check('tanpa tangan -> no_hand (bukan tebakan asal)', res['status'] == 'no_hand' and 'top' not in res)
check('gambar gelap diberi pesan cahaya', 'gelap' in res['message'].lower())

# 3. hanya 2 frame bertangan (di bawah WL_MIN_HAND_FRAMES) -> ditolak
res = client.post('/api/predict-clip', json={'images': [jpeg(128)] * 2 + [jpeg(0)] * 12}).get_json()
check('terlalu sedikit frame bertangan -> no_hand', res['status'] == 'no_hand' and res['hand_frames'] == 2)

# 4. tangan hanya di tengah rekaman (awal/akhir kosong) tetap dikenali
res = client.post('/api/predict-clip', json={'images': [jpeg(0)] * 4 + [jpeg(128)] * 8 + [jpeg(0)] * 3}).get_json()
check('frame kosong di awal/akhir dibuang, sisanya diprediksi', res['status'] in ('success', 'uncertain') and res['hand_frames'] == 8)

# 5. permintaan buruk
check('tanpa gambar -> error', client.post('/api/predict-clip', json={}).get_json()['status'] == 'error')
check('terlalu banyak gambar -> error', client.post('/api/predict-clip', json={'images': [jpeg(128)] * (config.WL_MAX_CLIP_FRAMES + 1)}).get_json()['status'] == 'error')
check('gambar rusak -> error', client.post('/api/predict-clip', json={'images': ['data:image/jpeg;base64,AAAA']}).get_json()['status'] == 'error')

# 6. pratinjau mode wl lewat /api/predict
res = client.post('/api/predict', json={'image': jpeg(128), 'mode': 'wl'}).get_json()
check('pratinjau WL: status preview + titik tangan, tanpa tebakan', res['status'] == 'preview' and res['hands'] and 'word' not in res)

# 7. mode lama tidak terganggu
res = client.post('/api/predict', json={'image': jpeg(128), 'mode': 'alphabet'}).get_json()
check('mode huruf masih berjalan', res['status'] in ('success', 'uncertain', 'hand_cut', 'no_hand'))

res = client.post('/api/predict', json={'image': jpeg(128), 'mode': 'alphabet'}).get_json()
if res['status'] in ('success', 'uncertain'):
    c = [x['confidence'] for x in res['top']]
    check('mode huruf menyertakan 3 tebakan teratas & jumlah tangan untuk diagnosis',
          len(res['top']) == 3 and c == sorted(c, reverse=True) and res['n_hands'] == 1)

# 8. halaman Penerjemah memuat opsi & tombol WL
html = client.get('/translator').get_data(as_text=True)
check('halaman Penerjemah punya opsi mode WL, tombol Rekam & daftar kata', 'value="wl"' in html and 'id="rec-btn"' in html and 'DI MANA' in html)

# 9. ambang ukuran tangan terpisah
check('ambang webcam (0.06) dan WL (0.03) terpisah', config.MIN_PALM_SIZE == 0.06 and config.WL_MIN_PALM_SIZE == 0.03)

print('\nSemua tes lulus.')
