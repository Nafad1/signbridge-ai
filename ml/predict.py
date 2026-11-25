"""
Prediksi isyarat dari frame kamera. Dipakai oleh app.py.
Tiga mode:
  - 'word'     : 5 kosakata, 1 tangan (model lama random_forest.pkl)
  - 'alphabet' : huruf A-Z, 1-2 tangan (model baru alphabet_rf.pkl dari dataset BISINDO)
  - 'wl'       : 32 kata bergerak (WL-BISINDO, wl_words_rf.pkl). Berbeda dari dua mode di atas: yang dikenali
                 adalah KLIP pendek (sekitar 2,5 detik), bukan satu frame. Lihat predict_clip().

Perbaikan pelacakan:
  - static_image_mode=True : tiap frame dideteksi mandiri. Frame dikirim browser satu-satu
    (bukan video berurutan) dan detektor dipakai bersama banyak pengguna, jadi mode "tracking"
    bisa membawa sisa frame sebelumnya / sesi lain.
  - Selalu mendeteksi sampai 2 tangan lalu memilih sendiri (ml/hand_filter.py): tangan latar,
    deteksi ganda, dan tangan terpotong tidak ikut masuk ke model.
"""
import os
import sys
import threading

import cv2
import joblib
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from extract_features import alphabet_features, word_features  # noqa: E402
from hand_filter import is_cut, select_hands  # noqa: E402
from wl_features import frame_vector, sequence_features  # noqa: E402

MAX_HANDS = {'word': 1, 'alphabet': 2, 'wl': 2}
CONF_KEY = {'word': 'word', 'alphabet': 'alphabet', 'wl': 'wl_word'}   # kunci di config.DETECTION_CONFIDENCE


def _load(path, name):
    try:
        model = joblib.load(path)
        print(f"Model {name} berhasil dimuat!")
        return model
    except Exception as e:
        print(f"Peringatan: model {name} tidak dapat dimuat. Error: {e}")
        return None


def _overlay(kept, dropped):
    """Titik tangan untuk digambar di browser (koordinat 0-1 pada gambar yang sudah dicerminkan)."""
    out = []
    for group, flag in ((kept, True), (dropped, False)):
        for h in group:
            out.append({'kept': flag, 'reason': h.get('reason'),
                        'points': [[round(float(x), 3), round(float(y), 3)] for x, y, _ in h['pts']]})
    return out


class SignPredictor:
    def __init__(self):
        self.models = {
            'word': _load(config.WORD_MODEL_PATH, 'kata'),
            'alphabet': _load(config.ALPHABET_MODEL_PATH, 'huruf'),
            'wl': _load(config.WL_WORDS_MODEL_PATH, 'kata gerak WL'),
        }
        mp_hands = mp.solutions.hands
        self.detectors = {
            mode: mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                 min_detection_confidence=config.DETECTION_CONFIDENCE[CONF_KEY[mode]])
            for mode in ('word', 'alphabet', 'wl')
        }
        self.threshold = {'word': config.WORD_CONFIDENCE, 'alphabet': config.ALPHABET_CONFIDENCE}
        self._lock = threading.Lock()   # MediaPipe tidak aman dipakai banyak thread sekaligus

    def available(self, mode):
        return mode in self.models and self.models[mode] is not None

    def wl_words(self):
        """Daftar kata yang dikenali model WL (kosong jika modelnya belum ada)."""
        if not self.available('wl'):
            return []
        return sorted(w.upper() for w in self.models['wl']['words'].values())

    @staticmethod
    def _min_palm(mode):
        return config.WL_MIN_PALM_SIZE if mode == 'wl' else config.MIN_PALM_SIZE

    def predict_frame(self, frame_bgr, mode='word'):
        """frame_bgr sudah dibalik (mirror) seperti saat pengumpulan data."""
        if not self.available(mode):
            msg = ('Model kata gerak belum ada. Jalankan: python ml/train_wl_words.py'
                   if mode == 'wl' else 'Model AI belum siap.')
            return {'status': 'error', 'message': msg}

        h, w = frame_bgr.shape[:2]
        gray_mean = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            results = self.detectors[mode].process(rgb)

        raw = [np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark])
               for hl in (results.multi_hand_landmarks or [])]
        kept, dropped = select_hands(
            raw, w, h, max_hands=MAX_HANDS[mode], min_palm=self._min_palm(mode),
            min_rel_palm=config.MIN_REL_PALM, dup_iou=config.DUPLICATE_IOU)
        overlay = _overlay(kept, dropped)

        if not kept:
            msg = 'Tangan tidak terdeteksi'
            if gray_mean < config.DARK_BRIGHTNESS:
                msg = 'Gambar terlalu gelap, tambah cahaya'
            return {'status': 'no_hand', 'message': msg, 'hands': overlay}
        if mode == 'wl':
            # Mode WL tidak memprediksi per frame; frame ini hanya untuk pratinjau (membantu memposisikan tangan).
            cut = any(is_cut(k['pts'], config.EDGE_MARGIN) for k in kept)
            msg = 'Tangan terpotong bingkai (boleh saat gerakan, tapi usahakan tetap terlihat)' if cut \
                else f'{len(kept)} tangan terlihat. Tekan "Rekam kata" untuk mulai.'
            return {'status': 'preview', 'message': msg, 'hands': overlay}
        if any(is_cut(k['pts'], config.EDGE_MARGIN) for k in kept):
            return {'status': 'hand_cut', 'message': 'Tangan terpotong bingkai, geser ke tengah', 'hands': overlay}

        result = self.predict_landmarks([k['pts'] for k in kept], mode, w, h)
        result['hands'] = overlay
        return result

    def predict_landmarks(self, hands, mode, width=1.0, height=1.0):
        """Terpisah dari deteksi MediaPipe supaya mudah diuji."""
        model = self.models[mode]
        if mode == 'word':
            feat = word_features(hands[0])
        else:
            feat = alphabet_features(hands, width, height)

        probs = model.predict_proba([feat])[0]
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        order = np.argsort(probs)[::-1][:3]
        # 'top' dan 'n_hands' membantu diagnosis di layar (mis. kenapa D terbaca P: seberapa tipis selisihnya)
        extra = {'n_hands': len(hands),
                 'top': [{'word': str(model.classes_[i]), 'confidence': float(probs[i])} for i in order]}
        if confidence > self.threshold[mode]:
            return {'status': 'success', 'word': str(model.classes_[best]),
                    'confidence': confidence, 'mode': mode, **extra}
        return {'status': 'uncertain', 'message': 'Gerakan belum dapat dikenali', **extra}

    # ------------------------------------------------------------------ mode WL (klip)
    def predict_clip(self, frames_bgr):
        """frames_bgr: daftar gambar (sudah dicerminkan, urut waktu). Deteksi tangan per frame lalu tebak kata."""
        if not self.available('wl'):
            return {'status': 'error', 'message': 'Model kata gerak belum ada. Jalankan: python ml/train_wl_words.py'}
        per_frame, brightness = [], []
        for f in frames_bgr:
            h, w = f.shape[:2]
            brightness.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
            with self._lock:
                results = self.detectors['wl'].process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            raw = [np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark])
                   for hl in (results.multi_hand_landmarks or [])]
            per_frame.append((raw, w, h))
        res = self.predict_clip_landmarks(per_frame)
        if res['status'] == 'no_hand' and brightness and float(np.mean(brightness)) < config.DARK_BRIGHTNESS:
            res['message'] = 'Gambar terlalu gelap, tambah cahaya'
        return res

    def predict_clip_landmarks(self, per_frame, top_k=3):
        """per_frame: daftar (hands, width, height). Terpisah dari MediaPipe supaya mudah diuji.
        Alurnya sama persis dengan ekstraksi data latih (ml/extract_wl_landmarks.py): filter tangan ->
        frame_vector -> sequence_features."""
        vectors, with_hand = [], 0
        for raw, w, h in per_frame:
            kept, _ = select_hands(raw, w, h, max_hands=2, min_palm=config.WL_MIN_PALM_SIZE,
                                   min_rel_palm=config.MIN_REL_PALM, dup_iou=config.DUPLICATE_IOU)
            vectors.append(frame_vector([k['pts'] for k in kept], w, h))
            with_hand += bool(kept)
        if with_hand < config.WL_MIN_HAND_FRAMES:
            return {'status': 'no_hand', 'message': 'Tangan hampir tidak terlihat selama rekaman. Coba lagi.',
                    'hand_frames': with_hand, 'total_frames': len(per_frame)}
        feat = sequence_features(np.array(vectors))
        bundle = self.models['wl']
        model, words = bundle['model'], bundle['words']
        probs = model.predict_proba([feat])[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [{'word': words[int(model.classes_[i])].upper(), 'confidence': float(probs[i])} for i in order]
        status = 'success' if top[0]['confidence'] >= config.WL_WORDS_CONFIDENCE else 'uncertain'
        out = {'status': status, 'mode': 'wl', 'top': top, 'hand_frames': with_hand, 'total_frames': len(per_frame)}
        if status == 'success':
            out.update({'word': top[0]['word'], 'confidence': top[0]['confidence']})
        else:
            out['message'] = 'Belum yakin dengan gerakannya'
        return out
