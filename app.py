from flask import Flask, render_template, request, jsonify
import sqlite3
import cv2
import numpy as np
import base64

import config
from ml.predict import SignPredictor
import sign_sequence

app = Flask(__name__)

# Muat model AI & MediaPipe saat server dinyalakan
predictor = SignPredictor()
POSES = sign_sequence.load_poses()


def get_vocabulary():
    """Ambil seluruh kosakata dari database -> dict {KATA: {...}}."""
    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute("SELECT word, category, description, video_filename FROM vocabulary").fetchall()
    conn.close()
    return {w: {'category': c, 'description': d, 'video_filename': v} for w, c, d, v in rows}


# --- ROUTING HALAMAN ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/translator')
def translator(): return render_template('translator.html')

@app.route('/text-to-sign')
def text_to_sign(): return render_template('text_to_sign.html')

@app.route('/dictionary')
def dictionary():
    vocab = get_vocabulary()
    words = []
    for word in sorted(vocab):
        e = vocab[word]
        is_letter = e['category'] == 'Abjad'
        video = sign_sequence.letter_video(word) if is_letter else sign_sequence.word_video(word, e['video_filename'])
        pool = POSES.get('alphabet' if is_letter else 'words', {})
        words.append({'word': word, 'category': e['category'], 'description': e['description'],
                      'video': video, 'kind': sign_sequence.video_kind(video), 'pose_key': word if word in pool else None,
                      'pose_group': 'alphabet' if is_letter else 'words'})
    return render_template('dictionary.html', words=words)

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/about')
def about():
    return render_template('about.html')


# --- API TEKS KE ISYARAT ---
@app.route('/api/text-to-sign')
def api_text_to_sign():
    text = request.args.get('text', '').strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'Teks masih kosong.'})
    items, skipped = sign_sequence.build_sequence(text, get_vocabulary(), POSES)
    if not items:
        return jsonify({'status': 'error', 'message': 'Tidak ada huruf atau kata yang dapat diterjemahkan.'})
    return jsonify({
        'status': 'success',
        'items': items,
        'skipped': skipped,
        'all_video': all(i['video'] for i in items),
        'any_pose_only': any(not i['video'] for i in items),
        'skeleton_labels': sorted({i['label'] for i in items if i.get('video_kind') == 'skeleton'}),
    })


# --- API PREDIKSI MACHINE LEARNING ---
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    if 'image' not in data:
        return jsonify({'status': 'error', 'message': 'Gambar tidak ditemukan'})

    mode = data.get('mode', 'word')
    if mode not in ('word', 'alphabet'):
        mode = 'word'

    # Decode gambar Base64 dari Javascript
    img_bytes = base64.b64decode(data['image'].split(',')[1])
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Gambar tidak valid'})

    # Balik gambar (JS mengirim gambar asli; data latih direkam dalam tampilan cermin)
    frame = cv2.flip(frame, 1)
    return jsonify(predictor.predict_frame(frame, mode))


if __name__ == '__main__':
    app.run(debug=True)