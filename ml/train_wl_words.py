"""
Langkah 2: latih model kata WL-BISINDO (32 kata) dari hasil ekstraksi.

    python ml/train_wl_words.py

Evaluasi memakai pembagian RESMI signer-independent dari pembuat dataset:
  latih = penanda 0-3, uji = penanda 4 (orang yang TIDAK pernah dilihat model).
Itu perkiraan yang jujur untuk orang baru di depan kamera. Model final dilatih dengan SEMUA data.

Hasil: ml/model/wl_words_rf.pkl, ml/model/wl_words_metrics.json
"""
import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_wl_bisindo import LABELS  # noqa: E402
from wl_features import SEQ_DIM, mirror_frames, sequence_features  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(BASE, 'dataset', 'wl', 'wl_frames.joblib')
SPLIT = os.path.join(BASE, 'dataset', 'wl', 'SI_split_metadata.json')
MODEL_DIR = os.path.join(BASE, 'ml', 'model')


def build_xy(records, mirror=False):
    X, y, signers, names = [], [], [], []
    for r in records:
        for frames in ([r['frames']] + ([mirror_frames(r['frames'])] if mirror else [])):
            f = sequence_features(frames)
            if f is None:
                continue
            X.append(f); y.append(r['label']); signers.append(r['signer']); names.append(r['name'])
    return np.array(X), np.array(y), np.array(signers), names


def new_model():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)


def evaluate(train_records, test_records, mirror):
    Xtr, ytr, _, _ = build_xy(train_records, mirror=mirror)
    Xte, yte, _, _ = build_xy(test_records, mirror=False)
    m = new_model().fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte)), len(Xtr), len(Xte)


def main():
    data = joblib.load(FRAMES)
    records = data['records']
    split = json.load(open(SPLIT, encoding='utf-8'))
    role = {i['video_id'] + '.mp4': i['split'] for k in split for i in k['instances']}
    train = [r for r in records if role.get(r['name']) in ('train', 'val')]
    test = [r for r in records if role.get(r['name']) == 'test']
    print(f'Video: {len(records)} | latih {len(train)} | uji (penanda baru) {len(test)}')
    if not train or not test:
        sys.exit('Pembagian latih/uji kosong: pastikan semua video sudah diekstrak (nama file sesuai).')

    print('\nEvaluasi pada penanda yang belum pernah dilihat (acak = %.1f%%):' % (100 / len(LABELS)))
    acc_plain, ntr, nte = evaluate(train, test, mirror=False)
    print(f'  tanpa augmentasi cermin : {acc_plain * 100:5.1f}%  (latih {ntr}, uji {nte})')
    acc_mirror, ntr, _ = evaluate(train, test, mirror=True)
    print(f'  dengan augmentasi cermin: {acc_mirror * 100:5.1f}%  (latih {ntr})')
    use_mirror = acc_mirror >= acc_plain

    print(f'\nMelatih model final dengan SEMUA video (augmentasi cermin: {"ya" if use_mirror else "tidak"})...')
    X, y, _, _ = build_xy(records, mirror=use_mirror)
    final = new_model().fit(X, y)
    words = {int(i): LABELS[int(i)] for i in sorted(set(y))}
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, 'wl_words_rf.pkl')
    joblib.dump({'model': final, 'words': words, 'seq_dim': SEQ_DIM, 'stride': data.get('stride'),
                 'flip': data.get('flip')}, path, compress=3)
    metrics = {'accuracy_new_signer_plain': round(float(acc_plain), 4),
               'accuracy_new_signer_mirror_aug': round(float(acc_mirror), 4),
               'mirror_aug_used': bool(use_mirror), 'chance': round(1 / len(LABELS), 4),
               'videos': len(records), 'model_size_mb': round(os.path.getsize(path) / 1e6, 1),
               'note': 'Lisensi data CC BY-NC 4.0 berlaku juga untuk model turunan (non-komersial).'}
    json.dump(metrics, open(os.path.join(MODEL_DIR, 'wl_words_metrics.json'), 'w', encoding='utf-8'), indent=2)
    print(f'Model disimpan: {path} ({metrics["model_size_mb"]} MB)')


if __name__ == '__main__':
    main()
