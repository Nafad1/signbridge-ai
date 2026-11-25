"""
Melatih model HURUF BISINDO (A-Z) dari dataset realtime-bisindo-classification.

Jalankan dari folder utama proyek:
    python ml/train_alphabet.py

Hasil:
    ml/model/alphabet_rf.pkl        -> model yang dipakai aplikasi
    ml/model/alphabet_metrics.json  -> angka akurasi (untuk laporan/README)
"""
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import alphabet_features, hands_from_dataset_row  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, 'dataset', 'bisindo', 'bisindo_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'alphabet_rf.pkl')
METRICS_FILE = os.path.join(MODEL_DIR, 'alphabet_metrics.json')
SANJAYA_CSV = os.path.join(BASE_DIR, 'dataset', 'bisindo', 'sanjaya_features.csv')       # ml/import_sanjaya_alphabet.py (foto dari atas, banyak orang)
TRUNOJOYO_CSV = os.path.join(BASE_DIR, 'dataset', 'bisindo', 'trunojoyo_features.csv')   # ml/import_trunojoyo_frames.py (video dari depan, mirip webcam)
EXTRA_SOURCES = (('Sanjaya', SANJAYA_CSV), ('Trunojoyo', TRUNOJOYO_CSV))
OLD_WEIGHT_OPTIONS = (1.0, 0.25)   # bobot data lama (1 penanda, augmentasi 50x -> banyak kembaran); dipilih lewat validasi


def new_model():
    return RandomForestClassifier(n_estimators=60, min_samples_leaf=3, n_jobs=-1, random_state=42)


def main():
    print("Memuat dataset BISINDO...")
    df = pd.read_csv(CSV_FILE)
    labels = df['label'].values
    raw = df.drop(columns='label').values
    print(f"Total baris: {len(df)} | jumlah huruf: {len(set(labels))}")

    print("Mengubah koordinat mentah menjadi fitur ternormalisasi...")
    X = np.array([alphabet_features(hands_from_dataset_row(r)) for r in raw])
    y = labels

    # --- Evaluasi 1: split acak (sama seperti notebook asli) -------------------------
    # Tiap foto asli diaugmentasi 50x, jadi split acak menaruh "kembaran" foto yang sama
    # di data latih DAN data uji. Angkanya cenderung terlalu optimis.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    acc_random = accuracy_score(yte, new_model().fit(Xtr, ytr).predict(Xte))

    # --- Evaluasi 2: holdout berurutan ----------------------------------------------
    # Baris dalam CSV tersusun per huruf lalu per foto asli. Untuk tiap huruf, 75% baris
    # pertama dipakai latih, ada jarak pemisah 4%, dan ~21% baris terakhir dipakai uji.
    # Ini mengurangi (tapi tidak menjamin menghilangkan) kebocoran antar foto yang sama.
    train_idx, test_idx = [], []
    for letter in sorted(set(labels)):
        idx = np.where(labels == letter)[0]
        n = len(idx)
        cut, gap = int(n * 0.75), int(n * 0.04)
        train_idx += list(idx[:cut - gap])
        test_idx += list(idx[cut:])
    model_h = new_model().fit(X[train_idx], y[train_idx])
    pred_h = model_h.predict(X[test_idx])
    acc_holdout = accuracy_score(y[test_idx], pred_h)

    print("\n" + "=" * 44)
    print("HASIL EVALUASI MODEL HURUF")
    print("=" * 44)
    print(f"Akurasi split acak (gaya notebook asli) : {acc_random * 100:.2f}%")
    print(f"Akurasi holdout berurutan               : {acc_holdout * 100:.2f}%")
    print("\nLaporan per huruf (holdout berurutan):")
    report = classification_report(y[test_idx], pred_h, output_dict=True, zero_division=0)
    print(classification_report(y[test_idx], pred_h, zero_division=0))
    weakest = sorted(((k, v['f1-score']) for k, v in report.items() if len(k) == 1),
                     key=lambda kv: kv[1])[:5]
    print("Huruf paling lemah:", ", ".join(f"{k} ({v:.2f})" for k, v in weakest))


    # --- Data tambahan (banyak orang): Sanjaya dan/atau Trunojoyo, jika sudah diimpor ------------------
    extras = []
    for name, path in EXTRA_SOURCES:
        if not os.path.exists(path):
            continue
        ds = pd.read_csv(path)
        if len(ds) < 500 or ds['group'].nunique() < 3:
            print(f"\nPERINGATAN: {path} hanya berisi {len(ds)} baris dan {ds['group'].nunique()} grup, diabaikan "
                  "(ulangi impornya).")
            continue
        extras.append({'name': name, 'X': ds.drop(columns=['label', 'group']).values,
                       'y': ds['label'].values.astype(str), 'g': np.array([f'{name}:{g}' for g in ds['group']])})
        print(f"\nDataset {name} ditemukan: {len(ds)} baris, {ds['group'].nunique()} grup")
    extra_metrics = {}
    old_weight = 1.0
    if extras:
        Xe = np.vstack([e['X'] for e in extras])
        ye = np.concatenate([e['y'] for e in extras])
        ge = np.concatenate([e['g'] for e in extras])

        def fit_predict(parts, test_X, w_old):
            """parts: daftar (X, y). Data lama diberi bobot w_old, data tambahan bobot 1."""
            Xtr2 = np.vstack([X] + [p[0] for p in parts])
            ytr2 = np.concatenate([y] + [p[1] for p in parts])
            wtr = np.concatenate([np.full(len(y), w_old)] + [np.ones(len(p[1])) for p in parts])
            return new_model().fit(Xtr2, ytr2, sample_weight=wtr).predict(test_X)

        # (a) Model dari data lama saja diuji pada tiap dataset baru = orang/kamera baru
        extra_metrics['accuracy_old_model_on_new'] = {}
        for e in extras:
            acc = accuracy_score(e['y'], new_model().fit(X, y).predict(e['X']))
            extra_metrics['accuracy_old_model_on_new'][e['name']] = round(float(acc), 4)
            print(f"Model data lama -> diuji pada {e['name']} (orang/kamera baru): {acc * 100:.2f}%")

        # (b) Antar-dataset: latih dengan semua KECUALI satu, uji pada yang satu itu (orang, kamera, sudut pandang baru)
        if len(extras) >= 2:
            extra_metrics['accuracy_leave_one_dataset_out'] = {}
            for e in extras:
                others = [(o['X'], o['y']) for o in extras if o is not e]
                pr = fit_predict(others, e['X'], min(OLD_WEIGHT_OPTIONS))
                acc = accuracy_score(e['y'], pr)
                rep_e = classification_report(e['y'], pr, output_dict=True, zero_division=0)
                weak_e = sorted(((k, v['f1-score']) for k, v in rep_e.items() if len(k) == 1), key=lambda kv: kv[1])[:5]
                extra_metrics['accuracy_leave_one_dataset_out'][e['name']] = round(float(acc), 4)
                print(f"Dilatih TANPA {e['name']}, diuji pada {e['name']}: {acc * 100:.2f}%  "
                      f"(lemah: {', '.join(f'{k} {v:.2f}' for k, v in weak_e)})")

        # (c) Validasi silang per grup (perangkat/sesi/blok penanda yang diuji tidak ikut dilatih)
        n_splits = min(5, len(set(ge)))
        best = None
        for w_old in OLD_WEIGHT_OPTIONS:
            pred = np.empty(len(ye), dtype=object)
            for tr, te in GroupKFold(n_splits=n_splits).split(Xe, ye, ge):
                pred[te] = fit_predict([(Xe[tr], ye[tr])], Xe[te], w_old)
            acc = accuracy_score(ye, pred)
            print(f"Validasi per grup, bobot data lama {w_old}: {acc * 100:.2f}%")
            if best is None or acc > best[0]:
                best = (acc, w_old, pred)
        acc_group, old_weight, pred_group = best
        print(f"\nDipakai: bobot data lama {old_weight} (akurasi {acc_group * 100:.2f}% pada grup yang belum pernah dilihat)")
        print("\nLaporan per huruf (validasi per grup, orang/kamera baru):")
        rep2 = classification_report(ye, pred_group.astype(str), output_dict=True, zero_division=0)
        print(classification_report(ye, pred_group.astype(str), zero_division=0))
        weak2 = sorted(((k, v['f1-score']) for k, v in rep2.items() if len(k) == 1), key=lambda kv: kv[1])[:5]
        print("Huruf paling lemah (orang/kamera baru):", ", ".join(f"{k} ({v:.2f})" for k, v in weak2))

        # Pasangan huruf yang paling sering tertukar (membantu menjawab 'kenapa D terbaca P')
        conf = Counter((a, b) for a, b in zip(ye, pred_group.astype(str)) if a != b)
        top_conf = conf.most_common(8)
        print("Paling sering tertukar (benar -> tebakan: jumlah):", ", ".join(f"{a}->{b}: {n}" for (a, b), n in top_conf))
        extra_metrics.update({
            'extra_sources': {e['name']: int(len(e['y'])) for e in extras},
            'accuracy_group_cv_new_people': round(float(acc_group), 4),
            'old_data_weight': old_weight,
            'weakest_letters_new_people': [{'letter': k, 'f1': round(float(v), 3)} for k, v in weak2],
            'most_confused_new_people': [{'true': a, 'predicted': b, 'count': int(n)} for (a, b), n in top_conf],
        })

    # --- Model final: dilatih dengan SEMUA data -------------------------------------
    print("\nMelatih model final dengan seluruh data...")
    if extras:
        final = new_model().fit(np.vstack([X, Xe]), np.concatenate([y, ye]),
                                sample_weight=np.concatenate([np.full(len(y), old_weight), np.ones(len(ye))]))
    else:
        final = new_model().fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final, MODEL_FILE, compress=3)
    size_mb = os.path.getsize(MODEL_FILE) / 1e6

    metrics = {
        'dataset_rows': int(len(df)),
        'classes': sorted(set(labels)),
        'accuracy_random_split': round(float(acc_random), 4),
        'accuracy_sequential_holdout': round(float(acc_holdout), 4),
        'weakest_letters': [{'letter': k, 'f1': round(float(v), 3)} for k, v in weakest],
        'model_size_mb': round(size_mb, 1),
        'note': ('Data lama hanya dari satu penanda; angka "orang/HP baru" (jika ada) adalah perkiraan yang lebih jujur.'
                 if extras else
                 'Semua foto berasal dari satu penanda; akurasi pada orang lain bisa lebih rendah.'),
    }
    metrics.update(extra_metrics)
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Model disimpan: {MODEL_FILE} ({size_mb:.1f} MB)")
    print(f"Metrik disimpan: {METRICS_FILE}")


if __name__ == '__main__':
    main()
