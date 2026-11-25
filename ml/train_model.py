import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. MENENTUKAN LOKASI FILE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_file = os.path.join(BASE_DIR, 'dataset', 'raw', 'data.csv')
model_dir = os.path.join(BASE_DIR, 'ml', 'model')
model_file = os.path.join(model_dir, 'random_forest.pkl')

print("Memuat dataset...")
# --- 2. MEMBACA DATASET ---
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print("Error: File data.csv tidak ditemukan. Pastikan Anda sudah mengumpulkan data.")
    exit()

# Cek apakah datanya kosong (hanya berisi header)
if len(df) == 0:
    print("Error: Dataset kosong. Silakan kumpulkan data terlebih dahulu.")
    exit()

print(f"Total data yang ditemukan: {len(df)} baris.")

# --- 3. MEMISAHKAN FITUR (X) DAN LABEL (y) ---
# X adalah koordinat tangan (semua kolom KECUALI kolom 'label')
X = df.drop('label', axis=1)
# y adalah jawaban (kolom 'label')
y = df['label']

# --- 4. MEMBAGI DATA TRAINING DAN TESTING ---
# 80% untuk latihan, 20% untuk ujian (menguji akurasi)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Melatih model Random Forest...")
# --- 5. MELATIH MODEL ---
# n_estimators=100 artinya kita membuat 100 "pohon keputusan" untuk voting
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 6. MENGUJI MODEL ---
y_pred = model.predict(X_test)

# --- 7. MENAMPILKAN HASIL EVALUASI ---
print("\n" + "="*40)
print("HASIL EVALUASI MODEL")
print("="*40)
print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nLaporan Klasifikasi (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- 8. MENYIMPAN MODEL ---
# Membuat folder model jika belum ada
os.makedirs(model_dir, exist_ok=True)
# Menyimpan model ke dalam file .pkl
joblib.dump(model, model_file)
print(f"\nModel berhasil disimpan di: {model_file}")