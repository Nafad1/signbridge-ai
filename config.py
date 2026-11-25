import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Database & model ---
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')
WORD_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'model', 'random_forest.pkl')        # 5 kata statis (1 tangan)
ALPHABET_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'model', 'alphabet_rf.pkl')      # huruf A-Z (1-2 tangan)
WL_WORDS_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'model', 'wl_words_rf.pkl')      # 32 kata bergerak WL (1-2 tangan)

# --- Video & pose untuk fitur Teks ke Isyarat ---
VIDEO_DIR = os.path.join(BASE_DIR, 'static', 'videos')
POSES_PATH = os.path.join(BASE_DIR, 'static', 'data', 'poses.json')

# --- Batas keyakinan minimal agar hasil prediksi ditampilkan ---
WORD_CONFIDENCE = 0.60
ALPHABET_CONFIDENCE = 0.50   # 26 kelas, jadi batasnya lebih rendah daripada model kata
WL_WORDS_CONFIDENCE = 0.25   # 32 kelas bergerak (acak = 3.1%)

# --- Pelacakan tangan (lihat ml/hand_filter.py) ---
MIN_PALM_SIZE = 0.03        # panjang telapak minimal, relatif tinggi gambar
MIN_REL_PALM = 0.5          # tangan ke-2 harus >= 50% ukuran tangan terbesar, jika tidak dianggap tangan lain/latar
DUPLICATE_IOU = 0.5         # dua kotak tangan yang tumpang tindih lebih dari ini = tangan yang sama terdeteksi 2x
EDGE_MARGIN = 0.005         # tangan yang titiknya keluar dari bingkai dianggap terpotong
DETECTION_CONFIDENCE = {'word': 0.7, 'alphabet': 0.6, 'wl_word': 0.5}
DARK_BRIGHTNESS = 60        # rata-rata kecerahan (0-255) di bawah ini = dianggap terlalu gelap