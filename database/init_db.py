import sqlite3
import os

# Menentukan lokasi file database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')

def init_db():
    # Menghubungkan ke database (akan dibuat jika belum ada)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Membuat tabel kosakata
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vocabulary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        description TEXT,
        video_filename TEXT
    )
    ''')

    # Data contoh (DUMMY DATA)
    # PERINGATAN: Ini hanya data uji. Video belum tersedia.
    dummy_data = [
        ('HALO', 'Salam', 'Gerakan telapak tangan terbuka di dekat kepala, lalu digerakkan ke depan.', 'dummy.mp4'),
        ('TERIMA KASIH', 'Salam', 'Gerakan tangan dari dagu ke depan.', 'dummy.mp4'),
        ('MAAF', 'Salam', 'Gerakan tangan memutar di dada.', 'dummy.mp4'),
        ('YA', 'Kata Umum', 'Menganggukkan kepala atau gerakan tangan mengepal.', 'dummy.mp4'),
        ('TIDAK', 'Kata Umum', 'Menggelengkan kepala atau gerakan telunjuk.', 'dummy.mp4')
    ]

    # Abjad BISINDO A-Z (dari dataset realtime-bisindo-classification).
    # Deskripsi gerakan sengaja netral; isi dengan penjelasan resmi bila tersedia.
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        dummy_data.append((letter, 'Abjad', f'Huruf {letter} dalam abjad BISINDO.', None))

    # Angka 1-10 dan nama hari (video dari BISINDO Video Dataset, lihat ml/import_videos.py)
    for n in range(1, 11):
        dummy_data.append((str(n), 'Angka', f'Angka {n} dalam BISINDO.', None))
    for hari in ['SENIN', 'SELASA', 'RABU', 'KAMIS', 'JUMAT', 'SABTU', 'MINGGU']:
        dummy_data.append((hari, 'Hari', f'Hari {hari.capitalize()} dalam BISINDO.', None))

    # 32 kata dari WL-BISINDO (video: ml/import_wl_bisindo.py). MAAF & TERIMA KASIH sudah ada di atas
    # dan otomatis memakai video barunya.
    wl_words = {
        'Kata Tanya': ['apa', 'siapa', 'kapan', 'di mana', 'mengapa', 'bagaimana'],
        'Warna': ['merah', 'kuning', 'hijau', 'hitam'],
        'Waktu': ['hari', 'lagi', 'pagi', 'siang', 'sore', 'malam'],
        'Kata Kerja': ['belajar', 'cari', 'ingat', 'makan', 'dengar', 'berangkat', 'datang'],
        'Kata Benda': ['air', 'motor', 'rumah', 'teman', 'keluarga'],
        'Kata Umum': ['saya', 'tuli', 'maaf', 'terima kasih'],
    }
    for kategori, kata_list in wl_words.items():
        for kata in kata_list:
            dummy_data.append((kata.upper(), kategori, f'Kata "{kata}" dalam BISINDO (varian Banten).', None))

    # Memasukkan data ke tabel (mengabaikan jika data sudah ada)
    for data in dummy_data:
        try:
            cursor.execute('''
                INSERT INTO vocabulary (word, category, description, video_filename)
                VALUES (?, ?, ?, ?)
            ''', data)
        except sqlite3.IntegrityError:
            pass # Data sudah ada

    conn.commit()
    conn.close()
    print("Database berhasil dibuat dan diisi dengan data contoh!")

if __name__ == '__main__':
    init_db()