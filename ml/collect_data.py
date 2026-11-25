import cv2
import mediapipe as mp
import csv
import os

# 1. Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi deteksi tangan
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 2. Tentukan Label Kosakata (BISINDO)
labels_dict = {
    '1': 'HALO',
    '2': 'TERIMA KASIH',
    '3': 'MAAF',
    '4': 'YA',
    '5': 'TIDAK'
}

current_label = "BELUM DIPILIH"

# 3. Menentukan lokasi file secara otomatis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_dir = os.path.join(BASE_DIR, 'dataset', 'raw')
os.makedirs(csv_dir, exist_ok=True)
csv_file = os.path.join(csv_dir, 'data.csv')

# Cek apakah file CSV sudah ada, jika belum buat header
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y']]
        writer.writerow(header)

print("Kamera terbuka. Tekan angka 1-5 untuk memilih kata.")
print("Tekan 'S' untuk MENYIMPAN data (tahan tombol S saat tangan di posisi yang benar).")
print("Tekan 'Q' untuk keluar.")

# 4. Buka Kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data_landmarks = []
            # Ambil titik pergelangan tangan (titik 0) sebagai patokan
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            
            for lm in hand_landmarks.landmark:
                # Kurangi koordinat dengan titik pergelangan tangan
                data_landmarks.extend([lm.x - base_x, lm.y - base_y])

    # Tampilkan teks status di layar kamera
    cv2.putText(frame, f"Label: {current_label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Tekan 1-5 utk ganti kata", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Pengumpulan Data BISINDO", frame)

    # 4. Kontrol Keyboard
    key = cv2.waitKey(1) & 0xFF

    if chr(key) in labels_dict:
        current_label = labels_dict[chr(key)]
        print(f"Label diubah menjadi: {current_label}")

    elif key == ord('s') or key == ord('S'):
        if current_label == "BELUM DIPILIH":
            print("Pilih label dulu (tekan 1-5)!")
        elif results.multi_hand_landmarks:
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + data_landmarks)
            print(f"Data {current_label} tersimpan!")
        else:
            print("Tangan tidak terdeteksi, data tidak disimpan.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()