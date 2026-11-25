"""
Deteksi tangan pada VIDEO LEBAR (mis. WL-BISINDO 1920x1080 dengan penanda yang kecil di bingkai).
Detektor telapak MediaPipe memperkecil gambar ke ~192 px, sehingga tangan di bingkai lebar bisa hanya
beberapa piksel. Solusinya: deteksi pada potongan (zoom) lalu petakan kembali ke koordinat bingkai penuh,
sehingga semua fitur turunan (ukuran telapak, posisi) tetap konsisten.
"""
import numpy as np


def square_box(W, H, side_frac, cx_frac=0.5, top_frac=0.0):
    """Kotak persegi (x0, y0, x1, y1) dalam piksel; sisi = side_frac x tinggi bingkai."""
    s = int(min(side_frac * H, W, H))
    x0 = int(min(max(cx_frac * W - s / 2, 0), W - s))
    y0 = int(min(max(top_frac * H, 0), H - s))
    return x0, y0, x0 + s, y0 + s


# Strategi uji (nama -> fungsi yang menghasilkan kotak, None = bingkai penuh)
STRATEGIES = {
    'full': None,
    'zoom_lebar': lambda W, H: square_box(W, H, 0.85, 0.5, 0.08),
    'zoom_ketat': lambda W, H: square_box(W, H, 0.62, 0.5, 0.15),
}


def detect_in_box(detect, frame, box):
    """Jalankan `detect` pada potongan frame, hasilnya dipetakan ke koordinat ternormalisasi bingkai PENUH.
    x,y -> (x*cw + x0)/W ; z (berskala lebar potongan) -> z*cw/W."""
    H, W = frame.shape[:2]
    if box is None:
        return detect(frame)
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    out = []
    for h in detect(frame[y0:y1, x0:x1]):
        p = np.array(h, dtype=float)
        p[:, 0] = (p[:, 0] * cw + x0) / W
        p[:, 1] = (p[:, 1] * ch + y0) / H
        p[:, 2] = p[:, 2] * cw / W
        out.append(p)
    return out
