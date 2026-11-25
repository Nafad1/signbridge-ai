document.addEventListener("DOMContentLoaded", function() {
    // Hanya jalankan script ini jika kita berada di halaman translator
    const video = document.getElementById('webcam');
    if (!video) return; // Jika elemen video tidak ada, hentikan script

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultText = document.getElementById('result-text');

    let stream = null;

    // Fungsi Menyalakan Kamera
    async function startCamera() {
        try {
            // Meminta izin akses kamera ke browser
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            video.srcObject = stream;
            
            // Ubah status tombol
            startBtn.classList.add('d-none');
            stopBtn.classList.remove('d-none');
        } catch (err) {
            alert("Gagal mengakses kamera: " + err.message + "\nPastikan Anda telah memberikan izin kamera di browser.");
            console.error(err);
        }
    }

    // Fungsi Mematikan Kamera
    function stopCamera() {
        if (stream) {
            // Hentikan semua jalur video
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            
            // Ubah status tombol
            startBtn.classList.remove('d-none');
            stopBtn.classList.add('d-none');
        }
    }

    // Fungsi Menghapus Teks Hasil
    function clearResult() {
        resultText.innerText = "-";
    }

    // Event Listener Tombol
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    clearBtn.addEventListener('click', clearResult);
});

// (Logika Teks ke Isyarat ada di sign_player.js)

// --- LOGIKA UNTUK FILTER KAMUS ---
const filterInput = document.getElementById('filter-input');
if (filterInput) {
    filterInput.addEventListener('keyup', function() {
        const filterValue = this.value.toLowerCase();
        const dictionaryItems = document.querySelectorAll('.dictionary-item');
        const noResult = document.getElementById('no-result');
        let foundCount = 0;

        dictionaryItems.forEach(function(item) {
            // Mengambil teks judul kata (h3)
            const wordText = item.querySelector('h3').innerText.toLowerCase();
            
            // Jika kata cocok dengan ketikan, tampilkan. Jika tidak, sembunyikan.
            if (wordText.includes(filterValue)) {
                item.classList.remove('d-none');
                foundCount++;
            } else {
                item.classList.add('d-none');
            }
        });

        // Tampilkan pesan "tidak ditemukan" jika hasilnya 0
        if (foundCount === 0) {
            noResult.classList.remove('d-none');
        } else {
            noResult.classList.add('d-none');
        }
    });
}

// --- LOGIKA PREDIKSI AI (SIGN TO TEXT) ---
// Dibungkus DOMContentLoaded agar tracking.js (dimuat setelah file ini) sudah tersedia.
document.addEventListener('DOMContentLoaded', function () {
    const videoElement = document.getElementById('webcam');
    if (!videoElement || !window.Tracking) return;

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const overlay = document.getElementById('overlay');
    const resultTextAI = document.getElementById('result-text');
    const statusText = document.getElementById('status-text');
    const modeSelect = document.getElementById('mode-select');
    const recBtn = document.getElementById('rec-btn');
    const wlInfo = document.getElementById('wl-info');

    const SEND_DELAY_MS = 250;   // jeda setelah jawaban server (bukan interval tetap, supaya tidak menumpuk)
    const smoother = new Tracking.PredictionSmoother({ window: 5, minVotes: 3, clearAfterMiss: 3 });
    let running = false;
    let timer = null;
    let capturing = false;       // true selama merekam klip kata gerak (pratinjau dijeda)
    let session = 0;             // naik tiap clearDisplay/stop, untuk membuang jawaban usang

    const STATUS_TEXT = {
        success: 'Terdeteksi',
        uncertain: 'Gerakan belum jelas, tahan sebentar',
        no_hand: 'Tangan tidak terdeteksi',
        hand_cut: 'Tangan terpotong bingkai, geser ke tengah',
        error: 'Terjadi kesalahan'
    };

    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const isWL = () => modeSelect.value === 'wl';

    function render(state, res) {
        resultTextAI.innerText = state.label || '-';
        resultTextAI.style.color = '#00b4d8';
        let msg = res && res.message ? res.message : (STATUS_TEXT[state.status] || '');
        if (state.status === 'success' && !state.label) msg = 'Menstabilkan...';
        // Info diagnosis: jumlah tangan dan 3 tebakan teratas frame ini (membantu melihat huruf yang hampir tertukar)
        if (res && res.top) {
            const pct = x => Math.round(x * 100) + '%';
            msg += '  [' + res.n_hands + ' tangan: ' + res.top.map(t => t.word + ' ' + pct(t.confidence)).join(' | ') + ']';
        }
        statusText.innerText = msg;
    }

    // Ambil satu frame kamera sebagai JPEG (rasio asli dipertahankan, lebar maksimal 640)
    function grabFrame(quality) {
        const vw = videoElement.videoWidth, vh = videoElement.videoHeight;
        if (videoElement.readyState < 2 || !vw) return null;
        const w = Math.min(640, vw), h = Math.round(w * vh / vw);
        canvas.width = w; canvas.height = h;
        ctx.drawImage(videoElement, 0, 0, w, h);
        return canvas.toDataURL('image/jpeg', quality);
    }

    // Mode huruf / 5 kata: satu gambar per permintaan. Mode WL: hanya pratinjau tangan.
    async function sendFrame() {
        if (!running) return;
        if (capturing) { timer = setTimeout(sendFrame, SEND_DELAY_MS); return; }
        const mode = modeSelect.value;
        const mySession = session;
        try {
            const vw = videoElement.videoWidth, vh = videoElement.videoHeight;
            const imageData = grabFrame(0.85);
            if (!imageData) return;

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData, mode: mode })
            });
            const res = await response.json();

            // Abaikan jawaban usang: kamera dihentikan / mode diganti / sedang merekam
            if (!running || session !== mySession || capturing || modeSelect.value !== mode) return;

            Tracking.drawHands(overlay, res.hands, vw, vh);
            if (mode === 'wl') {
                statusText.innerText = res.message || '';     // hasil klip terakhir tetap tampil di atas
            } else {
                render(smoother.push(res), res);
            }
        } catch (error) {
            console.error("Error prediksi:", error);
            statusText.innerText = 'Koneksi ke server bermasalah';
        } finally {
            if (running) timer = setTimeout(sendFrame, SEND_DELAY_MS);
        }
    }

    // Mode WL: hitung mundur, ambil N frame berurutan, kirim sekaligus, tampilkan 3 tebakan teratas
    async function recordWord() {
        if (!running || capturing || !isWL()) return;
        const mySession = session;
        const n = parseInt(recBtn.dataset.frames, 10) || 15;
        const interval = parseInt(recBtn.dataset.interval, 10) || 170;
        recBtn.disabled = true;
        resultTextAI.innerText = '-';
        try {
            for (let c = 3; c > 0; c--) {
                statusText.innerText = 'Siap-siap... ' + c;
                await sleep(700);
                if (!running || session !== mySession) return;
            }
            capturing = true;
            Tracking.drawHands(overlay, null, 0, 0);
            statusText.innerText = 'MEREKAM - lakukan isyarat sekarang';
            const frames = [];
            const t0 = performance.now();
            for (let i = 0; i < n; i++) {
                const f = grabFrame(0.8);
                if (f) frames.push(f);
                const wait = t0 + (i + 1) * interval - performance.now();
                if (wait > 0) await sleep(wait);
                if (!running || session !== mySession) return;
            }
            statusText.innerText = 'Menganalisis...';
            const response = await fetch('/api/predict-clip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ images: frames })
            });
            const res = await response.json();
            if (!running || session !== mySession) return;
            showClipResult(res);
        } catch (error) {
            console.error("Error rekaman:", error);
            statusText.innerText = 'Koneksi ke server bermasalah';
        } finally {
            capturing = false;
            if (running && isWL()) recBtn.disabled = false;
        }
    }

    function showClipResult(res) {
        resultTextAI.style.color = '#00b4d8';
        if (!res.top) {                       // error / tangan tidak terlihat
            resultTextAI.innerText = '-';
            statusText.innerText = res.message || STATUS_TEXT[res.status] || '';
            return;
        }
        const pct = x => Math.round(x * 100) + '%';
        const list = res.top.map(t => t.word + ' ' + pct(t.confidence)).join('  |  ');
        if (res.status === 'success') {
            resultTextAI.innerText = res.word;
            statusText.innerText = 'Tebakan teratas: ' + list + '  (frame bertangan ' + res.hand_frames + '/' + res.total_frames + ')';
        } else {
            resultTextAI.innerText = '?';
            statusText.innerText = (res.message || 'Belum yakin') + '. Mungkin: ' + list;
        }
    }

    function clearDisplay() {
        session += 1;
        smoother.reset();
        resultTextAI.innerText = '-';
        statusText.innerText = '';
        Tracking.drawHands(overlay, null, 0, 0);
    }

    function updateModeUI() {
        const wl = isWL();
        if (recBtn) { recBtn.classList.toggle('d-none', !wl); recBtn.disabled = !(wl && running) || capturing; }
        if (wlInfo) wlInfo.classList.toggle('d-none', !wl);
    }

    function startLoop() {
        if (running) return;          // cegah banyak loop jika tombol ditekan berulang
        running = true;
        clearDisplay();
        updateModeUI();
        sendFrame();
    }

    function stopLoop() {
        running = false;
        capturing = false;
        clearTimeout(timer);
        clearDisplay();
        updateModeUI();
    }

    // 'playing' = kamera benar-benar sudah menampilkan gambar (menggantikan setTimeout 1 detik)
    videoElement.addEventListener('playing', startLoop);
    document.getElementById('stop-btn').addEventListener('click', stopLoop);
    document.getElementById('clear-btn').addEventListener('click', clearDisplay);
    modeSelect.addEventListener('change', function () { clearDisplay(); updateModeUI(); });
    if (recBtn) recBtn.addEventListener('click', recordWord);
    updateModeUI();
});
