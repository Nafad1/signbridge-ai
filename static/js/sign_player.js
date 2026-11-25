// Menggambar pose tangan (kerangka 21 titik) dan memutar urutan isyarat.
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]
];
const HAND_COLORS = ['#00b4d8', '#ffb703'];
let POSES = null;

async function loadPoses() {
  if (!POSES) {
    try { POSES = await (await fetch('/static/data/poses.json')).json(); }
    catch (e) { POSES = { alphabet: {}, words: {} }; }
  }
  return POSES;
}

function drawPose(canvas, hands) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!hands) return;
  const pts = hands.flat();
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = Math.min(W, H) * 0.12;
  const scale = Math.min((W - 2 * pad) / Math.max(maxX - minX, 0.01), (H - 2 * pad) / Math.max(maxY - minY, 0.01));
  const ox = (W - (maxX - minX) * scale) / 2 - minX * scale;
  const oy = (H - (maxY - minY) * scale) / 2 - minY * scale;
  const T = p => [p[0] * scale + ox, p[1] * scale + oy];
  const lw = Math.max(2, W / 90);
  hands.forEach((hand, hi) => {
    ctx.strokeStyle = HAND_COLORS[hi % 2]; ctx.fillStyle = HAND_COLORS[hi % 2]; ctx.lineWidth = lw; ctx.lineCap = 'round';
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const [x1, y1] = T(hand[a]), [x2, y2] = T(hand[b]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    });
    hand.forEach(p => { const [x, y] = T(p); ctx.beginPath(); ctx.arc(x, y, lw * 1.2, 0, 7); ctx.fill(); });
  });
}

// Pratinjau kecil di halaman kamus
document.addEventListener('DOMContentLoaded', async () => {
  const previews = document.querySelectorAll('canvas[data-pose-key]');
  if (!previews.length) return;
  const poses = await loadPoses();
  previews.forEach(c => drawPose(c, (poses[c.dataset.poseGroup] || {})[c.dataset.poseKey]));
});

// Pemutar urutan isyarat (halaman Teks ke Isyarat)
document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('search-input');
  const stage = document.getElementById('stage-canvas');
  if (!input || !stage) return;

  const video = document.getElementById('stage-video');
  const caption = document.getElementById('stage-caption');
  const chips = document.getElementById('chips');
  const notice = document.getElementById('notice');
  const resultArea = document.getElementById('result-area');
  const errorArea = document.getElementById('error-area');
  const errorMsg = document.getElementById('error-message');
  const playBtn = document.getElementById('play-btn');
  const speedSel = document.getElementById('speed-select');
  const HOLD_MS = 1100;

  let items = [], idx = 0, playing = false, timer = null;

  function speed() { return parseFloat(speedSel.value); }
  function clearTimer() { clearTimeout(timer); timer = null; }

  function show(i) {
    clearTimer();
    idx = i;
    const item = items[i];
    [...chips.children].forEach((c, k) => c.classList.toggle('active', k === i));
    caption.textContent = item.label;
    video.onended = null; video.onerror = null;

    const advance = () => { if (playing) next(); };
    const showPose = () => {
      video.pause(); video.classList.add('d-none'); stage.classList.remove('d-none');
      if (item.pose_key) {
        const group = item.type === 'word' ? 'words' : 'alphabet';
        drawPose(stage, POSES[group][item.pose_key]);
      } else {
        drawPose(stage, null);
      }
      if (playing) timer = setTimeout(advance, HOLD_MS / speed());
    };

    if (item.video) {
      stage.classList.add('d-none'); video.classList.remove('d-none');
      video.src = item.video; video.playbackRate = speed();
      video.onended = advance;
      video.onerror = showPose;          // file hilang -> pakai pose
      if (playing) video.play().catch(showPose); else video.pause();
    } else {
      showPose();
    }
  }

  function next() {
    if (idx < items.length - 1) show(idx + 1);
    else { playing = false; setPlayIcon(); }
  }
  function setPlayIcon() {
    playBtn.innerHTML = playing
      ? '<i class="fa-solid fa-pause me-2"></i>Jeda'
      : '<i class="fa-solid fa-play me-2"></i>' + (idx >= items.length - 1 && items.length ? 'Ulangi' : 'Putar');
  }
  function togglePlay() {
    if (!items.length) return;
    if (!playing) {
      playing = true;
      show(idx >= items.length - 1 && !timer ? 0 : idx);
    } else { playing = false; clearTimer(); video.pause(); }
    setPlayIcon();
  }

  async function translate() {
    const text = input.value.trim();
    if (!text) return;
    resultArea.classList.add('d-none'); errorArea.classList.add('d-none');
    try {
      await loadPoses();
      const data = await (await fetch('/api/text-to-sign?text=' + encodeURIComponent(text))).json();
      if (data.status !== 'success') { errorMsg.textContent = data.message; errorArea.classList.remove('d-none'); return; }
      items = data.items;
      chips.innerHTML = '';
      items.forEach((it, k) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn-sm sign-chip ' + (it.type === 'word' ? 'chip-word' : 'chip-letter');
        b.textContent = it.label;
        b.onclick = () => { playing = false; setPlayIcon(); show(k); };
        chips.appendChild(b);
      });
      let msg = [];
      if (data.any_pose_only) msg.push('Sebagian isyarat ditampilkan sebagai animasi kerangka tangan dari data landmark (perkiraan, bukan video asli). Tambahkan video ke <code>static/videos/</code> untuk menggantikannya.');
      if (data.skeleton_labels && data.skeleton_labels.length) msg.push('Animasi kerangka tangan (bukan rekaman penanda) untuk: ' + data.skeleton_labels.join(', ') + '.');
      if (data.skipped.length) msg.push('Karakter dilewati (belum didukung): ' + data.skipped.join(', ') + '.');
      notice.innerHTML = msg.join('<br>');
      notice.classList.toggle('d-none', !msg.length);
      resultArea.classList.remove('d-none');
      playing = true; setPlayIcon(); show(0); setPlayIcon();
    } catch (e) {
      console.error(e);
      errorMsg.textContent = 'Terjadi kesalahan pada server.'; errorArea.classList.remove('d-none');
    }
  }

  document.getElementById('search-btn').addEventListener('click', translate);
  input.addEventListener('keypress', e => { if (e.key === 'Enter') translate(); });
  playBtn.addEventListener('click', togglePlay);
  speedSel.addEventListener('change', () => { video.playbackRate = speed(); });
  document.querySelectorAll('[data-example]').forEach(b => b.addEventListener('click', () => { input.value = b.dataset.example; translate(); }));
});
