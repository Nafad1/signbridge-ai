// Pelacakan sisi browser: penghalus prediksi + gambar landmark di atas video.
// Dipisah supaya bisa diuji tanpa browser:  node tests/test_smoother.js
(function (root) {
  // Penghalus: satu frame yang salah tidak boleh mengganti huruf yang tampil.
  //  - huruf baru tampil hanya jika menang >= minVotes dari `window` frame terakhir
  //  - tangan hilang `clearAfterMiss` frame berturut-turut -> hasil dikosongkan
  //  - ragu terus selama `window` frame -> hasil dikosongkan (tidak menahan huruf basi)
  class PredictionSmoother {
    constructor(opts = {}) {
      this.window = opts.window || 5;
      this.minVotes = opts.minVotes || 3;
      this.clearAfterMiss = opts.clearAfterMiss || 3;
      this.reset();
    }
    reset() { this.history = []; this.misses = 0; this.current = null; this.confidence = 0; }
    push(res) {
      const status = res && res.status;
      if (status === 'no_hand' || status === 'hand_cut') {
        this.misses += 1;
        if (this.misses >= this.clearAfterMiss) { this.history = []; this.current = null; this.confidence = 0; }
        return this._state(status);
      }
      if (status !== 'success' && status !== 'uncertain') return this._state(status || 'error');
      this.misses = 0;
      this.history.push(status === 'success' ? { label: res.word, conf: res.confidence } : { label: null, conf: 0 });
      if (this.history.length > this.window) this.history.shift();

      const votes = {};
      this.history.forEach(h => { if (h.label) { (votes[h.label] = votes[h.label] || []).push(h.conf); } });
      let best = null;
      Object.keys(votes).forEach(l => { if (!best || votes[l].length > votes[best].length) best = l; });
      if (best && votes[best].length >= this.minVotes) {
        this.current = best;
        this.confidence = votes[best].reduce((a, b) => a + b, 0) / votes[best].length;
      } else if (this.history.length >= this.window && !best) {
        this.current = null; this.confidence = 0;
      }
      return this._state(status);
    }
    _state(status) { return { label: this.current, confidence: this.confidence, status }; }
  }

  // Video ditampilkan dengan object-fit: cover; titik landmark harus dipetakan dengan aturan yang sama.
  function coverTransform(vw, vh, cw, ch) {
    const s = Math.max(cw / vw, ch / vh);
    return { s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2 };
  }

  const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const COLORS = ['#00b4d8', '#ffb703'];

  // hands: [{kept, points:[[x,y]*21]}] dengan x,y 0-1 pada gambar yang SUDAH dicerminkan server.
  // Video di layar juga dicerminkan (CSS), jadi tidak perlu membalik lagi di sini.
  function drawHands(canvas, hands, vw, vh) {
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);
    if (!hands || !vw || !vh) return;
    const t = coverTransform(vw, vh, cw, ch);
    let keptIdx = 0;
    hands.forEach(h => {
      const color = h.kept ? COLORS[keptIdx++ % 2] : '#9aa0a6';   // abu-abu = tangan yang DIABAIKAN
      const P = h.points.map(p => [p[0] * vw * t.s + t.ox, p[1] * vh * t.s + t.oy]);
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = h.kept ? 3 : 2;
      ctx.setLineDash(h.kept ? [] : [5, 5]);
      CONNECTIONS.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(P[a][0], P[a][1]); ctx.lineTo(P[b][0], P[b][1]); ctx.stroke(); });
      ctx.setLineDash([]);
      P.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], h.kept ? 4 : 3, 0, 7); ctx.fill(); });
    });
  }

  const api = { PredictionSmoother, coverTransform, drawHands };
  root.Tracking = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
