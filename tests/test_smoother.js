// Uji penghalus prediksi.  Jalankan: node tests/test_smoother.js
const assert = require('assert');
const { PredictionSmoother, coverTransform } = require('../static/js/tracking.js');
const ok = (label, w, c = 0.9) => ({ status: 'success', word: label, confidence: c });
const unsure = { status: 'uncertain' }, none = { status: 'no_hand' };
function run(name, fn) { fn(); console.log('OK   ' + name); }

run('satu frame salah tidak mengganti huruf', () => {
  const s = new PredictionSmoother();
  [ok('A'), ok('A'), ok('A')].forEach(r => s.push(r));
  assert.strictEqual(s.push(ok('B')).label, 'A');
});
run('huruf baru tampil setelah menang 3 dari 5', () => {
  const s = new PredictionSmoother();
  [ok('A'), ok('A'), ok('A')].forEach(r => s.push(r));
  let st; for (let i = 0; i < 3; i++) st = s.push(ok('B'));
  assert.strictEqual(st.label, 'B');
});
run('belum stabil -> belum tampil', () => {
  const s = new PredictionSmoother();
  assert.strictEqual(s.push(ok('A')).label, null);
  assert.strictEqual(s.push(ok('B')).label, null);
});
run('tangan hilang 3 frame -> dikosongkan (tidak ada huruf basi)', () => {
  const s = new PredictionSmoother();
  for (let i = 0; i < 3; i++) s.push(ok('A'));
  assert.strictEqual(s.push(none).label, 'A');       // 1 frame hilang masih ditahan
  s.push(none);
  assert.strictEqual(s.push(none).label, null);
});
run('tangan kembali -> riwayat lama tidak ikut', () => {
  const s = new PredictionSmoother();
  for (let i = 0; i < 3; i++) s.push(ok('A'));
  for (let i = 0; i < 3; i++) s.push(none);
  assert.strictEqual(s.push(ok('B')).label, null);
});
run('ragu terus 5 frame -> dikosongkan', () => {
  const s = new PredictionSmoother();
  for (let i = 0; i < 3; i++) s.push(ok('A'));
  let st; for (let i = 0; i < 5; i++) st = s.push(unsure);
  assert.strictEqual(st.label, null);
});
run('reset membersihkan semuanya', () => {
  const s = new PredictionSmoother();
  for (let i = 0; i < 3; i++) s.push(ok('A'));
  s.reset(); assert.strictEqual(s.push(ok('A')).label, null);
});
run('coverTransform: video 4:3 di kotak lebar memotong atas-bawah', () => {
  const t = coverTransform(640, 480, 800, 400);
  assert.ok(Math.abs(t.s - 1.25) < 1e-9 && t.ox === 0 && Math.abs(t.oy - (400 - 600) / 2) < 1e-9);
});
console.log('\nSemua tes lulus.');
