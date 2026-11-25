// Kamus: (1) video dimuat bertahap saat kartu terlihat, (2) judul bagian ikut disembunyikan saat pencarian.
// Pencarian per kartu tetap dikerjakan script.js (tidak diubah); file ini hanya menyesuaikan tampilan bagian.
(function (root) {
  // Jumlah kartu yang masih tampil per bagian. sections: [[hidden, hidden, ...], ...]
  function visibleCounts(sections) {
    return sections.map(items => items.filter(hidden => !hidden).length);
  }

  function init() {
    const sections = Array.from(document.querySelectorAll('.dict-section'));
    const pills = Array.from(document.querySelectorAll('.dict-pill'));

    // --- bagian tanpa hasil disembunyikan beserta pintasannya ---
    function syncSections() {
      const counts = visibleCounts(sections.map(s =>
        Array.from(s.querySelectorAll('.dictionary-item')).map(i => i.classList.contains('d-none'))));
      sections.forEach((s, k) => {
        s.classList.toggle('d-none', counts[k] === 0);
        const pill = pills.find(p => p.dataset.target === s.id);
        if (pill) pill.classList.toggle('d-none', counts[k] === 0);
      });
    }
    const input = document.getElementById('filter-input');
    // 'keyup' = peristiwa yang sama dengan pencarian di script.js, dan didaftarkan setelahnya, jadi hasilnya sudah terbarui
    if (input) input.addEventListener('keyup', syncSections);

    // --- video dimuat bertahap ---
    const videos = Array.from(document.querySelectorAll('video[data-src]'));
    const load = v => { if (v.dataset.src) { v.src = v.dataset.src; v.removeAttribute('data-src'); } };
    if (!('IntersectionObserver' in window)) { videos.forEach(load); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { load(e.target); io.unobserve(e.target); } });
    }, { rootMargin: '300px' });
    videos.forEach(v => io.observe(v));
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { visibleCounts };
  if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', init);
})(typeof window !== 'undefined' ? window : globalThis);
