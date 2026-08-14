// Cache-first service worker so the app (including the ~15MB MediaPipe
// WASM runtime + pose model) works fully offline after the first
// successful load. Bump CACHE_NAME to force a refresh on the next visit.
// v3: file-source lifecycle fix (no autoplay-on-select, no loop, no countdown
// for pre-recorded clips). MUST be bumped whenever a cached file changes --
// Chrome byte-diffs THIS file to decide whether to update, so an unchanged
// service worker means the phone keeps serving the old app.js indefinitely.
const CACHE_NAME = 'exergaming-mobile-v3';

const PRECACHE_URLS = [
  './',
  'index.html',
  'manifest.json',
  'css/style.css',
  'js/app.js',
  'js/exercise-definition.js',
  'js/generic-exercise.js',
  'js/pose-analyzer.js',
  'js/position-gate.js',
  'definitions/learned_squat.json',
  'definitions/learned_pushup.json',
  'definitions/learned_bicepcurl.json',
  'vendor/vision_bundle.mjs',
  'vendor/wasm/vision_wasm_internal.js',
  'vendor/wasm/vision_wasm_internal.wasm',
  'vendor/wasm/vision_wasm_nosimd_internal.js',
  'vendor/wasm/vision_wasm_nosimd_internal.wasm',
  'vendor/pose_landmarker_lite.task',
  'vendor/pose_landmarker_full.task',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-192-maskable.png',
  'icons/icon-512-maskable.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      await Promise.all(
        PRECACHE_URLS.map(async (url) => {
          try {
            const res = await fetch(url);
            if (res.ok) await cache.put(url, res);
            else console.warn('[sw] precache non-OK response, skipped:', url, res.status);
          } catch (err) {
            console.warn('[sw] precache failed, skipped:', url, err);
          }
        })
      );
      self.skipWaiting();
    })()
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)));
      self.clients.claim();
    })()
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  event.respondWith(
    (async () => {
      const cached = await caches.match(event.request);
      if (cached) return cached;
      try {
        const res = await fetch(event.request);
        if (res.ok && event.request.url.startsWith(self.location.origin)) {
          const cache = await caches.open(CACHE_NAME);
          cache.put(event.request, res.clone());
        }
        return res;
      } catch (err) {
        return cached || Response.error();
      }
    })()
  );
});
