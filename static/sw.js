/* Minimal service worker for installability.
 * We keep this intentionally small and non-invasive.
 */

const CACHE = 'mycelium-shell-v3';
const PRECACHE = [
  '/static/manifest.webmanifest',
  '/static/icon.svg',
  '/device',
  '/projects',
  '/knowledge',
  '/hive/health'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  // Network-first for HTML, cache-first for static.
  const url = new URL(req.url);
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => {});
        return resp;
      }))
    );
    return;
  }

  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => {});
          return resp;
        })
        .catch(() => caches.match(req).then((hit) => hit || caches.match('/device')))
    );
    return;
  }

  event.respondWith(fetch(req).catch(() => caches.match(req)));
});
