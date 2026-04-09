/* Myco service worker — enables PWA installability and offline support. */

const CACHE = 'myco-shell-v4';
const PRECACHE = [
  '/static/manifest.webmanifest',
  '/static/icon.svg',
  '/live',
  '/login',
  '/cinema',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  // Clean old caches
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // Cache-first for static assets (icons, CSS, JS)
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

  // Network-first for pages, fallback to cache, then /live
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => {});
          return resp;
        })
        .catch(() => caches.match(req).then((hit) => hit || caches.match('/live')))
    );
    return;
  }

  // API calls: network only (no caching)
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(fetch(req));
    return;
  }

  event.respondWith(fetch(req).catch(() => caches.match(req)));
});
