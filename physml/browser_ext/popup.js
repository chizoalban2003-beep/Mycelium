/**
 * Mycelium popup script.
 */

const BASE = "http://localhost:8000";
const EXT_BASE = `${BASE}/ext`;

const statusEl = document.getElementById("status");
const responseEl = document.getElementById("response");
const counterEl = document.getElementById("counter");
const chatInput = document.getElementById("chat-input");

async function get(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

async function post(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

function showResponse(text) {
  responseEl.style.display = "block";
  responseEl.textContent = text;
}

// Check connection on open
(async () => {
  const status = await get(`${EXT_BASE}/status`);
  if (status?.ok) {
    statusEl.textContent = `Connected — ${status.ingester?.ingested ?? 0} items learned`;
    statusEl.className = "status ok";
    counterEl.textContent = `Facts in memory: ${status.ingester?.facts_extracted ?? "–"}`;
  } else {
    statusEl.textContent = "Mycelium server not running (physml server)";
    statusEl.className = "status err";
  }
})();

// Send chat
document.getElementById("btn-send").addEventListener("click", async () => {
  const text = chatInput.value.trim();
  if (!text) return;
  showResponse("Thinking…");
  const result = await post(`${BASE}/chat`, { message: text });
  showResponse(result?.response || result?.reply || JSON.stringify(result) || "No response.");
  chatInput.value = "";
});
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("btn-send").click();
});

// Learn current page
document.getElementById("btn-page").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => document.body?.innerText?.slice(0, 4000) || "",
  });
  const text = results?.[0]?.result || "";
  const r = await post(`${EXT_BASE}/page-visit`, {
    url: tab.url, title: tab.title, text,
    timestamp: Date.now() / 1000, session_id: "popup",
  });
  showResponse(r?.ingested ? `Learned: ${tab.title}` : "Could not ingest — server may be busy.");
});

// Learn selected text
document.getElementById("btn-selection").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection()?.toString().trim() || "",
  });
  const sel = results?.[0]?.result || "";
  if (!sel) { showResponse("No text selected on the page."); return; }
  const r = await post(`${EXT_BASE}/selection`, {
    url: tab.url, selected_text: sel, page_title: tab.title,
    timestamp: Date.now() / 1000,
  });
  showResponse(r?.ingested ? `Learned selection (${sel.length} chars)` : "Server unreachable.");
});

// Bookmark
document.getElementById("btn-bookmark").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const r = await post(`${EXT_BASE}/bookmark`, {
    url: tab.url, title: tab.title, tags: [],
  });
  showResponse(r?.ingested ? "Bookmarked!" : "Server unreachable.");
});
