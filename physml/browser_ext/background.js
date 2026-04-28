/**
 * Mycelium browser extension — background service worker.
 *
 * Responsibilities:
 *  - Listen for page navigation events and POST to /ext/page-visit
 *  - Handle context-menu actions (bookmark selection, send to Myco)
 *  - Relay messages from content.js to the Mycelium local server
 */

const MYCO_BASE = "http://localhost:8000/ext";
const SESSION_ID = crypto.randomUUID();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function postToMyco(endpoint, payload) {
  try {
    const resp = await fetch(`${MYCO_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      console.warn(`Mycelium: POST ${endpoint} returned ${resp.status}`);
    }
    return await resp.json().catch(() => ({}));
  } catch (err) {
    // Silently fail when local server is not running
    console.debug("Mycelium: local server unreachable", err.message);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Page visit tracking
// ---------------------------------------------------------------------------

chrome.webNavigation.onCompleted?.addListener(
  async (details) => {
    if (details.frameId !== 0) return; // main frame only
    const tab = await chrome.tabs.get(details.tabId).catch(() => null);
    if (!tab) return;

    // Extract page text via scripting (MV3)
    let text = "";
    try {
      const results = await chrome.scripting.executeScript({
        target: { tabId: details.tabId },
        func: () => document.body?.innerText?.slice(0, 4000) || "",
      });
      text = results?.[0]?.result || "";
    } catch (_) {}

    await postToMyco("/page-visit", {
      url: tab.url,
      title: tab.title || "",
      text,
      timestamp: Date.now() / 1000,
      session_id: SESSION_ID,
    });
  },
  { url: [{ schemes: ["http", "https"] }] }
);

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "myco-selection",
    title: "Send selection to Mycelium",
    contexts: ["selection"],
  });
  chrome.contextMenus.create({
    id: "myco-bookmark",
    title: "Bookmark page in Mycelium",
    contexts: ["page"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "myco-selection" && info.selectionText) {
    const result = await postToMyco("/selection", {
      url: tab?.url || "",
      selected_text: info.selectionText,
      page_title: tab?.title || "",
      timestamp: Date.now() / 1000,
    });
    if (result?.ingested) {
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon48.png",
        title: "Mycelium",
        message: `Learned: "${info.selectionText.slice(0, 60)}…"`,
      });
    }
  } else if (info.menuItemId === "myco-bookmark") {
    await postToMyco("/bookmark", {
      url: tab?.url || "",
      title: tab?.title || "",
      tags: [],
    });
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icons/icon48.png",
      title: "Mycelium",
      message: "Page bookmarked and queued for learning.",
    });
  }
});

// ---------------------------------------------------------------------------
// Message relay from content.js
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "MYCO_SELECTION") {
    postToMyco("/selection", msg.payload).then((r) => sendResponse(r));
    return true; // keep channel open
  }
});
