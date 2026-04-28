/**
 * Mycelium browser extension — content script.
 *
 * Injected into every page. Watches for text selection events and
 * optionally sends highlighted text to the background worker for ingestion.
 *
 * Passive by default — only sends data when the user explicitly triggers
 * via context menu or keyboard shortcut (Ctrl+Shift+M / Cmd+Shift+M).
 */

(function () {
  "use strict";

  let lastSelection = "";

  // Track selection
  document.addEventListener("mouseup", () => {
    const sel = window.getSelection()?.toString().trim() || "";
    if (sel && sel !== lastSelection && sel.length > 10) {
      lastSelection = sel;
    }
  });

  // Keyboard shortcut: Ctrl+Shift+M / Cmd+Shift+M → send selection to Myco
  document.addEventListener("keydown", (e) => {
    const isMac = navigator.platform.includes("Mac");
    const trigger = isMac
      ? e.metaKey && e.shiftKey && e.key === "m"
      : e.ctrlKey && e.shiftKey && e.key === "M";

    if (trigger && lastSelection) {
      chrome.runtime.sendMessage({
        type: "MYCO_SELECTION",
        payload: {
          url: window.location.href,
          selected_text: lastSelection,
          page_title: document.title,
          timestamp: Date.now() / 1000,
        },
      });
      // Brief visual feedback
      _flashIndicator();
    }
  });

  function _flashIndicator() {
    const div = document.createElement("div");
    div.textContent = "🍄 Mycelium learned this";
    Object.assign(div.style, {
      position: "fixed",
      bottom: "20px",
      right: "20px",
      padding: "8px 16px",
      background: "#1a1a2e",
      color: "#e0e0ff",
      borderRadius: "8px",
      fontSize: "13px",
      zIndex: 2147483647,
      boxShadow: "0 2px 12px rgba(0,0,0,0.4)",
      transition: "opacity 0.3s",
    });
    document.body.appendChild(div);
    setTimeout(() => {
      div.style.opacity = "0";
      setTimeout(() => div.remove(), 400);
    }, 1800);
  }
})();
