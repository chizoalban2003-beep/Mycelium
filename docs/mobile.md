# Mobile & Browser Extension

Mycelium runs locally on your computer and exposes a REST API that both the
**mobile PWA** and **browser extension** connect to.

---

## Mobile PWA

The Progressive Web App works in any mobile browser and can be installed as a
home screen app (no app store required).

### Setup

1. Start the Mycelium server on your computer:

```bash
pip install "physml[server]"
uvicorn physml.server:app --host 0.0.0.0 --port 8000
```

2. Make sure your phone is on the same Wi-Fi network as your computer.

3. Find your computer's local IP address:

```bash
# Linux / macOS
ip route get 1 | awk '{print $7}' || ifconfig | grep "inet "
# Windows
ipconfig
```

4. On your phone, open `http://<your-ip>:8000/pwa/`

5. Tap **Add to Home Screen** (iOS: Share button → Add to Home Screen; Android: ⋮ menu → Add to Home Screen)

### Features

| Feature | Description |
|---|---|
| **Chat** | Conversational interface with your local Myco |
| **Ingest** | Send text/URLs directly from your phone to Myco's knowledge base |
| **Context** | See what Myco knows about your current work context |
| **Patterns** | View your behavioral patterns (top apps, peak hours, topics) |
| **Push intent** | Tell Myco your current task so it can be context-aware |
| **Offline indicator** | Shows when server is unreachable |

### Mobile API Endpoints

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/mobile/chat` | POST | `{"message": "..."}` | Chat with Myco |
| `/mobile/ingest` | POST | `{"source": "...", "topic": "..."}` | Ingest content |
| `/mobile/context` | GET | — | Current user context |
| `/mobile/patterns` | GET | — | Behavioral patterns |
| `/mobile/push-intent` | POST | `{"intent": "..."}` | Set current task |
| `/mobile/status` | GET | — | Server + subsystem status |

---

## Browser Extension

The Chrome/Firefox extension automatically learns from your browsing and lets you
send selections and bookmarks to Myco.

### Installation

**Chrome:**
1. Open `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `physml/browser_ext/` folder from the Mycelium project

**Firefox:**
1. Open `about:debugging#/runtime/this-firefox`
2. Click **Load Temporary Add-on**
3. Select `physml/browser_ext/manifest.json`

### Usage

| Action | How |
|---|---|
| **Auto page learning** | Just browse — every page visit is sent to Myco |
| **Send selection** | Highlight text → right-click → **Send selection to Myco** |
| **Bookmark** | Click extension icon → **Bookmark this page** |
| **Chat** | Click extension icon → type in the popup chat box |
| **Quick shortcut** | Press `Ctrl+Shift+M` on any page to send selected text |

### Extension API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ext/page-visit` | POST | Track a page visit (auto-called on navigation) |
| `/ext/selection` | POST | Ingest highlighted text |
| `/ext/bookmark` | POST | Bookmark current page |
| `/ext/command` | POST | Send a command to Myco |
| `/ext/status` | GET | Extension health check |

### CORS Note

The server must be running on `http://localhost:8000` for the extension to work.
The extension uses `host_permissions` in `manifest.json` to allow cross-origin
requests to localhost.  The FastAPI server already has CORS enabled.

---

## Security Considerations

- The server binds to `0.0.0.0` when you use `--host 0.0.0.0` — anyone on your
  local network can access it.  For personal use on a trusted home network this
  is fine; for shared networks, bind to `127.0.0.1` instead.
- No authentication is required by default.  If you need auth, add an API key
  check via FastAPI middleware.
- All data stays on your device — nothing is sent to external servers except
  optional LLM calls to the Anthropic API.
