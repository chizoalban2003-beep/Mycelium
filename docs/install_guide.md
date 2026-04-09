# Installing Myco on Your Device

## Quick Start (5 minutes)

### Prerequisites
- Python 3.11+ installed
- Git installed
- A device you use daily (laptop recommended for first run)

### Step 1: Install

```bash
git clone https://github.com/chizoalban2003-beep/Mycelium.git ~/myco
cd ~/myco
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
```

### Step 2: Configure

```bash
cp .env.example .env
```

Edit `.env` if you want to change anything (defaults work fine for first run).

### Step 3: Start

```bash
uvicorn mycelium_app.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Open

Go to **http://localhost:8000** in your browser.

1. Click "Create your companion"
2. Enter your name, email, password
3. Name your companion (or leave as "Myco")
4. Select your gender (companion will mirror you)
5. Click "Begin growing"
6. Watch the cinematic intro
7. You're in the live ecosystem

### Step 5: Let it run

Leave the server running. The background daemons will:
- Collect OS signals every 15 seconds
- Run a learning cycle every 2 minutes
- Build the force field and evolve the ecosystem

**The longer it runs, the smarter it gets.**

---

## Adding AI Chat (Optional but Recommended)

For real conversations with your companion, install Ollama:

```bash
# Install Ollama (one command)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the llama3 model (~4GB)
ollama pull llama3
```

Then add to your `.env`:

```
NARRATIVE_LLM_ENDPOINT=http://localhost:11434/api/generate
NARRATIVE_LLM_MODEL=llama3
```

Restart Myco. Now your companion can have real conversations.

---

## Running as a Background Service (Linux)

So Myco starts automatically when you boot:

```bash
bash scripts/install_child_agent_service.sh
```

This creates a systemd user service that starts Myco on login.

---

## Installing on Android

The PWA is installable:
1. Open **http://localhost:8000** in Chrome on your phone
2. Chrome will show "Add to Home Screen"
3. Tap it — Myco appears as a native app

For the TWA (native APK), see `docs/android_twa_packaging.md`.

---

## What Happens Over Time

| Time | What Your Companion Learns |
|------|---------------------------|
| **First hour** | Which apps you use, CPU/memory patterns, process landscape |
| **First day** | Circadian rhythm (peak hours), app session durations, context switching rate |
| **First week** | Routines (app transition sequences), focus patterns, attention distribution |
| **First month** | Long-term trends, behavioral anomalies, seasonal patterns |

The growth stages advance automatically:
- 🌱 **Infant**: Just observing (first hours)
- 🌿 **Toddler**: Starting to predict (after ~5 coherent signal groups form)
- 🌳 **Adolescent**: Proactive suggestions (after ~10 bound signals with 50%+ coherence)
- 🌲 **Adult**: Full autonomous companion (after ~15 bound signals with 70%+ coherence)

---

## Troubleshooting

**"uvicorn not found"**: Run `export PATH="$HOME/.local/bin:$PATH"` or use `.venv/bin/uvicorn`

**3D scene is dark**: Your browser needs WebGL. Try Chrome. Myco automatically falls back to 2D if WebGL is unavailable.

**No signals collected**: The collector needs `psutil`. Run `pip install psutil` in your venv.

**Window focus tracking not working**: Install `xdotool` on Linux: `sudo apt install xdotool`
