# Deployment Guide — Road Pothole Detection & Severity Analysis

## ⚠️ Why the whole project can't run *on* Netlify itself

Netlify hosts **static files** (HTML/CSS/JS) and Node/Go serverless functions.
It has **no Python runtime** and cannot run long-lived processes — so the
Flask + OpenCV + PyTorch backend (`api/app.py`, `src/*`) cannot execute there.

What Netlify **can** host: the dashboard UI (the `netlify/` folder).
The backend must live on a Python-capable host (Render, Railway,
PythonAnywhere, Hugging Face Spaces, any VPS/Docker).

Two working strategies below. Both are fully configured already.

---

## Option A (recommended) — Whole project on Render in one service

Deploy backend **and** dashboard together; open `https://<your-app>.onrender.com/dashboard`.

1. Push the project to GitHub:
   ```bash
   cd E:\Project\CODE\ProjectDev
   git init
   git add -A
   git commit -m "Pothole detection project"
   git remote add origin https://github.com/<you>/pothole-detection.git
   git push -u origin main
   ```
2. Go to https://render.com → **New → Blueprint** → pick your repo.
   Render reads `render.yaml` automatically (install → run tests → serve).
3. Deploy. Build installs `requirements-render.txt` (no torch needed — the
   classical detector runs and detection works fully), runs the test suite,
   then starts `waitress-serve ... api.app:app`.
4. Open `https://pothole-detection-api.onrender.com/dashboard`.

Live verification: `https://<app>.onrender.com/api/health` → `{"status":"healthy"}`

**Enable YOLOv8 on Render (optional, needs paid/512MB+ instance):**
add env `WITH_TORCH` → change buildCommand to
`pip install torch ultralytics --index-url https://download.pytorch.org/whl/cpu -r requirements-render.txt`.

---

## Option B — Dashboard on Netlify + API on Render (true "deploy on Netlify")

1. **Deploy the API** as in Option A (Render) → note its URL, e.g.
   `https://pothole-detection-api.onrender.com`
2. **Point the dashboard at that API** — edit `netlify/index.html` (and/or
   `frontend/templates/index.html`):
   ```html
   window.POTHOLE_API_BASE = "https://pothole-detection-api.onrender.com";
   ```
   (`app.js` prefixes every `fetch` with this value; CORS is already
   enabled on the API for cross-origin requests.)
3. **Deploy to Netlify** — either:
   - **Drag & drop:** open https://app.netlify.com/drop → drag the
     `netlify/` folder → done (instant URL), **or**
   - **Git-based:** app.netlify.com → *Add new site → Import from Git* →
     select the repo. `netlify.toml` publishes the `netlify/` folder and
     re-syncs it from `frontend/` on every build.
4. Result: UI at `https://<name>.netlify.app`, API at your Render URL.

---

## Option C — Docker anywhere (VPS, Fly.io, AWS, Azure…)

```bash
cd E:\Project\CODE\ProjectDev
docker build -t pothole-detection .          # classical backend (~350MB)
docker run -p 8000:5000 -e PORT=5000 pothole-detection
# YOLOv8 backend: docker build --build-arg WITH_TORCH=1 -t pothole-detection .
```

## Alternative hosts for the full backend

| Host | Type | Free? | Notes |
|------|------|-------|-------|
| **Render** | Python web service | ✅ free tier | Easiest, `render.yaml` ready |
| Railway | Python service | trial credit | `pip install -r requirements-render.txt` |
| PythonAnywhere | Flask/WSGI | ✅ free | Manual file upload, Flask config |
| Hugging Face Spaces | Gradio/Streamlit | ✅ free | Needs a small Gradio wrapper |
| Vercel | Node/Python serverless | ✅ | Per-request only; OpenCV/Torch too heavy — not recommended |
| Fly.io / VPS + Docker | container | — | Use `Dockerfile` above |

## Local production smoke test (what was verified in this project)

```bash
cd E:\Project\CODE\ProjectDev
pip install -r requirements-render.txt
pytest tests/ -q                      # 24 passed, 1 skipped
PORT=8080 waitress-serve --host=0.0.0.0 --port=8080 api.app:app
curl http://127.0.0.1:8080/api/health # {"status": "healthy", ...}
```

**Config that makes it production-safe** (in `config.py`):
- `PORT` is read from the environment (Render/Railway inject it)
- `API_DEBUG` defaults to **false** (Werkzeug debugger never exposed)
- `FLASK_SECRET_KEY` comes from env (auto-generated on Render)
- `waitress` replaces the dev server (threaded, production-grade)