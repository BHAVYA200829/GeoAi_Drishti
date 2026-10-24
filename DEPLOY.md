# Deploying Drishti

The challenge requires a **live HTTP endpoint** that stays available through the
whole judging window. Drishti runs comfortably on a free tier in its default
mode (no model, no keys) because all reasoning is deterministic and only the
optional SLM needs real RAM.

Test the endpoint before you rely on it:

```bash
curl -s https://<your-url>/health
curl -s -X POST https://<your-url>/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Where can I go running tomorrow evening near Koregaon Park?"}'
```

---

## Option A - Render (recommended, free)

1. Push this repo to GitHub.
2. Render dashboard → **New → Web Service** → connect the repo.
3. Render reads `render.yaml` automatically. If configuring manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/health`
4. Add any secret keys (`WAQI_TOKEN`, `IMD_API_KEY`, `IMD_JWT`) as environment
   variables in the dashboard. Never commit them.
5. Deploy. Your endpoint is `https://<service>.onrender.com`.

> Free instances sleep after inactivity and cold-start in ~30s. Hit `/health`
> once before judging so the instance is warm.

## Option B - Railway

1. New project → Deploy from GitHub repo.
2. Railway auto-detects the `Procfile`.
3. Set environment variables as needed. Endpoint is generated on deploy.

## Option C - Hugging Face Spaces (Docker)

Useful if you want the model hosted next to the API.

1. Create a new Space → **Docker** SDK.
2. Copy this repo in. The included `Dockerfile` builds the service.
3. In the Space's `README.md` front-matter set:
   ```
   ---
   title: Drishti
   sdk: docker
   app_port: 8000
   ---
   ```
4. Add secrets under **Settings → Variables and secrets** (never in code).

## Option D - Docker anywhere

```bash
docker build -t drishti .
docker run -p 8000:8000 --env-file .env drishti
```

---

## Enabling the SLM on a host

Only turn the model on if the host has the RAM (a 1.5B model in float32 needs
roughly 4-6 GB). Prefer the GGUF backend on CPU-only hosts:

```bash
export DRISHTI_ENABLE_SLM=1
export DRISHTI_SLM_BACKEND=llamacpp
export DRISHTI_GGUF_PATH=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

If the host cannot hold the model, leave `DRISHTI_ENABLE_SLM=0`. The assistant
still works fully - it simply uses the deterministic parser and narrator.

## Judging-window checklist

- [ ] `/health` returns `{"status":"ok"}` from a public URL
- [ ] `/ask` returns a grounded answer for a sample question
- [ ] The instance is warm and will not cold-sleep mid-demo
- [ ] README lists the base model, all data sources, and the endpoint URL
- [ ] The HF model repo is public and loads with `from_pretrained(...)`
