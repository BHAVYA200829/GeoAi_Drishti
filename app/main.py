"""Drishti API -- FastAPI entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Header, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from . import data_fetcher as df
from . import reasoning
from .config import ENABLE_SLM, MODEL_ID
from .schemas import AskRequest, AskResponse

app = FastAPI(
    title="Drishti",
    description="A grounded AI assistant that tells you where and when to head outdoors.",
    version=__version__,
)

_STATIC = os.path.join(os.path.dirname(__file__), "static")

# static assets (logo, favicon) live next to index.html
if os.path.isdir(_STATIC):
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")


def _page(name: str, fallback: str) -> HTMLResponse:
    path = os.path.join(_STATIC, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse(fallback)


@app.get("/", response_class=HTMLResponse)
def index():
    """The entry screen: logo, the namaste gesture and the greeting."""
    return _page("welcome.html",
                 "<h1>Drishti</h1><p>POST to /ask with {\"question\": \"...\"}</p>")


@app.get("/signin", response_class=HTMLResponse)
def signin():
    """The lamp / sign-in template, kept as uploaded."""
    return _page("index.html",
                 "<h1>Drishti</h1><p>POST to /ask with {\"question\": \"...\"}</p>")


@app.get("/app", response_class=HTMLResponse)
def chat_app():
    """The chat app (left sidebar history + voice), behind the entry page."""
    return _page("app.html",
                 "<h1>Drishti</h1><p>POST to /ask with {\"question\": \"...\"}</p>")


@app.get("/health")
def health():
    """Liveness + which routes this process actually loaded.

    If /app is missing from `routes`, the server is running old code -- restart it.
    """
    return {
        "status": "ok",
        "assistant": "Drishti",
        "version": __version__,
        "slm_enabled": ENABLE_SLM,
        "model": MODEL_ID if ENABLE_SLM else "deterministic-fallback",
        "slm_role": ("parses, decides and narrates; every decision is validated "
                     "against the fetched facts" if ENABLE_SLM
                     else "off: the deterministic engine parses, decides and narrates"),
        "routes": [r.path for r in app.routes
                   if hasattr(r, "path") and r.path.startswith("/")
                   and r.path not in ("/openapi.json",)],
    }


@app.get("/sources")
def sources():
    return {"sources": df.SOURCES}


class WarmRequest(BaseModel):
    lat: float
    lon: float


@app.post("/warm")
def warm(req: WarmRequest):
    """Fetch weather + air quality for a coordinate into the cache.

    The chat page calls this the moment a GPS fix lands, so the question that
    follows finds the data already cached and answers much faster.
    """
    df.weather(req.lat, req.lon)
    df.air_quality(req.lat, req.lon)
    return {"status": "warm", "lat": req.lat, "lon": req.lon}


@app.get("/languages")
def languages():
    """The chat languages the UI can switch between."""
    from .i18n import LANGS
    names = {"en": "English", "hi": "हिन्दी", "mr": "मराठी", "ta": "தமிழ்", "bn": "বাংলা"}
    return {"languages": [{"code": c, "name": names.get(c, c)} for c in LANGS]}


@app.get("/i18n")
def i18n_strings(lang: str = Query("en")):
    """All UI strings for a language, so the page can localise itself."""
    from .i18n import LANGS, DEFAULT
    code = (lang or DEFAULT)[:2]
    return {"lang": code if code in LANGS else DEFAULT, "strings": LANGS.get(code, LANGS[DEFAULT])}


@app.get("/activities")
def activities():
    """The pickable activities (tasks) shown in the UI."""
    from .profiles import FEATURED_ACTIVITIES
    return {"activities": [{"key": k, "label": lab} for k, lab in FEATURED_ACTIVITIES]}


@app.get("/cities")
def cities():
    """Current weather + AQI for the major cities (for the landing dashboard)."""
    return {"cities": df.city_conditions()}


@app.get("/conditions")
def conditions(lat: float = Query(...), lon: float = Query(...),
               label: str | None = Query(None)):
    """Current conditions for any location."""
    return df.conditions(lat, lon, label=label)


@app.get("/diag")
def diag(lat: float = Query(28.6139), lon: float = Query(77.2090)):
    """
    Reachability of every upstream, tested from this deployment. Open
    /diag on the live URL when a source looks dead. Defaults to Delhi.
    """
    return df.diagnose(lat, lon)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = reasoning.answer(req.question, lat=req.lat, lon=req.lon,
                              session_id=req.session_id, lang=req.lang or "en")
    return JSONResponse(result)


# ---------------- accounts: email + password (no OTP) ----------------
from . import auth as _auth


class AuthRequest(BaseModel):
    email: str
    password: str = ""
    name: Optional[str] = None


@app.post("/auth/signup")
def auth_signup(req: AuthRequest):
    ok, msg, user, token = _auth.signup(req.email, req.password, req.name)
    if not ok:
        return JSONResponse({"ok": False, "message": msg}, status_code=400)
    return {"ok": True, "message": msg, "user": user, "token": token}


@app.post("/auth/login")
def auth_login(req: AuthRequest):
    ok, msg, user, token = _auth.login(req.email, req.password)
    if not ok:
        return JSONResponse({"ok": False, "message": msg}, status_code=400)
    return {"ok": True, "message": msg, "user": user, "token": token}


@app.get("/auth/me")
def auth_me(authorization: str | None = Header(None)):
    token = (authorization or "").removeprefix("Bearer ").strip() or None
    user = _auth.user_for_token(token)
    if user is None:
        return JSONResponse({"ok": False}, status_code=401)
    return {"ok": True, "user": user}


@app.post("/auth/logout")
def auth_logout(authorization: str | None = Header(None)):
    token = (authorization or "").removeprefix("Bearer ").strip() or None
    _auth.logout(token)
    return {"ok": True}
