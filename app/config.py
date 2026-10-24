"""Drishti configuration. All secrets come from environment variables only."""
from __future__ import annotations

import os
from pathlib import Path

# The project root (the folder that holds run.sh and app/). All local files
# (.env, var/) are anchored here, never to the terminal's current folder,
# so the app behaves the same no matter where it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAR_DIR = Path(os.environ.get("DRISHTI_VAR_DIR") or (PROJECT_ROOT / "var"))


def _load_dotenv() -> None:
    """Fill in any missing environment variables from the project root's
    .env file. Real environment variables always win."""
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:  # noqa: F821 - path is a Path here
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = val.strip().strip('"').strip("'")
    except OSError:
        pass


_load_dotenv()


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "True", "yes")


# --- SLM ---
MODEL_ID = os.environ.get("DRISHTI_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ENABLE_SLM = _flag("DRISHTI_ENABLE_SLM", "0")  # if off, deterministic fallbacks run
SLM_MAX_NEW_TOKENS = int(os.environ.get("DRISHTI_MAX_NEW_TOKENS", "512"))

# --- optional official overlays (never hardcode keys) ---
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")
IMD_API_KEY = os.environ.get("IMD_API_KEY")
IMD_JWT = os.environ.get("IMD_JWT")

# --- behaviour ---
CACHE_TTL_SECONDS = int(os.environ.get("DRISHTI_CACHE_TTL", "1800"))
HTTP_TIMEOUT = int(os.environ.get("DRISHTI_HTTP_TIMEOUT", "12"))
USER_AGENT = os.environ.get(
    "DRISHTI_USER_AGENT", "Drishti/1.0 (GeoAI Hackathon; contact: bhavyatrivedi200829@gmail.com)"
)
DEFAULT_RADIUS_KM = float(os.environ.get("DRISHTI_DEFAULT_RADIUS_KM", "5"))
MAX_CANDIDATES = int(os.environ.get("DRISHTI_MAX_CANDIDATES", "10"))
MAX_FORECAST_DAYS = int(os.environ.get("DRISHTI_FORECAST_DAYS", "7"))

# --- thresholds for the recommendation ---
GOOD_SCORE = 60      # >= this is a "go"
AVOID_SCORE = 35     # <= this is "avoid"
