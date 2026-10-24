"""
Lightweight in-memory conversation store.

Lets Drishti behave like a chat: when a question is missing a detail it asks
back, remembers the partial question, and merges the user's next message so the
answer can be completed in the same conversation.

In-memory is fine for a single-instance hackathon deployment. For multiple
workers, swap this for Redis behind the same four functions.
"""
from __future__ import annotations

import time
import uuid

_TTL = 1800          # 30 minutes of inactivity
_MAX_MESSAGES = 6    # keep the tail only, so context stays relevant

_SESSIONS: dict[str, dict] = {}


def _cleanup():
    now = time.time()
    for sid in [s for s, v in _SESSIONS.items() if now - v["last"] > _TTL]:
        _SESSIONS.pop(sid, None)


def create() -> str:
    _cleanup()
    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = {"messages": [], "pending": None, "created": time.time(), "last": time.time()}
    return sid


def exists(sid: str | None) -> bool:
    return bool(sid) and sid in _SESSIONS


def append(sid: str, text: str):
    if not exists(sid):
        sid = create()
    s = _SESSIONS[sid]
    s["messages"].append(text.strip())
    s["messages"] = s["messages"][-_MAX_MESSAGES:]
    s["last"] = time.time()
    return sid


def reset(sid: str, text: str):
    """Start the conversation fresh from this message (topic change)."""
    if not exists(sid):
        sid = create()
    _SESSIONS[sid]["messages"] = [text.strip()]
    _SESSIONS[sid]["pending"] = None
    _SESSIONS[sid]["activity_label"] = None
    _SESSIONS[sid]["activity"] = None
    _SESSIONS[sid]["last"] = time.time()
    return sid


def set_pending(sid: str | None, kind: str | None):
    """Record what the assistant just asked for: 'activity' or 'location'."""
    if exists(sid):
        _SESSIONS[sid]["pending"] = kind
        _SESSIONS[sid]["last"] = time.time()


def get_pending(sid: str | None) -> str | None:
    if not exists(sid):
        return None
    return _SESSIONS[sid].get("pending")


def set_activity_label(sid: str | None, label: str | None):
    """Remember the user's own word for the activity across turns."""
    if exists(sid) and label:
        _SESSIONS[sid]["activity_label"] = label
        _SESSIONS[sid]["last"] = time.time()


def get_activity_label(sid: str | None) -> str | None:
    if not exists(sid):
        return None
    return _SESSIONS[sid].get("activity_label")


def get_radius_mult(sid: str | None) -> float:
    st = _SESSIONS.get(sid or "")
    return float((st or {}).get("radius_mult", 1.0))


def grow_radius(sid: str | None) -> float:
    """Each "suggest more" doubles the search radius, up to 8x."""
    st = _SESSIONS.get(sid or "")
    if st is None:
        return 1.0
    mult = min(8.0, float(st.get("radius_mult", 1.0)) * 2.0)
    st["radius_mult"] = mult
    return mult


def reset_radius(sid: str | None):
    st = _SESSIONS.get(sid or "")
    if st is not None:
        st["radius_mult"] = 1.0


def set_activity(sid: str | None, activity: str | None):
    """Remember the resolved activity so a later reply can rely on it."""
    if exists(sid) and activity:
        _SESSIONS[sid]["activity"] = activity


def get_activity(sid: str | None) -> str | None:
    if not exists(sid):
        return None
    return _SESSIONS[sid].get("activity")


def context_text(sid: str | None) -> str:
    if not exists(sid):
        return ""
    return " | ".join(_SESSIONS[sid]["messages"])


def message_count(sid: str | None) -> int:
    if not exists(sid):
        return 0
    return len(_SESSIONS[sid]["messages"])
