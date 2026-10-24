"""Pydantic request/response schemas for the Drishti API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="Free-form outdoor-activity question")
    lat: Optional[float] = Field(None, description="Optional live location latitude")
    lon: Optional[float] = Field(None, description="Optional live location longitude")
    session_id: Optional[str] = Field(None, description="Conversation id for multi-turn follow-ups")
    lang: Optional[str] = Field("en", description="Reply language: en, hi, mr, ta, bn")


class Persona(BaseModel):
    age: Optional[int] = None
    condition: Optional[str] = None  # asthma | senior | child | pregnancy


class Slots(BaseModel):
    activity: Optional[str] = None
    location_text: Optional[str] = None
    radius_km: Optional[float] = None
    time_expression: Optional[str] = None
    persona: Persona = Persona()
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


class Provenance(BaseModel):
    claim: str
    source: str
    value: Any = None
    fetched_at: Optional[str] = None


class Candidate(BaseModel):
    name: str
    lat: float
    lon: float
    score: float
    travel_minutes: Optional[float] = None
    hour: Optional[str] = None
    reasons: list[str] = []


class Recommendation(BaseModel):
    best: Optional[Candidate] = None
    golden_window: Optional[dict] = None
    avoid: Optional[dict] = None
    alternatives: list[Candidate] = []
    flip_note: Optional[str] = None
    go_no_go: Optional[str] = None  # GO | CAUTION | AVOID


class AskResponse(BaseModel):
    assistant: str = "Drishti"
    query: str
    session_id: Optional[str] = None
    lang: str = "en"
    keywords: list[dict] = []
    more_available: bool = False
    clarifying: bool = False
    quick_replies: list[str] = []
    centre: dict = {}
    parsed: Slots
    recommendation: Optional[Recommendation] = None
    advice: str
    surface: Optional[list[dict]] = None
    evidence: dict = {}
    provenance: list[Provenance] = []
    sources: list[str] = []
    data_gaps: list[str] = []
    model: str = ""
