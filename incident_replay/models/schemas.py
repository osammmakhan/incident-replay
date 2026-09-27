"""
Stable backend → UI contract for incident replay.

All models are JSON-serializable Pydantic v2 dataclasses so that Streamlit
can consume an ``IncidentReport`` directly via ``model.model_dump()``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    """A single timestamped event in the incident timeline."""

    timestamp: str
    description: str


class Evidence(BaseModel):
    """A piece of supporting evidence collected during analysis."""

    source: Literal["log", "git", "code", "test"]
    location: str # Where in the codebase/log/history (e.g. "app/views.py:42", "commit abc1234")
    observation: str  # What was found (the raw fact)
    relevance : str  # Why it matters to the root cause


class RegressionTest(BaseModel):
    """A regression test generated to prevent recurrence."""

    name: str
    code: str
    language: str = "python"


class SuggestedFix(BaseModel):
    """A concrete fix proposal for the identified root cause."""

    description: str
    patch: str = ""


class VerificationResult(BaseModel):
    """Outcome of running the suggested fix against the regression test."""

    passed: bool
    output: str = ""


class IncidentReport(BaseModel):
    """Top-level report produced by the incident-replay pipeline."""

    incident_summary: str
    timeline: list[TimelineEvent]
    root_cause: str
    evidence: list[Evidence]
    affected_files: list[str]
    affected_functions: list[str]
    suspicious_commit: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    regression_test: RegressionTest
    suggested_fix: SuggestedFix
    verification_result: VerificationResult
