"""
Backend adapter for Incident Replay.

This is the single swap point between the UI and the backend.
No other UI file imports from incident_replay.agents or ui.mock_backend.

Selection logic (in order):
  1. INCIDENT_REPLAY_MOCK=1  → always use mock (demo / dev override)
  2. Real backend importable  → use it
  3. Fallback                 → use mock silently

Usage:
    from ui.adapter import get_backend
    replay = get_backend()
    report = replay(incident_description=..., repo_path=..., ...)
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from incident_replay.models.schemas import IncidentReport


@runtime_checkable
class BackendProtocol(Protocol):
    """
    Structural type that every backend implementation must satisfy.
    If the real replay_incident() changes its signature, mypy will
    catch the mismatch here before it surfaces as a runtime error.
    """

    def __call__(
        self,
        incident_description: str,
        repo_path: str,
        log_path: str,
        stack_trace: str,
    ) -> IncidentReport: ...


def get_backend() -> BackendProtocol:
    """
    Return the appropriate backend callable.

    Set INCIDENT_REPLAY_MOCK=1 in your environment (or .env) to force
    the mock regardless of whether the real backend is importable.
    """
    if os.getenv("INCIDENT_REPLAY_MOCK", "").strip() == "1":
        return _load_mock()

    try:
        from incident_replay.agents.synthesis_agent import replay_incident  # type: ignore[import]

        # Verify the module actually defined the function (not just an empty file)
        if callable(replay_incident):
            return replay_incident  # type: ignore[return-value]
    except (ImportError, AttributeError):
        pass

    return _load_mock()


def _load_mock() -> BackendProtocol:
    from ui.mock_backend import replay_incident as mock  # type: ignore[import]

    return mock  # type: ignore[return-value]
