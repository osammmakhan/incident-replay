"""
Application state model for Incident Replay.

A single AppState dataclass is the only thing written to / read from
st.session_state. No component accesses session_state directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import streamlit as st

from incident_replay.models.schemas import IncidentReport

_SESSION_KEY = "incident_replay_state"

# Widget keys used by the intake form — centralised so state helpers
# and the intake component stay in sync.
INTAKE_KEYS = {
    "description": "intake_description",
    "repo_path":   "intake_repo_path",
    "log_path":    "intake_log_path",
    "stack_trace": "intake_stack_trace",
}

# Every session key this app ever writes — cleared on nuclear reset.
_ALL_WIDGET_KEYS = (
    list(INTAKE_KEYS.values())
    + ["intake_load_demo", "intake_replay",
       "sb_new_incident", "sb_reset",
       "_demo_pending", "_pipeline_started"]
)

# Demo incident data (single source of truth)
DEMO_INCIDENT = {
    "description": "Checkout returns HTTP 500 when discount is null.",
    "repo_path":   "demo_project/",
    "log_path":    "demo_project/logs/error.log",
    "stack_trace": (
        "Traceback (most recent call last):\n"
        '  File "api/checkout.py", line 31, in post\n'
        "    total = apply_discount(cart.total, cart.discount)\n"
        '  File "payment/discount.py", line 12, in apply_discount\n'
        "    return total * (1 - discount)\n"
        "TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'"
    ),
}


class RunStatus(Enum):
    IDLE    = auto()   # intake form shown
    RUNNING = auto()   # pipeline in progress
    DONE    = auto()   # report ready
    ERROR   = auto()   # pipeline raised an exception


@dataclass
class AppState:
    """All mutable UI state in one place."""

    status:    RunStatus = RunStatus.IDLE
    report:    Optional[IncidentReport] = None
    error_msg: str = ""

    # Intake form inputs
    incident_description: str = ""
    repo_path:            str = ""
    log_path:             str = ""
    stack_trace:          str = ""


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def load() -> AppState:
    """Read AppState from session_state, creating a default if absent."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = AppState()
    return st.session_state[_SESSION_KEY]


def save(state: AppState) -> None:
    """Persist an AppState back into session_state."""
    st.session_state[_SESSION_KEY] = state


def reset() -> None:
    """
    Nuclear reset: wipe ALL session state this app has ever written,
    then rerun from a clean IDLE AppState.

    Clears:
    - AppState (report, status, inputs, error)
    - All intake widget keys (prevents stale form values on reload)
    - The _demo_pending buffer
    - The _pipeline_started guard (prevents double-submission)
    - All button keys (forces Streamlit to re-render them fresh)
    """
    for key in _ALL_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state[_SESSION_KEY] = AppState()
    st.rerun()


def load_demo() -> None:
    """
    Load the demo incident and return to IDLE intake form.

    Clears any existing report/error first (handles repeat demo runs).
    Uses the _demo_pending pattern so widget keys are set before
    any widget is instantiated on the next run.
    """
    d = DEMO_INCIDENT

    # Clear any previous run result before loading demo
    st.session_state.pop("_pipeline_started", None)

    state = AppState(
        incident_description=d["description"],
        repo_path=d["repo_path"],
        log_path=d["log_path"],
        stack_trace=d["stack_trace"],
    )
    save(state)

    # Store under a pending key — intake component drains this at the
    # top of the next run, before any widget is instantiated.
    st.session_state["_demo_pending"] = {
        INTAKE_KEYS["description"]: d["description"],
        INTAKE_KEYS["repo_path"]:   d["repo_path"],
        INTAKE_KEYS["log_path"]:    d["log_path"],
        INTAKE_KEYS["stack_trace"]: d["stack_trace"],
    }

    st.rerun()
