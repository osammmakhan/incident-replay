"""
Incident Replay — application entry point.

Run with:
    streamlit run app.py

Set INCIDENT_REPLAY_MOCK=1 in your environment (or .env) to force
the mock backend regardless of whether the real backend is importable.
"""

from incident_replay.models.schemas import IncidentReport


import time
import traceback

import streamlit as st

# ── Page config (must be the first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="Incident Replay",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.adapter import get_backend
from ui.state import AppState, RunStatus, load, save, reset
from ui.styles import STAGES, inject_css
from ui.components import (
    sidebar,
    stage_progress,
    landing,
    incident,
    investigate,
    evidence,
    root_cause,
    regression_test,
    fix,
)

# Inject CSS once per page load
inject_css()

# Resolve backend once (falls back to mock if real backend not implemented)
_backend = get_backend()

# Index of the INVESTIGATE stage within STAGES (resolved at startup)
_INVESTIGATE_IDX = next(
    i for i, (_, label, __) in enumerate(STAGES) if label == "INVESTIGATE"
)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(state: AppState) -> AppState:
    """
    Animate through pipeline stages then call the backend adapter.

    Guards:
    - _pipeline_started in session_state prevents re-entry on accidental rerun
      while the pipeline is mid-execution.
    - All exceptions are caught; error_msg is set to a clean one-liner.
    """
    placeholder = st.empty()

    for i, (_, __, detail) in enumerate(STAGES):
        if i == _INVESTIGATE_IDX:
            for inv_i in range(len(investigate.INVESTIGATORS)):
                with placeholder.container():
                    stage_progress.render_running(i, detail)
                    investigate.render_running(inv_i)
                time.sleep(0.5)
        else:
            with placeholder.container():
                stage_progress.render_running(i, detail)
            time.sleep(0.45)

    placeholder.empty()

    report: IncidentReport = _backend(
        incident_description=state.incident_description,
        repo_path=state.repo_path,
        log_path=state.log_path,
        stack_trace=state.stack_trace,
    )
    state.report = report
    state.status = RunStatus.DONE
    return state


# ── Report renderer ───────────────────────────────────────────────────────────

def _render_report(state: AppState) -> None:
    """Render the full completed report, one component per stage."""
    assert state.report is not None
    report = state.report

    stage_progress.render(len(STAGES))
    st.divider()

    incident.render(report)
    st.divider()

    investigate.render(report)
    st.divider()

    evidence.render(report)
    st.divider()

    root_cause.render(report)
    st.divider()

    regression_test.render(report)
    st.divider()

    fix.render(report)


# ── Error screen ──────────────────────────────────────────────────────────────

def _render_error(state: AppState) -> None:
    """Render the error state with a prominent Reset Demo action."""
    st.markdown(
        "<h2 style='color:#dc2626'>⚠️ Pipeline Error</h2>",
        unsafe_allow_html=True,
    )
    st.error(state.error_msg or "An unknown error occurred.", icon="🚨")
    st.markdown(
        "<p style='opacity:0.7'>The pipeline encountered an error and could not complete. "
        "Reset to try again.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    col_reset, _ = st.columns([1, 3])
    with col_reset:
        if st.button("🔄  Reset Demo", type="primary", use_container_width=True,
                     key="error_reset"):
            reset()
    stage_progress.render(0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    state = load()
    state = sidebar.render(state)

    # ── IDLE: intake form ────────────────────────────────────────────────────
    if state.status == RunStatus.IDLE:
        state = landing.render(state)
        if state.status == RunStatus.RUNNING:
            # Guard against double-submission: mark pipeline as started
            # before saving so a browser refresh mid-run doesn't re-trigger.
            if st.session_state.get("_pipeline_started"):
                # Already running in another callback — do nothing.
                return
            st.session_state["_pipeline_started"] = True
            save(state)
            st.rerun()

    # ── RUNNING: execute pipeline ────────────────────────────────────────────
    elif state.status == RunStatus.RUNNING:
        try:
            state = _run_pipeline(state)
        except Exception as exc:
            # Surface a clean one-liner; full traceback goes to server log.
            short = str(exc).split("\n")[0] or type(exc).__name__
            state.status    = RunStatus.ERROR
            state.error_msg = f"{type(exc).__name__}: {short}"
            traceback.print_exc()
        finally:
            st.session_state.pop("_pipeline_started", None)
        save(state)
        st.rerun()

    # ── ERROR: show error + Reset Demo ──────────────────────────────────────
    elif state.status == RunStatus.ERROR:
        _render_error(state)

    # ── DONE: render full report ─────────────────────────────────────────────
    elif state.status == RunStatus.DONE and state.report is not None:
        _render_report(state)

    # ── Impossible state: reset silently ────────────────────────────────────
    else:
        reset()


if __name__ == "__main__":
    main()
