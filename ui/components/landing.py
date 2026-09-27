"""
Incident Intake stage — the starting point of every incident investigation.

Shown when RunStatus.IDLE. Provides:
- Full-width intake form (description, repo path, log path, stack trace)
- "Load Demo Incident" prominent button
- "REPLAY INCIDENT" primary CTA
- Inline validation and error feedback
- Path existence warning (non-blocking)

Public API:
    render(state: AppState) -> AppState
        Renders the intake form and returns state with updated inputs.
        Sets state.status = RunStatus.RUNNING when the user submits valid inputs.
        Sets state.status = RunStatus.ERROR on backend exception (propagated from app.py).
"""

from __future__ import annotations

import os

import streamlit as st

from ui.state import AppState, DEMO_INCIDENT, INTAKE_KEYS, RunStatus, load_demo, reset


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate(description: str, repo_path: str) -> list[str]:
    """Return a list of blocking error strings. Empty list = valid."""
    errors: list[str] = []
    if not description.strip():
        errors.append("Incident description is required.")
    if not repo_path.strip():
        errors.append("Repository path is required.")
    return errors


def _path_warnings(repo_path: str, log_path: str) -> list[str]:
    """
    Return non-blocking warnings about paths that don't exist locally.
    Suppressed for the demo incident (mock backend doesn't need real files).
    """
    from ui.state import DEMO_INCIDENT
    if repo_path.strip() == DEMO_INCIDENT["repo_path"].strip():
        return []   # demo paths are intentionally absent — no warning
    warnings: list[str] = []
    if repo_path.strip() and not os.path.exists(repo_path.strip()):
        warnings.append(f"Repository path not found locally: `{repo_path.strip()}`")
    if log_path.strip() and not os.path.exists(log_path.strip()):
        warnings.append(f"Log path not found locally: `{log_path.strip()}`")
    return warnings


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(state: AppState) -> AppState:
    """Render the Incident Intake stage. Returns updated state."""

    # Drain any pending demo values before widgets are instantiated
    if "_demo_pending" in st.session_state:
        for k, v in st.session_state.pop("_demo_pending").items():
            st.session_state[k] = v

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0'>🔁 Incident Replay</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='opacity:0.6;margin-top:4px;font-size:1rem'>"
        "AI-assisted root cause analysis · regression test generation · verified fix"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Two-column layout: form left, context right ──────────────────────────
    col_form, col_info = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown("### 🚨 Incident Intake")
        st.markdown(
            "Describe the production incident. The pipeline will investigate, "
            "identify the root cause, generate a regression test, and propose a verified fix."
        )
        st.markdown("")

        # Demo loader — prominent, above the form
        if st.button(
            "⚡ Load Demo Incident",
            key="intake_load_demo",
            use_container_width=False,
            help="Pre-fill the form with a real checkout incident scenario",
        ):
            load_demo()

        # Show a subtle indicator when demo is loaded
        is_demo_loaded = (
            st.session_state.get(INTAKE_KEYS["description"], "").strip()
            == DEMO_INCIDENT["description"].strip()
        )
        if is_demo_loaded:
            st.success("✓ Demo incident loaded — click **REPLAY INCIDENT** to run.", icon=None)

        st.markdown("---")

        # ── Form fields ───────────────────────────────────────────────────────
        description = st.text_area(
            "Incident description \\*",
            placeholder="e.g. Checkout returns HTTP 500 when discount is null.",
            height=100,
            key=INTAKE_KEYS["description"],
            help="Describe the symptom as observed in production.",
        )

        repo_path = st.text_input(
            "Repository path \\*",
            placeholder="demo_project/  or  /path/to/your/repo",
            key=INTAKE_KEYS["repo_path"],
            help="Local path to the repository to be analysed.",
        )

        log_path = st.text_input(
            "Log file path",
            placeholder="demo_project/logs/error.log  (optional)",
            key=INTAKE_KEYS["log_path"],
            help="Path to the application log file. Optional but improves analysis.",
        )

        stack_trace = st.text_area(
            "Stack trace",
            placeholder=(
                "Paste the exception traceback here…  (optional)\n\n"
                "Traceback (most recent call last):\n"
                "  File \"api/checkout.py\", line 31, in post\n"
                "    ..."
            ),
            height=140,
            key=INTAKE_KEYS["stack_trace"],
            help="Full exception traceback from logs or monitoring. Optional.",
        )

        # ── Validation ────────────────────────────────────────────────────────
        errors   = _validate(description, repo_path)
        warnings = _path_warnings(repo_path, log_path)

        for w in warnings:
            st.warning(w, icon="⚠️")

        # ── Primary action ────────────────────────────────────────────────────
        st.markdown("")
        fields_ready = bool(description.strip() and repo_path.strip())
        replay_clicked = st.button(
            "▶  REPLAY INCIDENT",
            type="primary",
            use_container_width=True,
            key="intake_replay",
            disabled=not fields_ready,
            help=None if fields_ready else "Enter an incident description and repository path first.",
        )

        if replay_clicked:
            if errors:
                for e in errors:
                    st.error(e, icon="🚫")
            else:
                state.incident_description = description
                state.repo_path            = repo_path
                state.log_path             = log_path
                state.stack_trace          = stack_trace
                state.status               = RunStatus.RUNNING

        st.markdown(
            "<p style='font-size:0.75rem;opacity:0.45;margin-top:8px'>"
            "\\* Required fields"
            "</p>",
            unsafe_allow_html=True,
        )

    # ── Right column: pipeline overview ──────────────────────────────────────
    with col_info:
        st.markdown("### Pipeline")
        st.markdown(
            "<p style='opacity:0.6;font-size:0.9rem'>"
            "When you replay an incident, the pipeline runs these stages automatically."
            "</p>",
            unsafe_allow_html=True,
        )

        from ui.styles import STAGES
        for icon, label, desc in STAGES:
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;gap:12px;"
                f"padding:10px 0;border-bottom:1px solid rgba(128,128,128,0.15)'>"
                f"  <div style='font-size:1.4rem;width:28px;flex-shrink:0'>{icon}</div>"
                f"  <div>"
                f"    <div style='font-weight:600;font-size:0.85rem;"
                f"text-transform:uppercase;letter-spacing:0.04em'>{label}</div>"
                f"    <div style='font-size:0.78rem;opacity:0.55;margin-top:2px'>{desc}</div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown(
            "<p style='font-size:0.78rem;opacity:0.5'>"
            "No investigation logic runs in the UI. "
            "All analysis is performed by the backend pipeline."
            "</p>",
            unsafe_allow_html=True,
        )

    return state
