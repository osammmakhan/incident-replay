"""
Sidebar component.

Displays context-appropriate content for each RunStatus:

  IDLE     — instructions + Load Demo shortcut + Reset
  RUNNING  — pipeline progress message (no buttons to avoid interference)
  DONE     — compact ✓ stage checklist + Reset Demo + New Incident
  ERROR    — error indicator + Reset Demo

Reset Demo is always prominent (except during RUNNING) because it is the
primary demo recovery action.

Public API:
    render(state: AppState) -> AppState
"""

from __future__ import annotations

import streamlit as st

from ui.state import AppState, RunStatus, load_demo, reset
from ui.styles import STAGES


def render(state: AppState) -> AppState:
    """Render the sidebar. Returns state unchanged (sidebar is read-only)."""

    # ── Logo / header ─────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='padding:8px 0 4px'>"
        "<span style='font-size:1.4rem'>🔁</span>"
        "<span style='font-weight:700;font-size:1.1rem;margin-left:8px'>"
        "Incident Replay</span>"
        "</div>"
        "<div style='font-size:0.78rem;opacity:0.55;margin-bottom:4px'>"
        "AI-assisted root cause &amp; fix pipeline"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # ── Status-specific content ───────────────────────────────────────────────
    if state.status == RunStatus.IDLE:
        st.sidebar.markdown(
            "<p style='font-size:0.85rem;opacity:0.7;margin-bottom:12px'>"
            "Fill in the incident details and click "
            "<strong>▶ REPLAY INCIDENT</strong> to start."
            "</p>",
            unsafe_allow_html=True,
        )
        # Load Demo shortcut in sidebar too
        if st.sidebar.button(
            "⚡  Load Demo Incident",
            use_container_width=True,
            key="sb_load_demo",
        ):
            load_demo()

    elif state.status == RunStatus.RUNNING:
        st.sidebar.markdown(
            "<div style='display:flex;align-items:center;gap:8px;"
            "font-size:0.85rem;opacity:0.7'>"
            "  <span>⏳</span>"
            "  <span>Pipeline running…</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # No buttons during run — prevents any state interference.
        return state

    elif state.status == RunStatus.DONE:
        # Compact ✓ stage checklist
        st.sidebar.markdown(
            "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.06em;opacity:0.45;margin-bottom:8px'>Pipeline complete</p>",
            unsafe_allow_html=True,
        )
        for icon, label, _ in STAGES:
            st.sidebar.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"font-size:0.82rem;padding:2px 0'>"
                f"  <span style='color:#059669;font-weight:700;width:14px'>✓</span>"
                f"  <span style='opacity:0.75'>{icon} {label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.sidebar.markdown("")

        if st.sidebar.button(
            "＋  New Incident",
            use_container_width=True,
            key="sb_new_incident",
        ):
            reset()

    elif state.status == RunStatus.ERROR:
        st.sidebar.markdown(
            "<div style='color:#dc2626;font-size:0.85rem;font-weight:600;"
            "margin-bottom:8px'>⚠️ Pipeline error</div>"
            "<p style='font-size:0.82rem;opacity:0.7'>"
            "An error occurred. Reset to try again."
            "</p>",
            unsafe_allow_html=True,
        )

    # ── Reset Demo — always visible except during RUNNING ────────────────────
    st.sidebar.divider()
    if st.sidebar.button(
        "🔄  Reset Demo",
        use_container_width=True,
        type="primary" if state.status == RunStatus.ERROR else "secondary",
        key="sb_reset",
    ):
        reset()

    return state
