"""
Stage 4 — Root Cause component.

Displays the root cause statement returned by the backend, immediately
followed by the supporting evidence items that led to it, then the
diagnostic metadata (affected files/functions, suspicious commit,
confidence).

Layout:
  ROOT CAUSE statement
  └─ Supporting evidence  (compact: source tag · location · relevance)
  Diagnostics grid
    Affected files | Affected functions | Commit | Confidence

All values are sourced exclusively from IncidentReport. Nothing is
inferred, invented, or labelled as an AI opinion.

Public API:
    render(report: IncidentReport) -> None
"""

from __future__ import annotations

import streamlit as st

from incident_replay.models.schemas import Evidence, IncidentReport

# Source → (accent colour, short label)
_SOURCE_META: dict[str, tuple[str, str]] = {
    "log":  ("#f59e0b", "LOG"),
    "git":  ("#8b5cf6", "GIT"),
    "code": ("#3b82d4", "CODE"),
    "test": ("#10b981", "TEST"),
}

_DIVIDER = (
    "<div style='border-top:1px solid rgba(128,128,128,0.12);margin:20px 0'></div>"
)

_LABEL = (
    "font-size:0.68rem;font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.07em;opacity:0.45"
)


# ── Root cause statement ──────────────────────────────────────────────────────

def _render_statement(report: IncidentReport) -> None:
    st.markdown(
        f"<p style='{_LABEL};margin-bottom:8px'>Root Cause</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:rgba(245,158,11,0.07);"
        f"border:1px solid rgba(245,158,11,0.28);"
        f"border-left:4px solid #f59e0b;"
        f"border-radius:6px;padding:16px 20px;"
        f"font-size:1rem;line-height:1.7'>"
        f"{report.root_cause}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Supporting evidence (compact) ────────────────────────────────────────────

def _render_supporting_evidence(evidence: list[Evidence]) -> None:
    if not evidence:
        return

    st.markdown(
        f"<p style='{_LABEL};margin:18px 0 10px'>Supporting Evidence</p>",
        unsafe_allow_html=True,
    )

    for ev in evidence:
        accent, source_label = _SOURCE_META.get(ev.source, ("#888888", ev.source.upper()))
        st.markdown(
            f"<div style='display:grid;grid-template-columns:auto 1fr;"
            f"gap:10px;align-items:start;"
            f"border-left:3px solid {accent};"
            f"padding:8px 12px;margin-bottom:6px;"
            f"background:rgba(128,128,128,0.04);border-radius:0 5px 5px 0'>"

            # Left: source badge
            f"  <div>"
            f"    <span style='font-size:0.65rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.07em;background:{accent}1a;color:{accent};"
            f"border-radius:3px;padding:2px 6px;white-space:nowrap'>{source_label}</span>"
            f"  </div>"

            # Right: location + relevance
            f"  <div>"
            f"    <div style='font-family:ui-monospace,monospace;font-size:0.75rem;"
            f"opacity:0.55;margin-bottom:3px'>{ev.location}</div>"
            f"    <div style='font-size:0.87rem;line-height:1.5'>{ev.relevance}</div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _render_diagnostics(report: IncidentReport) -> None:
    st.markdown(
        f"<p style='{_LABEL};margin-bottom:12px'>Diagnostics</p>",
        unsafe_allow_html=True,
    )

    col_files, col_fns, col_meta = st.columns([2, 2, 2], gap="large")

    # Affected files
    with col_files:
        st.markdown(
            f"<p style='{_LABEL};margin-bottom:6px'>Affected Files</p>",
            unsafe_allow_html=True,
        )
        if report.affected_files:
            for f in report.affected_files:
                st.markdown(
                    f"<div style='font-family:ui-monospace,monospace;font-size:0.82rem;"
                    f"padding:3px 0;opacity:0.85'>📄 {f}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span style='font-size:0.82rem;opacity:0.4'>None identified</span>",
                unsafe_allow_html=True,
            )

    # Affected functions
    with col_fns:
        st.markdown(
            f"<p style='{_LABEL};margin-bottom:6px'>Affected Functions</p>",
            unsafe_allow_html=True,
        )
        if report.affected_functions:
            for fn in report.affected_functions:
                st.markdown(
                    f"<div style='font-family:ui-monospace,monospace;font-size:0.82rem;"
                    f"padding:3px 0;opacity:0.85'>ƒ {fn}()</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span style='font-size:0.82rem;opacity:0.4'>None identified</span>",
                unsafe_allow_html=True,
            )

    # Commit + confidence
    with col_meta:
        if report.suspicious_commit:
            st.markdown(
                f"<p style='{_LABEL};margin-bottom:6px'>Suspicious Commit</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='display:inline-flex;align-items:center;gap:7px;"
                f"background:rgba(139,92,246,0.10);"
                f"border:1px solid rgba(139,92,246,0.25);"
                f"border-radius:5px;padding:4px 11px;margin-bottom:14px'>"
                f"  <span style='opacity:0.6'>🔀</span>"
                f"  <span style='font-family:ui-monospace,monospace;font-size:0.88rem;"
                f"font-weight:600'>{report.suspicious_commit}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        pct = int(report.confidence * 100)
        bar_colour = (
            "#059669" if pct >= 80
            else "#f59e0b" if pct >= 60
            else "#dc2626"
        )
        st.markdown(
            f"<p style='{_LABEL};margin-bottom:6px'>Confidence</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background:rgba(128,128,128,0.12);border-radius:4px;"
            f"height:7px;overflow:hidden;margin-bottom:5px'>"
            f"  <div style='width:{pct}%;height:100%;background:{bar_colour};"
            f"border-radius:4px'></div>"
            f"</div>"
            f"<span style='font-size:1rem;font-weight:700;color:{bar_colour}'>{pct}%</span>"
            f"<span style='font-size:0.75rem;opacity:0.45;margin-left:5px'>"
            f"based on {len(report.evidence)} evidence item"
            f"{'s' if len(report.evidence) != 1 else ''}</span>",
            unsafe_allow_html=True,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def render(report: IncidentReport) -> None:
    """Render the Root Cause stage."""

    st.markdown(
        '<div class="ir-section-header">'
        '<span class="ir-section-icon">🎯</span>'
        '<span class="ir-section-title">Root Cause</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _render_statement(report)

    _render_supporting_evidence(report.evidence)

    st.markdown(_DIVIDER, unsafe_allow_html=True)

    _render_diagnostics(report)
