"""
Stage 3 — Evidence & Reconstruction component.

Presents the full reconstruction of the incident from observed signal to
correlated evidence to root-cause hypothesis. Two primary sections:

  1. Incident Timeline  — timestamped sequence of observed events
  2. Evidence           — each item with source / location / observation / relevance

Below the evidence, correlated diagnostics are shown:
  - Affected files & functions
  - Suspicious commit
  - Confidence score

Engineering-grade terminology throughout. No vague AI language.

Public API:
    render(report: IncidentReport) -> None
"""

from __future__ import annotations

import streamlit as st

from incident_replay.models.schemas import IncidentReport

# Source → (accent colour, label)
_SOURCE_META: dict[str, tuple[str, str]] = {
    "log":  ("#f59e0b", "LOG"),
    "git":  ("#8b5cf6", "GIT"),
    "code": ("#3b82d4", "CODE"),
    "test": ("#10b981", "TEST"),
}


# ── Section heading helper ────────────────────────────────────────────────────

def _section_heading(icon: str, title: str, subtitle: str = "") -> None:
    sub_html = (
        f"<span style='font-size:0.8rem;opacity:0.55;margin-left:10px'>{subtitle}</span>"
        if subtitle else ""
    )
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:6px;margin-bottom:12px'>"
        f"  <span style='font-size:1.1rem'>{icon}</span>"
        f"  <span style='font-weight:700;font-size:1rem;text-transform:uppercase;"
        f"letter-spacing:0.04em'>{title}</span>"
        f"  {sub_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Timeline ──────────────────────────────────────────────────────────────────

def _render_timeline(report: IncidentReport) -> None:
    count = len(report.timeline)
    _section_heading("📅", "Incident Timeline", f"{count} event{'s' if count != 1 else ''} observed")

    if not report.timeline:
        st.caption("No timeline events recorded.")
        return

    rows: list[str] = []
    for i, event in enumerate(report.timeline):
        is_last = i == len(report.timeline) - 1
        border = "none" if is_last else "1px solid rgba(128,128,128,0.12)"
        rows.append(
            f"<div style='display:grid;grid-template-columns:110px 1fr;"
            f"gap:12px;padding:9px 0;border-bottom:{border};align-items:baseline'>"
            f"  <span style='font-family:ui-monospace,monospace;font-size:0.78rem;"
            f"opacity:0.55;white-space:nowrap'>{event.timestamp}</span>"
            f"  <span style='font-size:0.9rem'>{event.description}</span>"
            f"</div>"
        )
    st.markdown("\n".join(rows), unsafe_allow_html=True)


# ── Evidence cards ────────────────────────────────────────────────────────────

def _render_evidence(report: IncidentReport) -> None:
    count = len(report.evidence)
    _section_heading("🧩", "Observed Evidence", f"{count} item{'s' if count != 1 else ''} correlated")

    if not report.evidence:
        st.info("No evidence was collected during analysis.")
        return

    for i, ev in enumerate(report.evidence):
        accent, source_label = _SOURCE_META.get(ev.source, ("#888", ev.source.upper()))

        st.markdown(
            f"<div style='border-left:3px solid {accent};"
            f"background:rgba(128,128,128,0.04);border-radius:0 6px 6px 0;"
            f"padding:10px 14px;margin-bottom:10px'>"

            # Header row: source tag + location
            f"  <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
            f"    <span style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.07em;background:{accent}22;color:{accent};"
            f"border-radius:3px;padding:1px 7px'>{source_label}</span>"
            f"    <span style='font-family:ui-monospace,monospace;font-size:0.78rem;"
            f"opacity:0.6'>{ev.location}</span>"
            f"  </div>"

            # Observation
            f"  <div style='margin-bottom:4px'>"
            f"    <span style='font-size:0.68rem;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:0.06em;opacity:0.45'>Observed</span>"
            f"  </div>"
            f"  <div style='font-size:0.9rem;font-family:ui-monospace,monospace;"
            f"background:rgba(128,128,128,0.07);border-radius:4px;padding:6px 10px;"
            f"margin-bottom:8px;line-height:1.5'>{ev.observation}</div>"

            # Relevance
            f"  <div style='margin-bottom:4px'>"
            f"    <span style='font-size:0.68rem;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:0.06em;opacity:0.45'>Correlated</span>"
            f"  </div>"
            f"  <div style='font-size:0.87rem;opacity:0.85;line-height:1.55'>{ev.relevance}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def render(report: IncidentReport) -> None:
    """Render the Evidence & Reconstruction stage."""

    st.markdown(
        '<div class="ir-section-header">'
        '<span class="ir-section-icon">🧩</span>'
        '<span class="ir-section-title">Evidence &amp; Reconstruction</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _render_timeline(report)

    st.markdown(
        "<div style='border-top:1px solid rgba(128,128,128,0.12);margin:18px 0'></div>",
        unsafe_allow_html=True,
    )

    _render_evidence(report)
