"""
Stage 2 — Investigation component.

Displays the four investigation areas and their results:
  • Log Investigator
  • Git Investigator
  • Code Investigator
  • Test Investigator

State is DERIVED from the IncidentReport — an investigator is COMPLETE
if report.evidence contains evidence with that source, NO FINDINGS if the
report exists but has no evidence for that source.

WAITING / ANALYZING states are only shown during the pipeline run via
render_running(), which is called from app.py before the report arrives.

Public API:
    render(report: IncidentReport) -> None
        Called after RunStatus.DONE — all investigators are complete.

    render_running(active_index: int) -> None
        Called during RunStatus.RUNNING — shows live investigator states.
        active_index: 0=log, 1=git, 2=code, 3=test
"""

from __future__ import annotations

import streamlit as st

from incident_replay.models.schemas import Evidence, IncidentReport

# ── Investigator definitions ─────────────────────────────────────────────────

INVESTIGATORS: list[dict] = [
    {
        "source": "log",
        "label":  "Log Investigator",
        "icon":   "📋",
        "desc":   "Scans application logs for exceptions, error patterns, and stack traces.",
    },
    {
        "source": "git",
        "label":  "Git Investigator",
        "icon":   "🔀",
        "desc":   "Analyses commit history, diffs, and recent changes near the failure.",
    },
    {
        "source": "code",
        "label":  "Code Investigator",
        "icon":   "📄",
        "desc":   "Inspects source code for the defect site and unsafe patterns.",
    },
    {
        "source": "test",
        "label":  "Test Investigator",
        "icon":   "🧪",
        "desc":   "Reviews test coverage to identify gaps that allowed the defect through.",
    },
]

# ── Status badge HTML helpers ────────────────────────────────────────────────

_STATUS_STYLES: dict[str, tuple[str, str]] = {
    # status → (background rgba, text colour)
    "WAITING":   ("rgba(128,128,128,0.10)", "inherit"),
    "ANALYZING": ("rgba(59,130,212,0.15)",  "#3b82d4"),
    "COMPLETE":  ("rgba(5,150,105,0.12)",   "#059669"),
    "NO FINDINGS": ("rgba(128,128,128,0.08)", "inherit"),
    "ERROR":     ("rgba(220,38,38,0.12)",   "#dc2626"),
}


def _badge(status: str) -> str:
    bg, color = _STATUS_STYLES.get(status, _STATUS_STYLES["WAITING"])
    pulse = (
        "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;"
        f"background:{color};margin-right:5px;opacity:0.9'></span>"
        if status == "ANALYZING"
        else ""
    )
    return (
        f"<span style='display:inline-flex;align-items:center;font-size:0.7rem;"
        f"font-weight:700;text-transform:uppercase;letter-spacing:0.06em;"
        f"background:{bg};color:{color};border-radius:4px;padding:2px 8px'>"
        f"{pulse}{status}</span>"
    )


# ── Card renderer ────────────────────────────────────────────────────────────

def _render_card(
    inv: dict,
    status: str,
    findings: list[Evidence],
) -> None:
    """Render one investigator card."""
    icon  = inv["icon"]
    label = inv["label"]
    desc  = inv["desc"]

    # Header row
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"padding:10px 14px 6px;'>"
        f"  <span style='font-size:1.3rem'>{icon}</span>"
        f"  <div style='flex:1'>"
        f"    <span style='font-weight:600;font-size:0.95rem'>{label}</span>"
        f"    <span style='font-size:0.75rem;opacity:0.5;margin-left:10px'>{desc}</span>"
        f"  </div>"
        f"  {_badge(status)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Findings (only when COMPLETE and evidence exists)
    if status == "COMPLETE" and findings:
        for ev in findings:
            st.markdown(
                f"<div style='margin:2px 14px 4px 48px;padding:8px 12px;"
                f"background:rgba(128,128,128,0.06);border-radius:6px;"
                f"border-left:3px solid rgba(128,128,128,0.2)'>"
                f"  <div style='font-size:0.78rem;opacity:0.55;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px'>"
                f"  {ev.location}</div>"
                f"  <div style='font-size:0.88rem'>{ev.observation}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    elif status == "NO FINDINGS":
        st.markdown(
            "<div style='margin:2px 14px 6px 48px;font-size:0.82rem;opacity:0.45'>"
            "No relevant evidence found for this source.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='border-bottom:1px solid rgba(128,128,128,0.12);margin:0 0 4px'></div>",
        unsafe_allow_html=True,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def render(report: IncidentReport) -> None:
    """
    Render investigation results from a completed IncidentReport.
    All investigators are COMPLETE (with or without findings).
    """
    st.markdown(
        '<div class="ir-section-header">'
        '<span class="ir-section-icon">🔍</span>'
        '<span class="ir-section-title">Investigation</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Group evidence by source for O(1) lookup per investigator
    by_source: dict[str, list[Evidence]] = {inv["source"]: [] for inv in INVESTIGATORS}
    for ev in report.evidence:
        if ev.source in by_source:
            by_source[ev.source].append(ev)

    for inv in INVESTIGATORS:
        source   = inv["source"]
        findings = by_source[source]
        status   = "COMPLETE" if findings else "NO FINDINGS"
        _render_card(inv, status, findings)

    # Timeline below the investigators
    if report.timeline:
        st.markdown(
            "<div style='margin-top:1rem'>"
            "<p style='font-size:0.75rem;font-weight:600;text-transform:uppercase;"
            "letter-spacing:0.05em;opacity:0.45;margin-bottom:6px'>Incident timeline</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        rows = []
        for event in report.timeline:
            rows.append(
                f'<div class="ir-timeline-row">'
                f'  <span class="ir-timeline-ts">{event.timestamp}</span>'
                f'  <span class="ir-timeline-desc">{event.description}</span>'
                f'</div>'
            )
        st.markdown("\n".join(rows), unsafe_allow_html=True)


def render_running(active_index: int) -> None:
    """
    Render investigator states during pipeline execution.
    active_index: which investigator is currently running (0–3).
    Investigators before active_index show COMPLETE,
    the active one shows ANALYZING, the rest show WAITING.
    No findings are shown — the report hasn't arrived yet.
    """
    st.markdown(
        '<div class="ir-section-header">'
        '<span class="ir-section-icon">🔍</span>'
        '<span class="ir-section-title">Investigation</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, inv in enumerate(INVESTIGATORS):
        if i < active_index:
            status = "COMPLETE"
        elif i == active_index:
            status = "ANALYZING"
        else:
            status = "WAITING"
        _render_card(inv, status, [])
