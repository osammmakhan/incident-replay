"""
Stage 5 — Regression Test component.

Displays the generated regression test: what it tests and the code.
The before/after execution results live in fix.py so the full
❌ → fix → ✅ sequence is visible in one connected view.

Data sources (all from IncidentReport):
  - regression_test.name      Test identifier
  - regression_test.code      Generated test body
  - regression_test.language  Language for syntax highlighting

Public API:
    render(report: IncidentReport) -> None
"""

from __future__ import annotations

import streamlit as st

from incident_replay.models.schemas import IncidentReport

_LABEL = (
    "font-size:0.68rem;font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.07em;opacity:0.45"
)


def render(report: IncidentReport) -> None:
    """Render the Regression Test stage."""

    rt = report.regression_test

    st.markdown(
        '<div class="ir-section-header">'
        '<span class="ir-section-icon">🧪</span>'
        '<span class="ir-section-title">Regression Test</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Metadata row
    st.markdown(
        f"<div style='display:flex;gap:28px;margin-bottom:14px;flex-wrap:wrap'>"

        f"  <div>"
        f"    <div style='{_LABEL};margin-bottom:3px'>Test name</div>"
        f"    <div style='font-family:ui-monospace,monospace;font-size:0.88rem'>"
        f"    {rt.name}</div>"
        f"  </div>"

        f"  <div>"
        f"    <div style='{_LABEL};margin-bottom:3px'>Language</div>"
        f"    <div style='font-family:ui-monospace,monospace;font-size:0.88rem'>"
        f"    {rt.language}</div>"
        f"  </div>"

        f"  <div>"
        f"    <div style='{_LABEL};margin-bottom:3px'>Represents</div>"
        f"    <div style='font-size:0.88rem;opacity:0.8'>"
        f"    The exact production scenario that caused the incident</div>"
        f"  </div>"

        f"</div>",
        unsafe_allow_html=True,
    )

    st.code(rt.code, language=rt.language)

    st.markdown(
        "<p style='font-size:0.78rem;opacity:0.45;margin-top:4px'>"
        "This test is used twice: once to confirm the defect is reproducible, "
        "and again after the fix to confirm it is resolved."
        "</p>",
        unsafe_allow_html=True,
    )
