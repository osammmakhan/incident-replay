"""
Stage 1 — Incident component.

Renders the incident summary banner at the top of the report.

Public API:
    render(report: IncidentReport) -> None
"""

from __future__ import annotations

import streamlit as st

from incident_replay.models.schemas import IncidentReport


def render(report: IncidentReport) -> None:
    """Render the incident summary section header and banner."""

    st.markdown(
        '<div class="ir-section-header">'
        '<span class="ir-section-icon">🚨</span>'
        '<span class="ir-section-title">Incident</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="ir-incident-banner">{report.incident_summary}</div>',
        unsafe_allow_html=True,
    )
