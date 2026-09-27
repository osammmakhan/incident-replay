"""
Stage 7 — Verification component.

Renders the PASS/FAIL badge and expandable test runner output.

Public API:
    render(report: IncidentReport) -> None
"""

from __future__ import annotations

import streamlit as st

from incident_replay.models.schemas import IncidentReport


def render(report: IncidentReport) -> None:
    """Render the verification result section."""

    vr = report.verification_result

    st.markdown(
        '<div class="ir-section-header">'
        '<span class="ir-section-icon">✅</span>'
        '<span class="ir-section-title">Verification</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if vr.passed:
        badge_cls = "ir-badge ir-badge-pass"
        badge_text = "✓ PASS — fix verified"
    else:
        badge_cls = "ir-badge ir-badge-fail"
        badge_text = "✗ FAIL — fix did not resolve the issue"

    st.markdown(
        f'<div style="margin-bottom:1rem">'
        f'<span class="{badge_cls}">{badge_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if vr.output:
        with st.expander("Test runner output"):
            st.code(vr.output, language="")
