"""
Styles and shared visual constants for Incident Replay UI.

All CSS is injected once from app.py via inject_css().
Components import constants (colours, icons) from here rather than
embedding them inline.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EVIDENCE_SOURCE_ICONS: dict[str, str] = {
    "log": "📋",
    "git": "🔀",
    "code": "📄",
    "test": "🧪",
}

# Pipeline stage definitions: (icon, label, running description)
STAGES: list[tuple[str, str, str]] = [
    ("🚨", "INCIDENT",        "Parsing incident description…"),
    ("🔍", "INVESTIGATE",     "Collecting logs, git history, and code…"),
    ("🧩", "EVIDENCE",        "Correlating evidence sources…"),
    ("🎯", "ROOT CAUSE",      "Synthesising root cause…"),
    ("🧪", "REGRESSION TEST", "Generating regression test…"),
    ("🔧", "FIX",             "Applying suggested fix…"),
    ("✅", "VERIFICATION",    "Verifying fix passes regression test…"),
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Stage progress bar ──────────────────────────────────────── */
.ir-stage-strip {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0 2rem 0;
    overflow-x: auto;
}
.ir-stage-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    min-width: 80px;
}
.ir-stage-dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    border: 2px solid #e5e7eb;
    background: #ffffff;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    position: relative;
    z-index: 1;
}
.ir-stage-dot.active {
    border-color: #3b82d4;
    background: #eff6ff;
}
.ir-stage-dot.done {
    border-color: #059669;
    background: #d1fae5;
}
.ir-stage-connector {
    flex: 1;
    height: 2px;
    background: #e5e7eb;
    margin-top: -16px; /* align with dot centre */
}
.ir-stage-connector.done {
    background: #059669;
}
.ir-stage-label {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: inherit;
    opacity: 0.6;
    margin-top: 6px;
    text-align: center;
}
.ir-stage-label.active { color: #3b82d4; opacity: 1; }
.ir-stage-label.done   { color: #059669; opacity: 1; }

/* ── Section headers ─────────────────────────────────────────── */
.ir-section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 0.8rem;
    background: rgba(128,128,128,0.08);
    border-radius: 8px 8px 0 0;
    border-bottom: 2px solid rgba(128,128,128,0.2);
    margin-bottom: 1rem;
}
.ir-section-icon { font-size: 1.2rem; }
.ir-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: inherit;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Evidence cards ──────────────────────────────────────────── */
.ir-evidence-card {
    border-left: 3px solid #3b82d4;
    background: rgba(128,128,128,0.06);
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.ir-evidence-source {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: inherit;
    opacity: 0.65;
    margin-bottom: 3px;
}
.ir-evidence-desc {
    font-size: 0.95rem;
    color: inherit;
    font-weight: 500;
}

/* Source colour accents */
.ir-evidence-card.source-log  { border-color: #f59e0b; }
.ir-evidence-card.source-git  { border-color: #8b5cf6; }
.ir-evidence-card.source-code { border-color: #3b82d4; }
.ir-evidence-card.source-test { border-color: #10b981; }

/* ── Commit pill ─────────────────────────────────────────────── */
.ir-commit-pill {
    font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
    background: rgba(128,128,128,0.12);
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.88rem;
    color: inherit;
}

/* ── Verification badges ─────────────────────────────────────── */
.ir-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 6px 20px;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.03em;
}
.ir-badge-pass { background: #d1fae5; color: #065f46; }
.ir-badge-fail { background: #fee2e2; color: #991b1b; }

/* ── Incident summary banner ─────────────────────────────────── */
.ir-incident-banner {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.4);
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 1.5rem;
    font-size: 0.97rem;
    color: inherit;
    line-height: 1.6;
}

/* ── Timeline ────────────────────────────────────────────────── */
.ir-timeline-row {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    padding: 6px 0;
    border-bottom: 1px solid rgba(128,128,128,0.15);
}
.ir-timeline-ts {
    font-family: ui-monospace, monospace;
    font-size: 0.82rem;
    color: inherit;
    opacity: 0.6;
    white-space: nowrap;
    min-width: 100px;
}
.ir-timeline-desc {
    font-size: 0.95rem;
    color: inherit;
}

/* ── Confidence meter label ──────────────────────────────────── */
.ir-confidence-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: inherit;
    opacity: 0.6;
    margin-bottom: 2px;
}
</style>
"""


def inject_css() -> None:
    """Inject all UI styles. Call once at the top of app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
