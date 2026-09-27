"""
Stage progress strip component.

Renders a horizontal pipeline indicator showing which stage is active,
which are complete, and which are pending.

Public API:
    render(active_index: int) -> None
        active_index: 0-based index of the currently running stage.
        Pass len(STAGES) to mark all stages as done.
"""

from __future__ import annotations

import streamlit as st

from ui.styles import STAGES


def render(active_index: int) -> None:
    """Render the horizontal stage progress strip."""

    total = len(STAGES)

    # Build HTML for the strip
    parts: list[str] = ['<div class="ir-stage-strip">']

    for i, (icon, label, _) in enumerate(STAGES):
        # Determine state
        if i < active_index:
            dot_cls = "ir-stage-dot done"
            lbl_cls = "ir-stage-label done"
            display_icon = "✓"
        elif i == active_index:
            dot_cls = "ir-stage-dot active"
            lbl_cls = "ir-stage-label active"
            display_icon = icon
        else:
            dot_cls = "ir-stage-dot"
            lbl_cls = "ir-stage-label"
            display_icon = icon

        # Connector before each item (except the first)
        if i > 0:
            connector_cls = "ir-stage-connector done" if i <= active_index else "ir-stage-connector"
            parts.append(f'<div class="{connector_cls}"></div>')

        parts.append(
            f'<div class="ir-stage-item">'
            f'  <div class="{dot_cls}">{display_icon}</div>'
            f'  <div class="{lbl_cls}">{label}</div>'
            f'</div>'
        )

    parts.append("</div>")

    st.markdown("\n".join(parts), unsafe_allow_html=True)


def render_running(active_index: int, detail: str) -> None:
    """
    Render the stage strip plus a live status line below it.
    Used during pipeline execution to show per-stage detail text.
    """
    render(active_index)
    icon, label, _ = STAGES[active_index]
    st.markdown(
        f"**{icon} {label}** — {detail}",
    )
    st.progress(
        (active_index + 1) / len(STAGES),
        text=f"Stage {active_index + 1} of {len(STAGES)}",
    )
