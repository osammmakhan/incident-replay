# How IBM Bob was used to build Incident Replay UI

IBM Bob (the AI coding assistant embedded in VS Code) was the primary development tool for the entire UI layer of this project. This document describes how it was used across different phases.

---

## What Bob built

The entire `ui/` layer was authored through Bob in a single extended session, starting from a blank repository with only the teammate's `incident_replay/models/schemas.py` as the contract.

Bob produced:

| File | Lines | What it does |
|---|---|---|
| `ui/state.py` | 140 | `AppState` dataclass, `RunStatus` enum, `INTAKE_KEYS`, `_ALL_WIDGET_KEYS`, `load/save/reset/load_demo` with nuclear reset and demo-pending pattern |
| `ui/adapter.py` | 69 | `BackendProtocol` (structural typing), `get_backend()` with env var override and import fallback |
| `ui/mock_backend.py` | 119 | Deterministic `IncidentReport` with full checkout/null-discount scenario |
| `ui/styles.py` | 220 | Full CSS theme, `STAGES` pipeline constants, `EVIDENCE_SOURCE_ICONS` |
| `ui/components/sidebar.py` | 118 | Status-aware sidebar: Load Demo shortcut, Reset Demo, stage checklist on DONE |
| `ui/components/landing.py` | 220 | Two-column intake form with validation, path warnings suppressed for demo, demo indicator |
| `ui/components/stage_progress.py` | 74 | Horizontal pipeline progress strip with dot/connector/label HTML |
| `ui/components/incident.py` | 32 | Incident summary banner with amber left-border accent |
| `ui/components/investigate.py` | 213 | 4 investigator cards with WAITING/ANALYZING/COMPLETE/NO FINDINGS status badges; `render_running()` for pipeline animation |
| `ui/components/evidence.py` | 147 | Timeline grid + per-source evidence cards (observation/correlated fields) |
| `ui/components/root_cause.py` | 214 | Root cause statement + supporting evidence compact list + diagnostics 3-col grid (files, functions, commit, confidence bar) |
| `ui/components/regression_test.py` | 77 | Test metadata row + `st.code()` block |
| `ui/components/fix.py` | 300 | Centrepiece: ① ❌ BEFORE FIX → ② fix description+diff → ③ ✅ AFTER FIX → ④ INCIDENT RESOLVED, with arrow connectors and bordered panels |
| `app.py` | 189 | Entry point: `set_page_config`, pipeline runner with investigate animation, double-submit guard, error screen, impossible state guard |
| `README.md` | 186 | Full project documentation |
| `docs/architecture.md` | 152 | System architecture, data flow, state model, component map |

---

## How Bob was prompted

### Phase 1 — Schema discovery

The session began by asking Bob to inspect the existing repository without modifying anything and report on:

- What files exist
- What `schemas.py` contains
- What backend interfaces are implemented
- Integration risks

Bob read `schemas.py`, the directory structure, and the existing stub files, then produced a detailed assessment. This grounded all subsequent work in the actual schema rather than assumptions.

### Phase 2 — Architecture decisions

Key architectural questions were discussed with Bob before writing any code:

- How to handle the Streamlit `value=`/`key=` widget conflict when loading demo data
- Whether to centralise session state in a dataclass or scatter it across `session_state`
- How to prevent double-submission during a long pipeline run
- Where the single backend adapter swap point should live

Bob proposed the `_demo_pending` pattern, the `AppState` dataclass, the `_pipeline_started` guard, and the `BackendProtocol` structural type. These decisions were reviewed and accepted before implementation began.

### Phase 3 — Iterative component construction

Each component was built in sequence, with Bob:

1. Reading the relevant schema fields before writing any rendering code
2. Writing components that consumed only real schema fields (no fabrication)
3. Using `unsafe_allow_html=True` only for CSS class injection and layout — never for user content
4. Keeping all styling in `styles.py` and shared constants in one place

Schema-specific things Bob caught and handled correctly:

- `evidence.source` is `Literal["log","git","code","test"]` — not a free string
- `verification_result` is a nested `VerificationResult(passed: bool, output: str)` — not a string
- `confidence` is `float` with Pydantic `ge/le` validators — rendered as a percentage bar
- `suspicious_commit` is `Optional[str]` — only rendered when not `None`
- `RegressionTest` has `name/code/language` — no `path` or `scenario` field

### Phase 4 — Demo reliability hardening

The demo flow (`⚡ Load Demo Incident` → `▶ REPLAY INCIDENT` → full report → `🔄 Reset Demo` → repeat) was tested for repeatability, and Bob fixed:

- The widget key conflict: removed `value=` from all intake widgets; all form values flow through the `_demo_pending` drain at the top of `landing.render()`
- The nuclear reset: `_ALL_WIDGET_KEYS` list in `state.py` ensures every widget key is cleared before fresh `AppState` is written, preventing stale values after reset
- Path warnings: suppressed when `repo_path` matches `DEMO_INCIDENT["repo_path"]` so demo runs don't show spurious filesystem warnings

### Phase 5 — Fix component as centrepiece

The `fix.py` component was the most design-intensive part. Bob was asked to make the full ❌→fix→✅ sequence visible in a single connected view so a first-time demo viewer immediately understands the before/after without scrolling. Bob produced the arrow-connected panel layout with bordered sections and integrated the verdict (`INCIDENT RESOLVED`) at the bottom — gated strictly on `verification_result.passed is True`.

---

## Key Bob capabilities used

| Capability | Usage |
|---|---|
| **File inspection before writing** | Bob read `schemas.py`, existing stubs, and directory structure before producing any code |
| **Schema contract adherence** | Bob never invented schema fields; it checked field names and types before rendering |
| **Multi-file consistency** | Constants defined in one file (`STAGES`, `INTAKE_KEYS`, `_ALL_WIDGET_KEYS`) were referenced correctly across all components |
| **Pydantic v2 awareness** | Bob knew `BaseModel` field validators, `Optional` handling, and `Literal` types |
| **Streamlit-specific patterns** | Bob handled widget key conflicts, `st.session_state` discipline, `st.rerun()` timing, and `st.empty()` placeholder patterns |
| **Iterative refinement** | Bob updated individual components based on feedback without touching unrelated files |
| **Minimal change principle** | Each edit was scoped to exactly what was asked; no unrequested refactors or additions |

---

## Development workflow

```
1. Bob reads existing files (never writes before reading)
2. Discuss architecture decisions in chat
3. Bob writes one component at a time
4. Run: python -c "from ui.components.X import render" to smoke-test import
5. Run streamlit to validate visually
6. Bob fixes any issues found during manual testing
7. Repeat for next component
```

Total components written: **14 files**, **~2,200 lines of Python**, in a single Bob session.
