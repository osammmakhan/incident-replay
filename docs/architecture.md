# Architecture

## Overview

Incident Replay is a two-layer system with a strict ownership boundary between the UI presentation layer and the backend investigation pipeline.

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
│                                                             │
│  landing → [pipeline] → incident → investigate → evidence   │
│                      → root_cause → regression_test → fix   │
└────────────────────────────┬────────────────────────────────┘
                             │ ui/adapter.py  (single swap point)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend Pipeline (incident_replay/)            │
│                                                             │
│  agents/  →  analysis/  →  execution/  →  IncidentReport   │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer responsibilities

### UI layer (`ui/`, `app.py`)

Owned by the UI developer. Presents backend results — no investigation logic.

| File | Responsibility |
|---|---|
| `app.py` | Page config, routing by `RunStatus`, pipeline runner with stage animation |
| `ui/adapter.py` | Single import point for the backend callable; mock fallback |
| `ui/state.py` | `AppState` dataclass, `RunStatus` enum, `load/save/reset/load_demo` |
| `ui/mock_backend.py` | Deterministic `IncidentReport` for demo and UI development |
| `ui/styles.py` | CSS injection, `STAGES` constants, `EVIDENCE_SOURCE_ICONS` |
| `ui/components/` | One file per pipeline stage; each exposes a `render(report)` function |

### Backend layer (`incident_replay/`)

Owned by the backend developer. Implements the investigation pipeline.

| Directory | Responsibility |
|---|---|
| `agents/` | Investigation agents (log, git, code, test); synthesis agent entry point |
| `analysis/` | Analysis and correlation logic |
| `execution/` | Fix application and test execution |
| `models/schemas.py` | Shared Pydantic v2 data contract (read-only for UI) |
| `utils/` | Shared utilities |

---

## Data flow

```
User fills intake form
        │
        ▼
AppState.status = RUNNING
        │
        ▼
app.py calls ui/adapter.get_backend()
        │
        ├── INCIDENT_REPLAY_MOCK=1 ──► ui/mock_backend.replay_incident()
        │
        └── else ────────────────────► incident_replay.agents.synthesis_agent.replay_incident()
                                                │
                                                ▼
                                        IncidentReport (Pydantic model)
                                                │
                                                ▼
                                    AppState.report = report
                                    AppState.status = DONE
                                                │
                                                ▼
                                    _render_report(state) in app.py
                                    (calls each component in order)
```

---

## State model

All UI state lives in a single `AppState` dataclass stored in `st.session_state["incident_replay_state"]`. No component reads from or writes to `session_state` directly.

```
RunStatus.IDLE    → intake form shown (landing.render)
RunStatus.RUNNING → pipeline animation running (_run_pipeline)
RunStatus.DONE    → full report rendered (_render_report)
RunStatus.ERROR   → error screen shown (_render_error)
```

### Widget key discipline

Streamlit widgets that use both `value=` and `key=` conflict on re-run. The solution used here:

1. Widget keys are defined in `INTAKE_KEYS` (centralised in `state.py`)
2. `load_demo()` writes to `st.session_state["_demo_pending"]` — a dict that is drained **before** any widget is instantiated (top of `landing.render`)
3. Nuclear reset (`reset()`) clears all known keys from `_ALL_WIDGET_KEYS` before writing a fresh `AppState`

### Double-submission guard

`st.session_state["_pipeline_started"]` is set to `True` when the pipeline begins and cleared in the `finally` block. A browser refresh mid-run will read `RunStatus.RUNNING` from session state and execute the pipeline as normal — the guard prevents a second concurrent trigger from the same callback.

---

## Backend contract

The UI adapter expects one callable matching:

```python
def replay_incident(
    incident_description: str,
    repo_path: str,
    log_path: str,
    stack_trace: str,
) -> IncidentReport: ...
```

The `BackendProtocol` in `ui/adapter.py` is a `typing.Protocol` — mypy will catch signature drift at type-check time before it becomes a runtime error.

---

## Schema contract (`incident_replay/models/schemas.py`)

```
IncidentReport
├── incident_summary: str
├── timeline: list[TimelineEvent]
│   └── timestamp: str, description: str
├── root_cause: str
├── evidence: list[Evidence]
│   └── source: Literal["log","git","code","test"]
│       location: str
│       observation: str
│       relevance: str
├── affected_files: list[str]
├── affected_functions: list[str]
├── suspicious_commit: str | None
├── confidence: float (0.0–1.0)
├── regression_test: RegressionTest
│   └── name: str, code: str, language: str
├── suggested_fix: SuggestedFix
│   └── description: str, patch: str
└── verification_result: VerificationResult
    └── passed: bool, output: str
```

The UI reads every field of this schema. The UI never writes to it, infers from it beyond what is stated, or fabricates fields that are absent.

---

## Component map

Each UI component corresponds to one pipeline stage:

| Stage | Component | Data consumed |
|---|---|---|
| INCIDENT | `incident.py` | `incident_summary` |
| INVESTIGATE | `investigate.py` | `evidence[].source`, `timeline` |
| EVIDENCE | `evidence.py` | `timeline`, `evidence[]` |
| ROOT CAUSE | `root_cause.py` | `root_cause`, `evidence[]`, `affected_files`, `affected_functions`, `suspicious_commit`, `confidence` |
| REGRESSION TEST | `regression_test.py` | `regression_test.{name,code,language}` |
| FIX | `fix.py` | `suggested_fix`, `verification_result`, `regression_test.name` |
| VERIFICATION | `fix.py` (integrated) | `verification_result.{passed,output}` |

---

## Demo repeatability

The demo is designed for live presentation and screen recording:

- `INCIDENT_REPLAY_MOCK=1` produces identical output on every run
- `⚡ Load Demo Incident` pre-fills the intake form in one click
- `🔄 Reset Demo` (sidebar or error screen) performs nuclear session reset and returns to IDLE
- No fabricated PASS results — `INCIDENT RESOLVED` only appears when `verification_result.passed is True`
- Path warnings are suppressed for the demo paths (mock backend doesn't need real files)
