# UI Integration Guide

This document describes how the Streamlit UI connects to the backend.
It is the authoritative reference for every integration point between
`app.py` / `ui/` and `incident_replay/`.

---

## Golden rule

> **The UI imports only from `incident_replay.api` (or the top-level
> `incident_replay` package which re-exports the same names).**

No UI file should reach into `incident_replay.orchestrator`,
`incident_replay.agents.*`, `incident_replay.execution.*`, or
`incident_replay.analysis.*`.  Those are internal implementation details.

---

## Swap point — `ui/adapter.py`

`ui/adapter.py` is the single point where the real backend is swapped in
or out.  It exports one function:

```python
from ui.adapter import get_backend

backend = get_backend()   # → BackendProtocol (callable)
report  = backend(
    incident_description="...",
    repo_path="...",
    log_path="...",
    stack_trace="...",
)
```

Selection order:

| Priority | Condition | Backend used |
|----------|-----------|--------------|
| 1 | `INCIDENT_REPLAY_MOCK=1` in environment | `ui.mock_backend.replay_incident` |
| 2 | `incident_replay.api.replay_incident` importable | Real backend |
| 3 | Fallback | `ui.mock_backend.replay_incident` |

---

## Integration point 1 — full pipeline

```python
from incident_replay.api import replay_incident, OrchestratorError
from incident_replay.models.schemas import IncidentReport
```

### Signature

```python
def replay_incident(
    incident_description: str,
    repo_path: str,
    log_path: str,
    stack_trace: str | None = None,
) -> IncidentReport:
```

### Parameters

| Name | Type | Notes |
|------|------|-------|
| `incident_description` | `str` | Free-text description; stored in the report |
| `repo_path` | `str` | Absolute or relative path to the git repo root |
| `log_path` | `str` | Path to the application log file |
| `stack_trace` | `str \| None` | Raw stack trace text; `None` to extract from log |

### Returns

`IncidentReport` — always returned, even when investigators fail.
Failures are embedded in the report as low-confidence findings.

### Raises

`OrchestratorError` — only when **both** `repo_path` and `log_path` are
blank.  All other errors are caught and embedded in the report.

---

## Integration point 2 — proof-workflow helpers

These five functions drive the step-by-step proof that the fix works.
All inputs come from the `IncidentReport` already in hand; all outputs
are `VerificationResult` objects from `incident_replay.models.schemas`.

```python
from incident_replay.api import (
    get_regression_test,
    run_regression_test,
    apply_suggested_fix,
    run_regression_test_after_fix,
    get_verification_result,
)
```

### `get_regression_test(report) → RegressionTest`

Extracts `report.regression_test`.  A stable accessor so the UI never
references `IncidentReport` field names directly.

### `get_verification_result(report) → VerificationResult`

Extracts `report.verification_result` — the outcome of the after-fix
test run that the pipeline performed internally.

### `run_regression_test(regression_test, repo_path) → VerificationResult`

Writes `regression_test.code` to a temporary file, runs it with
`python -m pytest`, deletes the temp file, and returns the result.

Expected outcome: **FAIL** (the bug is still present).

```python
vr = run_regression_test(report.regression_test, repo_path="/path/to/repo")
# vr.passed  → False  (expected before fix)
# vr.output  → raw pytest output
```

### `apply_suggested_fix(report, repo_path) → VerificationResult`

Applies the null-safety guard described in `report.suggested_fix` by
patching the first file listed in `report.affected_files`.

Returns `passed=True` when the file was actually modified.

```python
vr = apply_suggested_fix(report, repo_path="/path/to/repo")
# vr.passed  → True   (patch written)
# vr.output  → unified diff
```

### `run_regression_test_after_fix(regression_test, repo_path) → VerificationResult`

Identical to `run_regression_test` but labelled `phase="after"`.
Call this **after** `apply_suggested_fix`.

Expected outcome: **PASS** (the fix resolves the defect).

```python
vr = run_regression_test_after_fix(report.regression_test, repo_path="/path/to/repo")
# vr.passed  → True   (expected after fix)
# vr.output  → raw pytest output
```

---

## Schema reference

All types live in `incident_replay.models.schemas`.

```
IncidentReport
├── incident_summary:    str
├── timeline:            list[TimelineEvent]
│     ├── timestamp:     str
│     ├── event:         str
│     └── description:   str
├── root_cause:          str
├── evidence:            list[Evidence]
│     ├── source:        "log" | "git" | "code" | "test"
│     ├── location:      str
│     ├── observation:   str
│     └── relevance:     str
├── affected_files:      list[str]
├── affected_functions:  list[str]
├── suspicious_commit:   str | None
├── confidence:          float  (0.0 – 1.0)
├── regression_test:     RegressionTest
│     ├── name:          str
│     ├── code:          str
│     └── language:      str  (default "python")
├── suggested_fix:       SuggestedFix
│     ├── description:   str
│     └── patch:         str  (unified diff, may be empty)
└── verification_result: VerificationResult
      ├── passed:        bool
      └── output:        str
```

---

## Proof-workflow sequence

```
UI                          incident_replay.api              execution layer
 │                                │                                │
 │── replay_incident(...) ────────►│                               │
 │                                 │── orchestrator.replay_incident()
 │                                 │   (runs all investigators,    │
 │                                 │    generates test, patches,   │
 │                                 │    verifies internally)       │
 │◄── IncidentReport ──────────────│                               │
 │                                 │                               │
 │  [display report; offer         │                               │
 │   proof-workflow buttons]       │                               │
 │                                 │                               │
 │── get_regression_test(report) ──►│                               │
 │◄── RegressionTest ──────────────│                               │
 │                                 │                               │
 │── run_regression_test(...) ─────►│── write temp .py, run pytest ►│
 │◄── VerificationResult(FAIL) ────│◄──────────────────────────────│
 │                                 │                               │
 │── apply_suggested_fix(...) ─────►│── patcher.apply_null_guard() ►│
 │◄── VerificationResult(patch ok)─│◄──────────────────────────────│
 │                                 │                               │
 │── run_regression_test_after_fix ►│── write temp .py, run pytest ►│
 │◄── VerificationResult(PASS) ────│◄──────────────────────────────│
 │                                 │                               │
 │── get_verification_result(report)►│                              │
 │◄── VerificationResult(pipeline) │                               │
```

---

## What the UI must never import

| Forbidden import | Reason |
|-----------------|--------|
| `incident_replay.orchestrator` | Implementation detail; use `incident_replay.api.replay_incident` |
| `incident_replay.agents.*` | Internal; changes freely between phases |
| `incident_replay.execution.*` | Internal; use the api.py wrappers |
| `incident_replay.analysis.*` | Internal |

---

## Running with the mock backend

```bash
INCIDENT_REPLAY_MOCK=1 streamlit run app.py
```

The mock returns a hardcoded `IncidentReport` from `ui/mock_backend.py`.
It satisfies the same `IncidentReport` schema so every UI component
renders identically regardless of which backend is active.
