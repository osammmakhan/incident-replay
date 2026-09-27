# 🔁 Incident Replay

> **AI-assisted developer workflow that turns a production incident into an evidence-backed root cause, regression test, and verified fix — in a single automated pipeline.**

---

## What it does

Incident Replay takes a real production incident (description, logs, stack trace, repo) and runs it through a deterministic seven-stage pipeline:

```
INCIDENT → INVESTIGATE → EVIDENCE → ROOT CAUSE → REGRESSION TEST → FIX → VERIFICATION
```

At the end you have:

| Output | Description |
|---|---|
| **Root cause statement** | Precise, evidence-backed explanation of what failed and why |
| **Incident timeline** | Timestamped sequence from deploy to alert to resolution |
| **Evidence corpus** | Log entries, git commits, code locations, and test gaps — each correlated to the root cause |
| **Regression test** | A test that reproduces the exact production scenario |
| **Suggested fix** | A concrete code change with a unified diff |
| **Verification** | The regression test run against the patched code — PASS/FAIL with full runner output |

---

## Demo

### Run with mock backend (no dependencies needed)

```powershell
# Windows PowerShell
$env:INCIDENT_REPLAY_MOCK = "1"
.venv\Scripts\streamlit run app.py
```

```bash
# macOS / Linux
INCIDENT_REPLAY_MOCK=1 streamlit run app.py
```

Then click **⚡ Load Demo Incident** in the UI to pre-fill the form with the checkout/null-discount scenario and click **▶ REPLAY INCIDENT**.

### Run with real backend

```bash
streamlit run app.py
```

The adapter automatically detects whether `incident_replay.agents.synthesis_agent.replay_incident` is importable and falls back to the mock if it is not.

---

## Demo scenario

The built-in demo incident is a real-world-style checkout service failure:

> **"Checkout returns HTTP 500 when discount is null."**

The pipeline traces it from a `TypeError` in `apply_discount()` all the way to the git commit that introduced it (`d4e8f21`), generates a null-guard regression test, applies the fix, and verifies the test passes.

---

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### requirements

```
streamlit>=1.35
pydantic>=2.0
```

---

## Project structure

```
incident_replay/
├── agents/          ← Backend: investigation agents (teammate-owned)
├── analysis/        ← Backend: analysis layer (teammate-owned)
├── execution/       ← Backend: fix execution (teammate-owned)
├── models/
│   └── schemas.py   ← Shared contract (read-only for UI)
└── utils/           ← Backend utilities (teammate-owned)

ui/
├── adapter.py       ← Single swap point: real backend vs mock
├── mock_backend.py  ← Deterministic demo data (checkout/null-discount)
├── state.py         ← AppState dataclass, RunStatus enum, session helpers
├── styles.py        ← CSS injection, STAGES, EVIDENCE_SOURCE_ICONS
└── components/
    ├── sidebar.py         ← Status-aware sidebar with Load Demo / Reset Demo
    ├── landing.py         ← Incident intake form
    ├── stage_progress.py  ← Horizontal pipeline progress strip
    ├── incident.py        ← Incident summary banner
    ├── investigate.py     ← 4 investigator agent cards
    ├── evidence.py        ← Timeline + evidence cards
    ├── root_cause.py      ← Root cause statement + diagnostics grid
    ├── regression_test.py ← Generated test display
    └── fix.py             ← ❌ before → fix diff → ✅ after → INCIDENT RESOLVED

app.py               ← Entry point: pipeline runner, page routing
demo_project/        ← Backend demo target (teammate-owned)
tests/               ← Backend tests (teammate-owned)
docs/
├── architecture.md  ← System architecture
└── bob-usage.md     ← How IBM Bob was used in development
```

---

## Backend contract

The UI calls exactly one backend function:

```python
replay_incident(
    incident_description: str,
    repo_path: str,
    log_path: str,
    stack_trace: str,
) -> IncidentReport
```

`IncidentReport` schema (defined in `incident_replay/models/schemas.py`):

```python
class IncidentReport(BaseModel):
    incident_summary:    str
    timeline:            list[TimelineEvent]      # timestamp + description
    root_cause:          str
    evidence:            list[Evidence]           # source/location/observation/relevance
    affected_files:      list[str]
    affected_functions:  list[str]
    suspicious_commit:   str | None
    confidence:          float                    # 0.0–1.0
    regression_test:     RegressionTest           # name/code/language
    suggested_fix:       SuggestedFix             # description/patch
    verification_result: VerificationResult       # passed/output
```

To swap in the real backend, implement `replay_incident()` in `incident_replay/agents/synthesis_agent.py` matching this signature. The adapter in `ui/adapter.py` will pick it up automatically.

---

## Environment variables

| Variable | Values | Effect |
|---|---|---|
| `INCIDENT_REPLAY_MOCK` | `1` | Force mock backend regardless of real backend availability |

Copy `.env.example` to `.env` for local development.

---

## Architecture decisions

- **Single adapter swap point** — `ui/adapter.py` is the only place that imports the real backend. All other UI code is backend-agnostic.
- **No investigation logic in the UI** — the UI only presents what `IncidentReport` contains. No re-analysis, no fabrication.
- **PASS/INCIDENT RESOLVED gated on schema** — the green verdict is shown only when `verification_result.passed is True`. The UI never assumes it.
- **Deterministic demo** — `INCIDENT_REPLAY_MOCK=1` produces identical output on every run. Suitable for live demos and screen recordings.
- **Nuclear reset** — the Reset Demo button clears all session state (widget keys, pipeline guard, pending values) and returns to a clean IDLE state. Safe to repeat as many times as needed.

---

## Team

| Area | Owner |
|---|---|
| Backend agents, analysis, execution, schemas, utils | Teammate |
| UI, Streamlit workflow, demo controller, docs | UI developer |
