"""
Public backend API for the Incident Replay Streamlit UI.

Integration contract
--------------------
The UI (app.py and every module under ui/) should import exclusively from
this module.  No UI file should import directly from:

- incident_replay.orchestrator
- incident_replay.agents.*
- incident_replay.execution.*
- incident_replay.analysis.*

This keeps the UI decoupled from internal implementation details and makes
the backend easy to swap or mock.

Quick-start for the UI
----------------------
::

    from incident_replay.api import (
        replay_incident,
        get_regression_test,
        run_regression_test,
        apply_suggested_fix,
        run_regression_test_after_fix,
        get_verification_result,
        OrchestratorError,
    )
    from incident_replay.models.schemas import IncidentReport, VerificationResult

    # 1. Run the full pipeline
    report: IncidentReport = replay_incident(
        incident_description="Checkout 500 when discount is null",
        repo_path="/path/to/repo",
        log_path="/path/to/error.log",
        stack_trace="...",          # optional
    )

    # 2. Access the generated regression test
    test = get_regression_test(report)    # → RegressionTest

    # 3. Proof workflow (optional — report already contains the pipeline result)
    vr_before = run_regression_test(test, report.repo_path)
    vr_patch   = apply_suggested_fix(report, repo_path="/path/to/repo")
    vr_after   = run_regression_test_after_fix(test, repo_path="/path/to/repo")
    vr_final   = get_verification_result(report)

Exported names
--------------
replay_incident          Full pipeline → IncidentReport
OrchestratorError        Raised when both repo_path and log_path are blank
get_regression_test      Extract RegressionTest from a completed report
run_regression_test      Write test to temp file and run pytest (before fix)
apply_suggested_fix      Apply the patch from the report's suggested_fix
run_regression_test_after_fix  Re-run the test after the fix is applied
get_verification_result  Extract VerificationResult from a completed report
"""

from __future__ import annotations

import tempfile
import re
from pathlib import Path

from incident_replay.execution import patcher, test_runner
from incident_replay.models.schemas import (
    IncidentReport,
    RegressionTest,
    VerificationResult,
)
from incident_replay.orchestrator import OrchestratorError, replay_incident

__all__ = [
    "OrchestratorError",
    "replay_incident",
    "get_regression_test",
    "run_regression_test",
    "apply_suggested_fix",
    "run_regression_test_after_fix",
    "get_verification_result",
]


# ---------------------------------------------------------------------------
# Regex shared with orchestrator._field_token_from_synthesis — kept local so
# the UI never needs to import orchestrator internals.
# ---------------------------------------------------------------------------

_FIELD_QUOTED_RE = re.compile(r"field\s+['\"]([A-Za-z_]\w*)['\"]", re.IGNORECASE)
_KV_NULL_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*=\s*(?:null|none|nil)", re.IGNORECASE
)


def _field_token_from_report(report: IncidentReport) -> str | None:
    """Extract the incident field token from the report's root_cause text."""
    text = report.root_cause or ""
    m = _FIELD_QUOTED_RE.search(text)
    if m:
        return m.group(1)
    m = _KV_NULL_RE.search(text)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Simple report accessors
# ---------------------------------------------------------------------------

def get_regression_test(report: IncidentReport) -> RegressionTest:
    """
    Return the regression test embedded in *report*.

    Parameters
    ----------
    report:
        A completed :class:`IncidentReport` returned by :func:`replay_incident`.

    Returns
    -------
    RegressionTest
        Always present; may contain placeholder code when generation failed.
    """
    return report.regression_test


def get_verification_result(report: IncidentReport) -> VerificationResult:
    """
    Return the verification result embedded in *report*.

    This is the outcome of the after-fix test run that the pipeline performed
    internally.  Use :func:`run_regression_test_after_fix` to re-run the test
    independently (e.g. after the user manually applies a fix).

    Parameters
    ----------
    report:
        A completed :class:`IncidentReport` returned by :func:`replay_incident`.

    Returns
    -------
    VerificationResult
        ``passed=True`` only when the pipeline confirmed the fix resolved the
        incident.
    """
    return report.verification_result


# ---------------------------------------------------------------------------
# Proof-workflow operations
# ---------------------------------------------------------------------------

def _run_test_code(code: str, repo_path: str, phase: str) -> VerificationResult:
    """Write *code* to a temp file, run pytest, return a VerificationResult."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="ir_regression_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        result = test_runner.run_test_file(
            tmp_path,
            cwd=repo_path or None,
            phase=phase,
        )
        return test_runner.to_verification_result(result)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_regression_test(
    regression_test: RegressionTest,
    repo_path: str,
) -> VerificationResult:
    """
    Run the regression test against the *current* (unpatched) codebase.

    The test is expected to **fail** here — a passing result means either
    the bug was already fixed or the test does not target the correct
    behaviour.

    Parameters
    ----------
    regression_test:
        The :class:`RegressionTest` to execute (typically from
        ``report.regression_test`` or :func:`get_regression_test`).
    repo_path:
        Working directory for pytest — the root of the project under test.

    Returns
    -------
    VerificationResult
        ``passed=True`` when all collected tests pass.
    """
    return _run_test_code(regression_test.code, repo_path, phase="before")


def apply_suggested_fix(
    report: IncidentReport,
    repo_path: str,
) -> VerificationResult:
    """
    Apply the null-safety patch described in *report.suggested_fix*.

    Uses the ``affected_files`` and field token extracted from
    ``report.root_cause`` to locate and patch the relevant line in the
    source file.  The result indicates whether the patch was applied
    successfully — it does **not** run the test (call
    :func:`run_regression_test_after_fix` for that).

    Parameters
    ----------
    report:
        A completed :class:`IncidentReport`.
    repo_path:
        Root of the repository; used to resolve relative file paths in
        ``report.affected_files``.

    Returns
    -------
    VerificationResult
        ``passed=True`` when the patch was written to disk.
        ``output`` carries the unified diff (on success) or the failure
        reason (on failure).
    """
    affected_files = report.affected_files
    field_token = _field_token_from_report(report)

    if not affected_files:
        return VerificationResult(
            passed=False,
            output="apply_suggested_fix: no affected file in report",
        )
    if not field_token:
        return VerificationResult(
            passed=False,
            output="apply_suggested_fix: could not determine field token from root_cause",
        )
    if not repo_path or not repo_path.strip():
        return VerificationResult(
            passed=False,
            output="apply_suggested_fix: repo_path must not be empty",
        )

    patch_result = patcher.apply_null_guard(
        file_path=affected_files[0],
        field_token=field_token,
        repo_root=repo_path,
    )

    if patch_result.applied:
        return VerificationResult(
            passed=True,
            output=patch_result.diff or "Patch applied successfully.",
        )
    else:
        return VerificationResult(
            passed=False,
            output=patch_result.failure_reason or "Patch could not be applied.",
        )


def run_regression_test_after_fix(
    regression_test: RegressionTest,
    repo_path: str,
) -> VerificationResult:
    """
    Run the regression test against the *patched* codebase.

    The test is expected to **pass** here — a failing result means the fix
    is incomplete or was not applied.

    Parameters
    ----------
    regression_test:
        The same :class:`RegressionTest` used in :func:`run_regression_test`.
    repo_path:
        Working directory for pytest — the root of the project under test.

    Returns
    -------
    VerificationResult
        ``passed=True`` when all collected tests pass.
    """
    return _run_test_code(regression_test.code, repo_path, phase="after")
