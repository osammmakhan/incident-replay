"""
Safe test-execution layer for the Incident Replay workflow.

Purpose
-------
Execute the generated regression test before and after the suggested fix,
capture every observable signal from the run, and return structured
verification information.

The central rule
----------------
**PASS is only reported when pytest exits with code 0 and at least one
test was collected and run.**  Every other outcome — non-zero exit,
timeout, empty collection, import error, syntax error, interrupted run,
internal pytest error — is FAIL.  This rule is implemented once, in
:func:`_outcome`, so it cannot be bypassed.

Pytest exit codes (PEP 366 / pytest docs)
------------------------------------------
0   Tests passed (the only code that can produce PASS)
1   Tests failed / errors
2   Interrupted by user (Ctrl-C or signal)
3   Internal pytest error
4   Command-line usage error
5   No tests were collected

Isolation
---------
Tests run in a subprocess (``python -m pytest``).  This means:

* Imports are resolved in *the target project's* working directory, not
  in this package.
* A broken test module cannot crash the pipeline or the Streamlit UI.
* No pytest internals are imported here; only ``subprocess`` from the
  standard library.

Architecture
------------
``RunResult``
    Internal dataclass carrying every observable signal: exit code, raw
    stdout, raw stderr, combined output, failure reason, phase label, and
    the elapsed wall-clock time.  It is *not* part of the
    ``IncidentReport`` schema — it is richer internal state.

``VerificationResult``  (``incident_replay.models.schemas``)
    The thin schema object consumed by ``IncidentReport``.  ``passed``
    and ``output`` only.  Produced from a ``RunResult`` via
    :func:`to_verification_result`.

``run_test_file``
    Low-level executor.  Runs a single file/node-id, returns a
    ``RunResult``.

``run_before_fix`` / ``run_after_fix``
    Semantic wrappers that attach a phase label (``"before"`` /
    ``"after"``) to the ``RunResult``.  Use these in the pipeline so the
    UI can distinguish the two runs without parsing output text.

``RunComparison``
    The result of comparing a before-run with an after-run.
    ``fix_verified`` is ``True`` only when the before-run failed *and*
    the after-run passed — the expected pattern for a valid fix.

``compare``
    Builds a ``RunComparison`` from two ``RunResult`` values.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from incident_replay.models.schemas import VerificationResult

__all__ = [
    "RunnerError",
    "RunResult",
    "RunComparison",
    "run_test_file",
    "run_before_fix",
    "run_after_fix",
    "compare",
    "to_verification_result",
]

# ---------------------------------------------------------------------------
# Pytest exit-code constants (documented in pytest docs and _pytest/config/__init__.py)
# ---------------------------------------------------------------------------

_EXIT_OK = 0           # all tests passed
_EXIT_TESTSFAILED = 1  # some tests failed or had errors
_EXIT_INTERRUPTED = 2  # interrupted (Ctrl-C)
_EXIT_INTERNALERROR = 3
_EXIT_USAGEERROR = 4
_EXIT_NOTESTSCOLLECTED = 5  # no tests collected — NOT a pass


# Human-readable label for each non-zero code.
_EXIT_LABELS: dict[int, str] = {
    _EXIT_TESTSFAILED:      "test failure",
    _EXIT_INTERRUPTED:      "run interrupted",
    _EXIT_INTERNALERROR:    "pytest internal error",
    _EXIT_USAGEERROR:       "pytest usage error",
    _EXIT_NOTESTSCOLLECTED: "no tests collected",
}


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class RunnerError(RuntimeError):
    """
    Raised when the subprocess itself cannot be started.

    This is *not* raised for test failures, collection errors, or any
    outcome that pytest itself reports — those are captured in
    ``RunResult.passed = False``.  This error indicates an infrastructure
    problem (e.g. the Python interpreter binary is missing).
    """


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Every observable signal from a single pytest run.

    Attributes
    ----------
    passed:
        ``True`` only when pytest exited with code 0 and collected at
        least one test.  *Never* ``True`` merely because execution
        completed without an OS-level error.
    exit_code:
        The integer return code from the subprocess, or ``-1`` when the
        process was killed by a timeout.
    stdout:
        Raw standard output from pytest.
    stderr:
        Raw standard error from pytest.
    output:
        ``stdout + stderr`` stripped — ready for display.
    failure_reason:
        Short human-readable label explaining *why* this run failed,
        e.g. ``"test failure"``, ``"no tests collected"``,
        ``"timeout after 30s"``.  Empty string when ``passed`` is
        ``True``.
    phase:
        Label set by the caller — ``"before"`` or ``"after"`` — so
        consumers can distinguish the two runs without parsing output.
    elapsed_seconds:
        Wall-clock time the subprocess took, in seconds.
    """

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    output: str
    failure_reason: str
    phase: str
    elapsed_seconds: float


@dataclass
class RunComparison:
    """
    Side-by-side record of the before-fix and after-fix runs.

    Attributes
    ----------
    before:
        Run result from executing the regression test against the
        *unpatched* code.
    after:
        Run result from executing the regression test against the
        *patched* code.
    fix_verified:
        ``True`` only when ``before.passed is False`` and
        ``after.passed is True``.  This is the expected pattern for a
        valid fix: the test was a sentinel (failed before) and the fix
        resolved it (passes after).
    summary:
        One-sentence plain-English description of the comparison.
    """

    before: RunResult
    after: RunResult
    fix_verified: bool
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outcome(returncode: int, stdout: str, timeout: float | None) -> tuple[bool, str]:
    """
    Determine ``(passed, failure_reason)`` from *returncode* and captured
    output.

    This is the single place where PASS/FAIL is decided.

    Rules (in order):
    1. Timeout → FAIL, reason ``"timeout after {n}s"``.
    2. Exit code 5 → FAIL, reason ``"no tests collected"`` — the test
       file existed and was valid Python but pytest found nothing to run.
       This must never be reported as PASS.
    3. Exit code 0 → check the summary line for at least one
       ``"N passed"`` token.  If absent, FAIL with reason
       ``"exit 0 but no tests reported as passed"``.
    4. Any other exit code → FAIL, reason from ``_EXIT_LABELS`` or
       ``"exit code {n}"``.
    """
    if timeout is not None:
        return False, f"timeout after {timeout:.0f}s"

    if returncode == _EXIT_NOTESTSCOLLECTED:
        return False, "no tests collected"

    if returncode == _EXIT_OK:
        # Require at least one "N passed" token in the output.
        # pytest -q always prints "N passed" (or "N passed, M warning(s)")
        # when at least one test ran successfully.
        if " passed" in stdout:
            return True, ""
        return False, "exit 0 but no tests reported as passed"

    label = _EXIT_LABELS.get(returncode, f"exit code {returncode}")
    return False, label


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------

def run_test_file(
    test_path: str | Path,
    *,
    cwd: str | Path | None = None,
    timeout: float = 60.0,
    extra_args: list[str] | None = None,
    phase: str = "",
) -> RunResult:
    """
    Run *test_path* with ``python -m pytest`` and return a :class:`RunResult`.

    Parameters
    ----------
    test_path:
        Path to the ``.py`` test file, or a pytest node-id string such
        as ``"tests/test_foo.py::TestClass::test_method"``.  Resolved
        relative to *cwd* (or the process's cwd) when not absolute.
    cwd:
        Working directory for the subprocess.  ``None`` means the
        process's current directory.
    timeout:
        Seconds before the subprocess is killed.  The run is then marked
        FAIL with reason ``"timeout after {n}s"``.  Defaults to 60 s.
    extra_args:
        Additional pytest arguments (e.g. ``["-v", "--tb=long"]``).
        ``--tb=short`` and ``-q`` are always included.
    phase:
        Caller-supplied label stored in ``RunResult.phase`` — typically
        ``"before"`` or ``"after"``.  Does not affect execution.

    Returns
    -------
    RunResult
        ``passed`` reflects whether all collected tests passed.  The
        ``exit_code``, ``stdout``, ``stderr``, ``output``,
        ``failure_reason``, and ``elapsed_seconds`` fields carry the full
        picture.

    Raises
    ------
    RunnerError
        Only when the subprocess cannot be launched (e.g. ``python``
        binary missing or ``test_path`` is a directory, not a file).
        Test failures and collection errors are *not* raised — they are
        captured in ``RunResult.passed = False``.
    """
    # Resolve to an absolute path so pytest can find the file regardless of
    # the subprocess CWD (which may differ from the process CWD).
    p = Path(test_path).resolve()
    path_str = str(p)

    # Validate that the path exists and is a file before spinning up a
    # subprocess — gives a cleaner error than pytest's collection traceback.
    if not p.exists():
        return RunResult(
            passed=False,
            exit_code=-1,
            stdout="",
            stderr="",
            output=f"Test file not found: {path_str}",
            failure_reason="test file not found",
            phase=phase,
            elapsed_seconds=0.0,
        )
    if p.is_dir():
        return RunResult(
            passed=False,
            exit_code=-1,
            stdout="",
            stderr="",
            output=f"Path is a directory, not a test file: {path_str}",
            failure_reason="path is a directory",
            phase=phase,
            elapsed_seconds=0.0,
        )

    cmd = [
        sys.executable, "-m", "pytest",
        path_str,
        "--tb=short",
        "-q",
        *(extra_args or []),
    ]
    cwd_str = str(cwd) if cwd is not None else None
    timed_out: float | None = None
    t_start = time.monotonic()

    # Pass PYTHONIOENCODING so pytest writes UTF-8 regardless of the
    # platform default (cp1252 on Windows), preventing UnicodeDecodeError
    # in the subprocess stdout reader thread.
    import os as _os
    env = _os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd_str,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RunnerError(
            f"Could not launch pytest — Python interpreter not found: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        # Capture whatever was written before the timeout.
        raw_stdout = (exc.stdout or b"")
        raw_stderr = (exc.stderr or b"")
        if isinstance(raw_stdout, bytes):
            raw_stdout = raw_stdout.decode("utf-8", errors="replace")
        if isinstance(raw_stderr, bytes):
            raw_stderr = raw_stderr.decode("utf-8", errors="replace")
        elapsed = time.monotonic() - t_start
        timed_out = timeout
        combined = (raw_stdout + raw_stderr).strip()
        return RunResult(
            passed=False,
            exit_code=-1,
            stdout=raw_stdout,
            stderr=raw_stderr,
            output=combined or f"pytest timed out after {timeout:.0f}s.",
            failure_reason=f"timeout after {timeout:.0f}s",
            phase=phase,
            elapsed_seconds=elapsed,
        )

    elapsed = time.monotonic() - t_start
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = (stdout + stderr).strip()

    passed, failure_reason = _outcome(proc.returncode, combined, timed_out)

    return RunResult(
        passed=passed,
        exit_code=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        output=combined,
        failure_reason=failure_reason,
        phase=phase,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Semantic wrappers
# ---------------------------------------------------------------------------

def run_before_fix(
    test_path: str | Path,
    *,
    cwd: str | Path | None = None,
    timeout: float = 60.0,
    extra_args: list[str] | None = None,
) -> RunResult:
    """
    Run the regression test against the *unpatched* code.

    Identical to :func:`run_test_file` with ``phase="before"``.
    The test is *expected* to fail here — a passing result means either
    the bug was already fixed or the test does not target the right
    behaviour.
    """
    return run_test_file(
        test_path, cwd=cwd, timeout=timeout, extra_args=extra_args,
        phase="before",
    )


def run_after_fix(
    test_path: str | Path,
    *,
    cwd: str | Path | None = None,
    timeout: float = 60.0,
    extra_args: list[str] | None = None,
) -> RunResult:
    """
    Run the regression test against the *patched* code.

    Identical to :func:`run_test_file` with ``phase="after"``.
    The test is *expected* to pass here — a failing result means the fix
    is incomplete or the patch has not been applied.
    """
    return run_test_file(
        test_path, cwd=cwd, timeout=timeout, extra_args=extra_args,
        phase="after",
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(before: RunResult, after: RunResult) -> RunComparison:
    """
    Build a :class:`RunComparison` from *before* and *after* run results.

    ``fix_verified`` is ``True`` only when the before-run failed *and*
    the after-run passed — the expected pattern for a valid fix.

    Parameters
    ----------
    before:
        Result from running the regression test before applying the fix.
    after:
        Result from running the regression test after applying the fix.

    Returns
    -------
    RunComparison
    """
    fix_verified = (not before.passed) and after.passed

    if fix_verified:
        summary = (
            "Fix verified: the regression test failed before the fix "
            "and passed after."
        )
    elif before.passed and after.passed:
        summary = (
            "Test passed both before and after the fix — "
            "the test may not target the buggy behaviour."
        )
    elif not before.passed and not after.passed:
        summary = (
            f"Fix not yet effective: test still fails after the fix "
            f"({after.failure_reason or 'test failure'})."
        )
    else:
        # before passed, after failed — regression introduced by the fix
        summary = (
            "Fix introduced a regression: test passed before but "
            f"fails after ({after.failure_reason or 'test failure'})."
        )

    return RunComparison(
        before=before,
        after=after,
        fix_verified=fix_verified,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Schema bridge
# ---------------------------------------------------------------------------

def to_verification_result(run: RunResult) -> VerificationResult:
    """
    Convert a :class:`RunResult` to the thin
    :class:`~incident_replay.models.schemas.VerificationResult` schema
    object consumed by :class:`~incident_replay.models.schemas.IncidentReport`.

    The ``output`` field carries both the pytest output and, when the run
    failed, the ``failure_reason`` prefixed in brackets so the UI can
    display a clear signal without parsing raw pytest output.
    """
    if run.passed:
        display = run.output
    else:
        reason = run.failure_reason or "unknown failure"
        prefix = f"[{reason}] "
        display = prefix + run.output if run.output else prefix.rstrip()

    return VerificationResult(passed=run.passed, output=display)
