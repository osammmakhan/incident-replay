"""
Phase 16 — Complete backend-only validation of the Incident Replay pipeline.

Exercises all 16 required steps against the self-contained demo project at
``tests/demo_project/`` without using the Streamlit UI.

Steps validated
---------------
 1. Incident input is accepted.
 2. Logs are investigated.
 3. Git history is investigated.
 4. Source code is investigated.
 5. Existing tests are investigated.
 6. Timeline is constructed.
 7. Evidence is correlated.
 8. Root cause is produced.
 9. Suspicious commit is identified.
10. Regression test is generated.
11. The regression test FAILS before the fix.
12. The suggested fix is produced.
13. The fix is applied.
14. The SAME regression test is run again.
15. The regression test PASSES after the fix.
16. VerificationResult accurately reflects the actual result.

Execution order
---------------
Phase A  Pre-flight: restore buggy code, clean previous artefacts.
Phase B  Investigation: run replay_incident (generates test, patches, verifies).
Phase C  Before-fix check: restore buggy code, run the same test, confirm FAIL.
Phase D  Re-apply fix: re-patch the source.
Phase E  After-fix check: run the test again, confirm PASS.

The before/after checks (C-E) are done explicitly by this script using the
test_runner and patcher APIs so every assertion is independently observable.

Usage
-----
    python validate_pipeline.py

Exit codes: 0 = all passed, 1 = one or more failed, 2 = fatal pre-flight error.

Design constraints
------------------
* No Streamlit code.
* No mocking — every step is a real operation on real files.
* No faking — each assertion checks real output.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# Force UTF-8 on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DEMO = ROOT / "tests" / "demo_project"
DEMO_REPO = DEMO
DEMO_LOG = DEMO / "logs" / "production.log"
DEMO_STACK = DEMO / "incident" / "stacktrace.txt"
DEMO_CHECKOUT = DEMO / "app" / "checkout.py"
REGRESSION_DIR = DEMO / "tests" / "regression"

# The canonical buggy source line we expect the patcher to replace.
BUGGY_LINE = "        discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))\n"

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------


def _ok(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def _fail(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def _hdr(text: str) -> str:
    return f"\033[1;34m{text}\033[0m"


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

_failures: list[str] = []


def step(number: int, description: str, passed: bool, detail: str = "") -> bool:
    label = f"Step {number:2d}: {description}"
    if passed:
        print(f"  {_ok('PASS')}  {label}")
    else:
        print(f"  {_fail('FAIL')}  {label}")
        if detail:
            for line in textwrap.wrap(detail, width=76, initial_indent="        "):
                print(line)
        _failures.append(f"Step {number}: {description}")
    return passed


# ---------------------------------------------------------------------------
# Pre-flight helpers
# ---------------------------------------------------------------------------


def restore_buggy_checkout() -> bool:
    """
    Restore checkout.py to its known-buggy state.

    Returns True when the buggy line is confirmed present afterward.
    """
    current = DEMO_CHECKOUT.read_text(encoding="utf-8")
    if BUGGY_LINE in current:
        return True  # already buggy

    print("  [restore] Reverting checkout.py to buggy state …")
    result = subprocess.run(
        ["git", "checkout", "HEAD", "--", "app/checkout.py"],
        cwd=str(DEMO_REPO),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        after = DEMO_CHECKOUT.read_text(encoding="utf-8")
        return BUGGY_LINE in after

    # git failed — rewrite manually
    print("  [restore] git checkout failed; rewriting manually …")
    DEMO_CHECKOUT.write_text(
        '"""\nDemo checkout module with a deliberate null-safety bug.\n"""\n\n\n'
        "class CheckoutService:\n"
        "    MAX_DISCOUNT_RATE = 0.5\n\n"
        "    def calculate_discount(self, order):\n"
        '        """Return the capped discount rate for *order*.\n\n'
        "        BUG: order.discount may be None.\n"
        '        """\n'
        + BUGGY_LINE
        + "        return discount_rate\n\n"
        "    def total(self, order):\n"
        '        """Return the order total after applying the discount."""\n'
        "        rate = self.calculate_discount(order)\n"
        "        return order.subtotal * (1.0 - rate)\n",
        encoding="utf-8",
    )
    after = DEMO_CHECKOUT.read_text(encoding="utf-8")
    return BUGGY_LINE in after


def clean_regression_dir() -> None:
    if REGRESSION_DIR.exists():
        shutil.rmtree(REGRESSION_DIR)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def main() -> int:
    print()
    print(_hdr("=" * 60))
    print(_hdr("  Incident Replay — Phase 16 Pipeline Validation"))
    print(_hdr("=" * 60))
    print()

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------
    print(_hdr("Pre-flight checks"))

    if not DEMO_LOG.is_file():
        print(_fail(f"  ABORT: demo log not found: {DEMO_LOG}"))
        return 2
    if not DEMO_STACK.is_file():
        print(_fail(f"  ABORT: demo stack trace not found: {DEMO_STACK}"))
        return 2
    if not (DEMO_REPO / ".git").is_dir():
        print(_fail(f"  ABORT: demo project is not a git repo: {DEMO_REPO}"))
        return 2

    if not restore_buggy_checkout():
        print(_fail(
            f"  ABORT: could not restore buggy line into {DEMO_CHECKOUT}."
        ))
        return 2

    clean_regression_dir()

    print(f"  demo project  : {DEMO_REPO}")
    print(f"  log file      : {DEMO_LOG}")
    print(f"  stack trace   : {DEMO_STACK}")
    print(f"  checkout.py   : buggy line confirmed present")
    print()

    # ------------------------------------------------------------------
    # Import backend modules (no Streamlit anywhere)
    # ------------------------------------------------------------------
    sys.path.insert(0, str(ROOT))
    from incident_replay.execution import patcher as _patcher
    from incident_replay.execution import test_runner as _runner
    from incident_replay.models.schemas import IncidentReport
    from incident_replay.orchestrator import replay_incident

    # ------------------------------------------------------------------
    # Phase B: Run replay_incident (all 9 pipeline stages)
    # ------------------------------------------------------------------
    print(_hdr("Phase B — Running replay_incident (all stages) …"))
    stack_text = DEMO_STACK.read_text(encoding="utf-8")
    try:
        report: IncidentReport = replay_incident(
            incident_description=(
                "POST /checkout returns HTTP 500 when the discount field is null."
            ),
            repo_path=str(DEMO_REPO),
            log_path=str(DEMO_LOG),
            stack_trace=stack_text,
        )
        accepted = True
    except Exception as exc:
        accepted = False
        report = None  # type: ignore[assignment]
        print(f"  replay_incident raised: {exc}")

    print()
    print(_hdr("Steps 1-10 and 12: checking report fields"))

    # Step 1
    step(1, "Incident input accepted (replay_incident returned without raising)",
         accepted,
         "replay_incident raised an exception — see above.")

    if not accepted:
        print(_fail("  Cannot continue: replay_incident failed entirely."))
        return 1

    # Step 2
    has_log_ev = any(e.source == "log" for e in report.evidence)
    step(2, "Logs investigated (log-source evidence present)",
         has_log_ev,
         f"sources: {sorted({e.source for e in report.evidence})}")

    # Step 3
    has_git_signal = (
        any(e.source == "git" for e in report.evidence)
        or report.suspicious_commit is not None
    )
    step(3, "Git history investigated (git evidence or suspicious commit)",
         has_git_signal,
         f"suspicious_commit={report.suspicious_commit!r}, "
         f"git-evidence count={sum(1 for e in report.evidence if e.source=='git')}")

    # Step 4
    has_code = (
        any(e.source == "code" for e in report.evidence)
        or bool(report.affected_files)
        or bool(report.affected_functions)
    )
    step(4, "Source code investigated (code evidence or affected files/funcs)",
         has_code,
         f"affected_files={report.affected_files}, funcs={report.affected_functions}")

    # Step 5
    has_test = any(e.source == "test" for e in report.evidence)
    step(5, "Existing tests investigated (test-source evidence present)",
         has_test,
         f"sources present: {sorted({e.source for e in report.evidence})}")

    # Step 6
    step(6, "Timeline constructed (>=1 event, all timestamps non-empty)",
         len(report.timeline) >= 1 and all(e.timestamp for e in report.timeline),
         f"timeline: {[(e.timestamp, e.event) for e in report.timeline]}")

    # Step 7
    distinct_sources = {e.source for e in report.evidence}
    step(7, f"Evidence correlated from >=2 distinct sources: {sorted(distinct_sources)}",
         len(distinct_sources) >= 2,
         f"need >=2; got {len(distinct_sources)}: {sorted(distinct_sources)}")

    # Step 8
    rc_ok = bool(report.root_cause) and len(report.root_cause) > 10
    step(8, "Root cause produced",
         rc_ok,
         f"root_cause={report.root_cause!r}")
    if rc_ok:
        print(f"         -> {report.root_cause[:120]}")

    # Step 9
    step(9, "Suspicious commit identified",
         report.suspicious_commit is not None,
         f"suspicious_commit={report.suspicious_commit!r}")
    if report.suspicious_commit:
        print(f"         -> commit {report.suspicious_commit}")

    # Step 10
    test_code_ok = (
        bool(report.regression_test.name)
        and bool(report.regression_test.code)
        and "unavailable" not in report.regression_test.code.lower()
        and "could not be generated" not in report.regression_test.code.lower()
    )
    step(10, f"Regression test generated: {report.regression_test.name!r}",
         test_code_ok,
         f"code[:100]={report.regression_test.code[:100]!r}")

    # Find the test file on disk
    test_file: Path | None = None
    if REGRESSION_DIR.exists():
        candidates = sorted(REGRESSION_DIR.glob("test_regression_*.py"))
        if candidates:
            test_file = candidates[0]

    syntax_ok = False
    if test_file and test_file.exists():
        try:
            ast.parse(test_file.read_text(encoding="utf-8"))
            syntax_ok = True
        except SyntaxError:
            pass

    step(10, "Regression test file on disk with valid Python syntax",
         syntax_ok,
         f"REGRESSION_DIR={REGRESSION_DIR}, files={list(REGRESSION_DIR.glob('*.py')) if REGRESSION_DIR.exists() else []}")

    # Step 12
    fix_ok = (
        bool(report.suggested_fix.description)
        and "unavailable" not in report.suggested_fix.description.lower()
    )
    step(12, "Suggested fix produced",
         fix_ok,
         f"description={report.suggested_fix.description!r}")
    if fix_ok:
        print(f"         -> {report.suggested_fix.description[:120]}")

    if not test_file:
        step(11, "Regression test FAILS before the fix",
             False,
             "No regression test file was generated; cannot run before-fix check.")
        step(13, "Fix applied to source file on disk",
             False, "Skipped (no test file).")
        step(14, "Regression test re-run after the fix",
             False, "Skipped.")
        step(15, "Regression test PASSES after the fix",
             False, "Skipped.")
        step(16, "VerificationResult accurately reflects after-fix run",
             False, "Skipped.")
        return 1

    # ------------------------------------------------------------------
    # Phase C: before-fix check
    # Restore the buggy code so we can confirm the test fails against it.
    # ------------------------------------------------------------------
    print()
    print(_hdr("Phase C — Before-fix check (restoring buggy code)"))
    restored = restore_buggy_checkout()
    if not restored:
        step(11, "Regression test FAILS before the fix",
             False,
             "Could not restore buggy checkout.py — cannot confirm before-fix failure.")
    else:
        before = _runner.run_before_fix(
            test_file,
            cwd=str(DEMO_REPO),
            timeout=30.0,
        )
        step(11, "Regression test FAILS before the fix (confirmed by test_runner)",
             not before.passed,
             f"before-fix: passed={before.passed}, reason={before.failure_reason!r}\n"
             f"output[:200]={before.output[:200]!r}")
        if not before.passed:
            print(f"         -> before-fix: FAILED ({before.failure_reason})")
        else:
            print(f"         -> WARNING: test passed on buggy code — output: {before.output[:80]}")

    # ------------------------------------------------------------------
    # Phase D: Re-apply the null-safety fix
    # ------------------------------------------------------------------
    print()
    print(_hdr("Phase D — Applying the fix"))
    field_token = "discount"
    patch_result = _patcher.apply_null_guard(
        file_path="app/checkout.py",
        field_token=field_token,
        repo_root=str(DEMO_REPO),
    )
    patch_applied = patch_result.applied
    step(13, "Fix applied to source file on disk",
         patch_applied,
         f"patcher result: applied={patch_result.applied}, "
         f"reason={patch_result.failure_reason!r}")
    if patch_applied:
        for i, ln in enumerate(
            DEMO_CHECKOUT.read_text(encoding="utf-8").splitlines(), 1
        ):
            if "discount" in ln and "or 0.0" in ln:
                print(f"         -> patched line {i}: {ln.strip()}")

    # ------------------------------------------------------------------
    # Phase E: After-fix check (run the SAME test file)
    # ------------------------------------------------------------------
    print()
    print(_hdr("Phase E — After-fix check (same test file)"))
    after = _runner.run_after_fix(
        test_file,
        cwd=str(DEMO_REPO),
        timeout=30.0,
    )
    step(14, "Regression test re-run after the fix (same test file)",
         True,  # the run itself completed regardless of result
         "")
    step(15, "Regression test PASSES after the fix",
         after.passed,
         f"after-fix: passed={after.passed}, reason={after.failure_reason!r}\n"
         f"output[:200]={after.output[:200]!r}")
    if after.passed:
        print(f"         -> after-fix: PASSED")
    else:
        print(f"         -> after-fix: FAILED — {after.failure_reason}")

    # ------------------------------------------------------------------
    # Step 16: VerificationResult accurately reflects the after-fix run
    # ------------------------------------------------------------------
    vr = report.verification_result
    # The report's VerificationResult was set by the orchestrator's own
    # after-fix run (same code path, same test file).  It must agree with
    # the explicit after_result above (both should be True).
    step(16, "VerificationResult.passed matches after-fix outcome",
         vr.passed == after.passed,
         f"report.verification_result.passed={vr.passed!r} vs "
         f"explicit after-fix passed={after.passed!r}")
    print(f"         -> VerificationResult.passed={vr.passed}, "
          f"output={vr.output[:80]!r}...")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(_hdr("=" * 60))
    if not _failures:
        print(_ok("  ALL 16 STEPS PASSED"))
    else:
        print(_fail(f"  {len(_failures)} STEP(S) FAILED:"))
        for f_name in _failures:
            print(_fail(f"    * {f_name}"))
    print(_hdr("=" * 60))
    print()

    print(_hdr("IncidentReport summary"))
    print(f"  incident_summary : {report.incident_summary[:200]!r}")
    print(f"  root_cause       : {report.root_cause[:150]!r}")
    print(f"  confidence       : {report.confidence}")
    print(f"  affected_files   : {report.affected_files}")
    print(f"  affected_funcs   : {report.affected_functions}")
    print(f"  suspicious_commit: {report.suspicious_commit!r}")
    print(f"  timeline events  : {len(report.timeline)}")
    print(f"  evidence count   : {len(report.evidence)}")
    print(f"  regression_test  : {report.regression_test.name!r}")
    print(f"  suggested_fix    : {report.suggested_fix.description[:100]!r}")
    print(f"  verification     : passed={report.verification_result.passed}")
    print()

    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
