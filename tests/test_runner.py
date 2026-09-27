"""
Tests for incident_replay.execution.test_runner.

Covers every public symbol and every PASS/FAIL decision path:

TestOutcome           — _outcome() pure function (the PASS/FAIL gate)
TestRunResultFields   — RunResult dataclass structure
TestRunTestFile       — run_test_file() against real pytest files
TestMissingAndDir     — missing file and directory path handling
TestPhaseLabel        — phase label propagation
TestRunBeforeAfter    — run_before_fix() / run_after_fix() wrappers
TestCompare           — compare() all four before/after combinations
TestToVerificationResult — to_verification_result() schema bridge
TestIsolation         — subprocess isolation (broken import, no crash)
TestDemoBeforeAfter   — end-to-end before/after fix against demo project
                        (skipped when demo project is absent)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from incident_replay.execution.test_runner import (
    RunComparison,
    RunResult,
    RunnerError,
    _EXIT_INTERNALERROR,
    _EXIT_INTERRUPTED,
    _EXIT_NOTESTSCOLLECTED,
    _EXIT_OK,
    _EXIT_TESTSFAILED,
    _EXIT_USAGEERROR,
    _outcome,
    compare,
    run_after_fix,
    run_before_fix,
    run_test_file,
    to_verification_result,
)
from incident_replay.models.schemas import VerificationResult

# ---------------------------------------------------------------------------
# Demo-project availability guard
# ---------------------------------------------------------------------------

DEMO_REPO = Path("f:/GenAI/incident-replay-demo")
DEMO_AVAILABLE = (DEMO_REPO / ".git").is_dir() and (DEMO_REPO / "app" / "checkout.py").is_file()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return p


def _fake_run(
    passed: bool = True,
    exit_code: int = 0,
    stdout: str = "1 passed in 0.01s",
    stderr: str = "",
    failure_reason: str = "",
    phase: str = "",
    output: str = "",
) -> RunResult:
    out = output or (stdout + stderr).strip()
    return RunResult(
        passed=passed,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        output=out,
        failure_reason=failure_reason,
        phase=phase,
        elapsed_seconds=0.0,
    )


# ===========================================================================
# TestOutcome  — the single PASS/FAIL gate
# ===========================================================================

class TestOutcome:
    """
    _outcome(returncode, stdout, timeout) is the central decision function.
    It must be exhaustively tested because every other path runs through it.
    """

    # --- PASS cases ---------------------------------------------------------

    def test_exit_0_with_passed_token_is_pass(self):
        ok, reason = _outcome(0, "1 passed in 0.12s", None)
        assert ok is True
        assert reason == ""

    def test_exit_0_multiple_passed_is_pass(self):
        ok, _ = _outcome(0, "42 passed in 1.23s", None)
        assert ok is True

    def test_exit_0_passed_with_warnings(self):
        ok, _ = _outcome(0, "3 passed, 1 warning in 0.45s", None)
        assert ok is True

    # --- exit 0 but NOT a real pass ----------------------------------------

    def test_exit_0_collect_only_no_passed_token_is_fail(self):
        # --collect-only exits 0 but emits "N tests collected", not "N passed"
        ok, reason = _outcome(0, "53 tests collected in 0.11s", None)
        assert ok is False
        assert "no tests reported" in reason

    def test_exit_0_empty_output_is_fail(self):
        ok, reason = _outcome(0, "", None)
        assert ok is False
        assert reason != ""

    def test_exit_0_only_warnings_no_passed_is_fail(self):
        ok, reason = _outcome(0, "0 warnings in 0.01s", None)
        assert ok is False
        assert reason != ""

    # --- Non-zero exit codes -----------------------------------------------

    def test_exit_1_test_failure(self):
        ok, reason = _outcome(1, "FAILED test_foo.py::test_bar", None)
        assert ok is False
        assert "test failure" in reason

    def test_exit_2_interrupted(self):
        ok, reason = _outcome(2, "", None)
        assert ok is False
        assert "interrupt" in reason

    def test_exit_3_internal_error(self):
        ok, reason = _outcome(3, "", None)
        assert ok is False
        assert "internal" in reason

    def test_exit_4_usage_error(self):
        ok, reason = _outcome(4, "", None)
        assert ok is False
        assert "usage" in reason

    def test_exit_5_no_tests_collected_is_fail(self):
        ok, reason = _outcome(5, "", None)
        assert ok is False
        assert "no tests collected" in reason

    def test_exit_5_never_passes_even_with_passed_token(self):
        # Paranoia: even if output somehow contains "passed", exit 5 is FAIL
        ok, reason = _outcome(5, "1 passed somehow", None)
        assert ok is False

    def test_unknown_exit_code_is_fail(self):
        ok, reason = _outcome(99, "", None)
        assert ok is False
        assert "99" in reason

    def test_negative_exit_code_is_fail(self):
        ok, reason = _outcome(-1, "", None)
        assert ok is False

    # --- Timeout -----------------------------------------------------------

    def test_timeout_overrides_everything(self):
        # Even exit 0 with "passed" output is overridden by timeout
        ok, reason = _outcome(0, "1 passed in 0.1s", 30.0)
        assert ok is False
        assert "timeout" in reason
        assert "30" in reason

    def test_timeout_reason_includes_seconds(self):
        _, reason = _outcome(1, "", 45.0)
        assert "45" in reason


# ===========================================================================
# TestRunResultFields
# ===========================================================================

class TestRunResultFields:
    def test_all_fields_present(self, tmp_path):
        p = _write(tmp_path, "test_f.py", "def test_ok(): assert True\n")
        r = run_test_file(p)
        assert hasattr(r, "passed")
        assert hasattr(r, "exit_code")
        assert hasattr(r, "stdout")
        assert hasattr(r, "stderr")
        assert hasattr(r, "output")
        assert hasattr(r, "failure_reason")
        assert hasattr(r, "phase")
        assert hasattr(r, "elapsed_seconds")

    def test_passed_is_bool(self, tmp_path):
        p = _write(tmp_path, "test_bool.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert isinstance(r.passed, bool)

    def test_exit_code_is_int(self, tmp_path):
        p = _write(tmp_path, "test_code.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert isinstance(r.exit_code, int)

    def test_exit_code_0_on_pass(self, tmp_path):
        p = _write(tmp_path, "test_ec0.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert r.exit_code == 0

    def test_exit_code_1_on_failure(self, tmp_path):
        p = _write(tmp_path, "test_ec1.py", "def test_x(): assert False\n")
        r = run_test_file(p)
        assert r.exit_code == 1

    def test_exit_code_5_on_no_tests(self, tmp_path):
        # A valid Python file with no test_ functions → exit 5
        p = _write(tmp_path, "notests.py", "x = 1\n")
        r = run_test_file(p)
        assert r.exit_code == 5

    def test_stdout_is_string(self, tmp_path):
        p = _write(tmp_path, "test_so.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert isinstance(r.stdout, str)

    def test_stderr_is_string(self, tmp_path):
        p = _write(tmp_path, "test_se.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert isinstance(r.stderr, str)

    def test_output_is_combined(self, tmp_path):
        p = _write(tmp_path, "test_comb.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        # output must contain at least what stdout contains
        assert isinstance(r.output, str)

    def test_failure_reason_empty_on_pass(self, tmp_path):
        p = _write(tmp_path, "test_fr.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert r.failure_reason == ""

    def test_failure_reason_non_empty_on_fail(self, tmp_path):
        p = _write(tmp_path, "test_frfail.py", "def test_x(): assert False\n")
        r = run_test_file(p)
        assert r.failure_reason != ""

    def test_elapsed_seconds_positive(self, tmp_path):
        p = _write(tmp_path, "test_elapsed.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert r.elapsed_seconds >= 0.0


# ===========================================================================
# TestRunTestFile
# ===========================================================================

class TestRunTestFile:
    """Core executor — runs actual pytest subprocesses."""

    def test_passing_test_passed_true(self, tmp_path):
        p = _write(tmp_path, "test_pass.py", """\
            def test_ok():
                assert 1 + 1 == 2
        """)
        assert run_test_file(p).passed is True

    def test_failing_test_passed_false(self, tmp_path):
        p = _write(tmp_path, "test_fail.py", """\
            def test_boom():
                assert False, "intentional"
        """)
        assert run_test_file(p).passed is False

    def test_exception_in_test_passed_false(self, tmp_path):
        p = _write(tmp_path, "test_exc.py", """\
            def test_exc():
                raise RuntimeError("boom")
        """)
        assert run_test_file(p).passed is False

    def test_import_error_passed_false(self, tmp_path):
        p = _write(tmp_path, "test_imp.py", """\
            import nonexistent_module_xyz_abc
            def test_x(): assert True
        """)
        assert run_test_file(p).passed is False

    def test_syntax_error_passed_false(self, tmp_path):
        p = _write(tmp_path, "test_syn.py", "def test_x(: pass")
        assert run_test_file(p).passed is False

    def test_no_test_functions_passed_false(self, tmp_path):
        p = _write(tmp_path, "notests.py", "x = 42\n")
        r = run_test_file(p)
        assert r.passed is False
        assert "no tests collected" in r.failure_reason

    def test_timeout_passed_false(self, tmp_path):
        p = _write(tmp_path, "test_slow.py", """\
            import time
            def test_slow():
                time.sleep(15)
        """)
        r = run_test_file(p, timeout=2.0)
        assert r.passed is False
        assert "timeout" in r.failure_reason

    def test_timeout_exit_code_negative_one(self, tmp_path):
        p = _write(tmp_path, "test_slow2.py", """\
            import time
            def test_slow():
                time.sleep(15)
        """)
        r = run_test_file(p, timeout=2.0)
        assert r.exit_code == -1

    def test_output_contains_pytest_summary_on_pass(self, tmp_path):
        p = _write(tmp_path, "test_sum.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert "passed" in r.output

    def test_output_contains_failure_details(self, tmp_path):
        p = _write(tmp_path, "test_det.py", """\
            def test_marker():
                assert False, "unique_failure_sentinel_abc"
        """)
        r = run_test_file(p)
        assert "unique_failure_sentinel_abc" in r.output

    def test_extra_args_select_subset(self, tmp_path):
        p = _write(tmp_path, "test_sel.py", """\
            def test_a():
                assert True
            def test_b():
                assert False
        """)
        # -k test_a selects only the passing test
        r = run_test_file(p, extra_args=["-k", "test_a"])
        assert r.passed is True

    def test_extra_args_select_failing_subset(self, tmp_path):
        p = _write(tmp_path, "test_sel2.py", """\
            def test_a():
                assert True
            def test_b():
                assert False
        """)
        r = run_test_file(p, extra_args=["-k", "test_b"])
        assert r.passed is False

    def test_cwd_is_respected(self, tmp_path):
        # Write a helper module and a test that imports it; the import only
        # resolves when cwd is tmp_path.
        (tmp_path / "myhelper.py").write_text("VALUE = 42\n", encoding="utf-8")
        p = _write(tmp_path, "test_cwd.py", """\
            from myhelper import VALUE
            def test_value():
                assert VALUE == 42
        """)
        # With correct cwd: passes
        r = run_test_file(p, cwd=tmp_path)
        assert r.passed is True

    def test_phase_default_is_empty(self, tmp_path):
        p = _write(tmp_path, "test_ph.py", "def test_x(): assert True\n")
        r = run_test_file(p)
        assert r.phase == ""

    def test_phase_stored_when_supplied(self, tmp_path):
        p = _write(tmp_path, "test_ph2.py", "def test_x(): assert True\n")
        r = run_test_file(p, phase="before")
        assert r.phase == "before"

    def test_collection_error_passed_false(self, tmp_path):
        # An exception at module level (not inside a test function)
        # causes a collection error — exit 2 or 1
        p = _write(tmp_path, "test_coll.py", """\
            raise ValueError("collection-time error")
            def test_x(): assert True
        """)
        r = run_test_file(p)
        assert r.passed is False


# ===========================================================================
# TestMissingAndDir
# ===========================================================================

class TestMissingAndDir:
    def test_missing_file_passed_false(self, tmp_path):
        r = run_test_file(tmp_path / "nonexistent_test.py")
        assert r.passed is False

    def test_missing_file_reason(self, tmp_path):
        r = run_test_file(tmp_path / "nonexistent_test.py")
        assert "not found" in r.failure_reason

    def test_missing_file_output_mentions_path(self, tmp_path):
        p = tmp_path / "no_such_test.py"
        r = run_test_file(p)
        assert "no_such_test" in r.output

    def test_missing_file_exit_code_minus_one(self, tmp_path):
        r = run_test_file(tmp_path / "ghost.py")
        assert r.exit_code == -1

    def test_directory_path_passed_false(self, tmp_path):
        r = run_test_file(tmp_path)
        assert r.passed is False

    def test_directory_path_reason(self, tmp_path):
        r = run_test_file(tmp_path)
        assert "directory" in r.failure_reason


# ===========================================================================
# TestPhaseLabel
# ===========================================================================

class TestPhaseLabel:
    def test_run_before_fix_sets_phase(self, tmp_path):
        p = _write(tmp_path, "test_p.py", "def test_x(): assert True\n")
        r = run_before_fix(p)
        assert r.phase == "before"

    def test_run_after_fix_sets_phase(self, tmp_path):
        p = _write(tmp_path, "test_p2.py", "def test_x(): assert True\n")
        r = run_after_fix(p)
        assert r.phase == "after"

    def test_run_before_fix_passes_correctly(self, tmp_path):
        p = _write(tmp_path, "test_bf.py", "def test_x(): assert True\n")
        r = run_before_fix(p)
        assert r.passed is True

    def test_run_after_fix_fails_correctly(self, tmp_path):
        p = _write(tmp_path, "test_af.py", "def test_x(): assert False\n")
        r = run_after_fix(p)
        assert r.passed is False


# ===========================================================================
# TestCompare
# ===========================================================================

class TestCompare:
    """compare() all four before/after combinations."""

    def test_fix_verified_when_before_fail_after_pass(self):
        before = _fake_run(passed=False, failure_reason="test failure", phase="before")
        after = _fake_run(passed=True, phase="after")
        c = compare(before, after)
        assert c.fix_verified is True

    def test_fix_not_verified_when_both_fail(self):
        before = _fake_run(passed=False, failure_reason="test failure", phase="before")
        after = _fake_run(passed=False, failure_reason="test failure", phase="after")
        c = compare(before, after)
        assert c.fix_verified is False

    def test_fix_not_verified_when_both_pass(self):
        before = _fake_run(passed=True, phase="before")
        after = _fake_run(passed=True, phase="after")
        c = compare(before, after)
        assert c.fix_verified is False

    def test_fix_not_verified_regression(self):
        # Before passed, after failed → regression
        before = _fake_run(passed=True, phase="before")
        after = _fake_run(passed=False, failure_reason="test failure", phase="after")
        c = compare(before, after)
        assert c.fix_verified is False

    def test_returns_run_comparison(self):
        c = compare(
            _fake_run(passed=False, phase="before"),
            _fake_run(passed=True, phase="after"),
        )
        assert isinstance(c, RunComparison)

    def test_before_and_after_preserved(self):
        before = _fake_run(passed=False, phase="before")
        after = _fake_run(passed=True, phase="after")
        c = compare(before, after)
        assert c.before is before
        assert c.after is after

    def test_summary_is_string(self):
        c = compare(_fake_run(passed=False), _fake_run(passed=True))
        assert isinstance(c.summary, str)
        assert len(c.summary) > 0

    def test_summary_mentions_verified_when_fix_verified(self):
        c = compare(_fake_run(passed=False), _fake_run(passed=True))
        assert "verified" in c.summary.lower()

    def test_summary_mentions_regression_when_regression(self):
        c = compare(_fake_run(passed=True), _fake_run(passed=False, failure_reason="test failure"))
        assert "regression" in c.summary.lower()

    def test_summary_mentions_not_effective_when_still_failing(self):
        c = compare(
            _fake_run(passed=False, failure_reason="test failure"),
            _fake_run(passed=False, failure_reason="test failure"),
        )
        assert "not yet effective" in c.summary.lower() or "still fails" in c.summary.lower()

    def test_summary_warns_when_both_pass(self):
        c = compare(_fake_run(passed=True), _fake_run(passed=True))
        assert "before" in c.summary.lower() or "target" in c.summary.lower()


# ===========================================================================
# TestToVerificationResult
# ===========================================================================

class TestToVerificationResult:
    def test_returns_verification_result(self):
        r = to_verification_result(_fake_run(passed=True))
        assert isinstance(r, VerificationResult)

    def test_passed_propagated(self):
        assert to_verification_result(_fake_run(passed=True)).passed is True
        assert to_verification_result(_fake_run(passed=False)).passed is False

    def test_output_is_string(self):
        r = to_verification_result(_fake_run(passed=True))
        assert isinstance(r.output, str)

    def test_passing_output_is_pytest_output(self):
        run = _fake_run(passed=True, stdout="1 passed in 0.12s", output="1 passed in 0.12s")
        vr = to_verification_result(run)
        assert "passed" in vr.output

    def test_failing_output_has_reason_prefix(self):
        run = _fake_run(
            passed=False,
            failure_reason="test failure",
            output="FAILED test_foo.py",
        )
        vr = to_verification_result(run)
        assert "[test failure]" in vr.output

    def test_failing_no_output_shows_reason(self):
        run = _fake_run(passed=False, failure_reason="no tests collected", output="")
        vr = to_verification_result(run)
        assert "no tests collected" in vr.output

    def test_schema_accepts_both_outcomes(self):
        # VerificationResult requires passed:bool and output:str
        for passed in (True, False):
            run = _fake_run(passed=passed, failure_reason="" if passed else "test failure")
            vr = to_verification_result(run)
            assert isinstance(vr.passed, bool)
            assert isinstance(vr.output, str)


# ===========================================================================
# TestIsolation — subprocess crash containment
# ===========================================================================

class TestIsolation:
    """
    A test file that crashes at import time must not crash the pipeline
    process.  The RunResult must simply carry passed=False.
    """

    def test_module_level_exception_does_not_crash_runner(self, tmp_path):
        p = _write(tmp_path, "test_crash.py", """\
            raise SystemExit("crash at import")
            def test_x(): assert True
        """)
        r = run_test_file(p)
        assert r.passed is False
        # The runner process is still alive (we got here)

    def test_infinite_print_does_not_hang(self, tmp_path):
        # Would hang if stdout was not captured; with capture_output=True it
        # will be terminated by the timeout.
        p = _write(tmp_path, "test_print.py", """\
            import sys
            def test_loud():
                for _ in range(100000):
                    sys.stdout.write("x")
                assert True
        """)
        r = run_test_file(p, timeout=15.0)
        # Either it finishes or times out — either way, not a crash
        assert isinstance(r.passed, bool)

    def test_os_exit_in_test_captured(self, tmp_path):
        p = _write(tmp_path, "test_os_exit.py", """\
            import os
            def test_os_exit():
                os._exit(42)
        """)
        # os._exit bypasses Python cleanup — pytest reports the test as
        # crashed (exit code 2 or 1); either way passed=False
        r = run_test_file(p, timeout=10.0)
        assert r.passed is False


# ===========================================================================
# TestDemoBeforeAfter — skipped when demo project is absent
# ===========================================================================

@pytest.mark.skipif(
    not DEMO_AVAILABLE,
    reason="Demo project not found at f:/GenAI/incident-replay-demo",
)
class TestDemoBeforeAfter:
    """
    End-to-end: run the full pipeline, generate the regression test, then
    run it before and after applying the fix.

    Before:  the buggy code raises TypeError → test FAILS (initially_failing=True)
    After:   the fix is applied in a tmp copy → test PASSES
    """

    @pytest.fixture
    def regression_test_path(self, tmp_path):
        """Generate the regression test and clean it up after the test."""
        import sys
        sys.path.insert(0, str(DEMO_REPO))
        from incident_replay.agents import (
            code_agent, git_agent, log_agent, synthesis_agent, test_agent,
        )
        from incident_replay.execution.test_generator import generate

        lf = log_agent.run(
            str(DEMO_REPO / "logs" / "production.log"),
            stack_trace=(DEMO_REPO / "incident" / "stacktrace.txt").read_text(),
        )
        gf = git_agent.run(
            str(DEMO_REPO),
            incident_time=lf.first_error_time,
            error_keywords=["discount", "null", "TypeError"],
            affected_files=["app/checkout.py"],
        )
        cf = code_agent.run(
            str(DEMO_REPO),
            stack_trace=(DEMO_REPO / "incident" / "stacktrace.txt").read_text(),
            error_keywords=["discount", "null", "TypeError"],
            changed_files=["app/checkout.py"],
        )
        tf = test_agent.run(
            str(DEMO_REPO / "tests"),
            affected_functions=cf.affected_functions,
            error_keywords=["discount", "null", "TypeError"],
        )
        sr = synthesis_agent.run(lf, gf, cf, tf)

        out = generate(
            sr,
            output_dir=tmp_path,
            project_cwd=DEMO_REPO,
            run_against_project=False,
        )
        yield out.written_path

    def test_before_fix_fails(self, regression_test_path):
        """Against the buggy code the test must FAIL."""
        r = run_before_fix(regression_test_path, cwd=DEMO_REPO)
        assert r.passed is False, (
            f"Expected test to fail against buggy code; output:\n{r.output}"
        )

    def test_before_fix_phase_label(self, regression_test_path):
        r = run_before_fix(regression_test_path, cwd=DEMO_REPO)
        assert r.phase == "before"

    def test_before_fix_exit_code_1(self, regression_test_path):
        r = run_before_fix(regression_test_path, cwd=DEMO_REPO)
        assert r.exit_code == 1

    def test_before_fix_failure_reason_is_test_failure(self, regression_test_path):
        r = run_before_fix(regression_test_path, cwd=DEMO_REPO)
        assert r.failure_reason == "test failure"

    def test_after_fix_passes(self, regression_test_path, tmp_path):
        """After applying the null-safety guard the test must PASS."""
        import shutil

        # Copy the demo project to tmp_path so we can patch it safely.
        patched_root = tmp_path / "patched_demo"
        shutil.copytree(str(DEMO_REPO), str(patched_root))

        checkout_path = patched_root / "app" / "checkout.py"
        original = checkout_path.read_text(encoding="utf-8")
        # Apply the fix: restore the null-safety guard
        fixed = original.replace(
            "discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))",
            "discount_rate = order.discount or 0.0\n        "
            "discount_rate = max(0.0, min(discount_rate, self.MAX_DISCOUNT_RATE))",
        )
        checkout_path.write_text(fixed, encoding="utf-8")

        r = run_after_fix(regression_test_path, cwd=patched_root)
        assert r.passed is True, (
            f"Expected test to pass after fix; output:\n{r.output}"
        )

    def test_compare_fix_verified(self, regression_test_path, tmp_path):
        """compare() reports fix_verified=True for the before/after pair."""
        import shutil

        patched_root = tmp_path / "patched_demo2"
        shutil.copytree(str(DEMO_REPO), str(patched_root))
        checkout_path = patched_root / "app" / "checkout.py"
        original = checkout_path.read_text(encoding="utf-8")
        fixed = original.replace(
            "discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))",
            "discount_rate = order.discount or 0.0\n        "
            "discount_rate = max(0.0, min(discount_rate, self.MAX_DISCOUNT_RATE))",
        )
        checkout_path.write_text(fixed, encoding="utf-8")

        before = run_before_fix(regression_test_path, cwd=DEMO_REPO)
        after = run_after_fix(regression_test_path, cwd=patched_root)
        cmp = compare(before, after)

        assert cmp.fix_verified is True
        assert "verified" in cmp.summary.lower()

    def test_to_verification_result_for_before_run(self, regression_test_path):
        r = run_before_fix(regression_test_path, cwd=DEMO_REPO)
        vr = to_verification_result(r)
        assert isinstance(vr, VerificationResult)
        assert vr.passed is False
        assert "[test failure]" in vr.output
