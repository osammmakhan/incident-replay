"""
Tests for incident_replay.execution.patcher.

Structure
---------
TestCountChangedLines   — _count_changed_lines() helper
TestApply               — apply() core: success, not-found, ambiguous,
                          no-change, missing file, directory, empty args
TestApplyNullGuard      — apply_null_guard() with demo-style fixture and
                          edge cases (no match, ambiguous, min-only)
TestBuildSuggestedFix   — build_suggested_fix() schema bridge
TestPatchResult         — PatchResult dataclass fields and invariants
TestRepoRoot            — repo_root resolution
TestDemoIntegration     — end-to-end against a copy of the demo project
                          (skipped when demo project is absent)
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from incident_replay.execution.patcher import (
    PatchError,
    PatchResult,
    _count_changed_lines,
    apply,
    apply_null_guard,
    build_suggested_fix,
)
from incident_replay.models.schemas import SuggestedFix

# ---------------------------------------------------------------------------
# Demo-project guard
# ---------------------------------------------------------------------------

DEMO_REPO = Path("f:/GenAI/incident-replay-demo")
DEMO_CHECKOUT = DEMO_REPO / "app" / "checkout.py"
DEMO_AVAILABLE = DEMO_CHECKOUT.is_file()

# ---------------------------------------------------------------------------
# The buggy line exactly as it appears in the demo checkout.py
# ---------------------------------------------------------------------------

BUGGY_LINE = (
    "        discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))\n"
)
FIXED_LINE1 = "        discount_rate = order.discount or 0.0\n"
FIXED_LINE2 = (
    "        discount_rate = max(0.0, min(discount_rate, self.MAX_DISCOUNT_RATE))\n"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _checkout_source() -> str:
    """Minimal checkout.py source that contains the buggy line."""
    return (
        '"""Checkout service."""\n'
        "\n"
        "class CheckoutService:\n"
        "    MAX_DISCOUNT_RATE = 1.0\n"
        "\n"
        "    def calculate_discount(self, order):\n"
        "        # Clamp discount rate\n"
        + BUGGY_LINE
        + "        return order.subtotal * discount_rate\n"
    )


# ===========================================================================
# TestCountChangedLines
# ===========================================================================

class TestCountChangedLines:
    def test_counts_added_and_removed(self):
        diff = (
            "--- a/f.py (before)\n"
            "+++ a/f.py (after)\n"
            "@@ -1,2 +1,3 @@\n"
            "-old line\n"
            "+new line 1\n"
            "+new line 2\n"
            " unchanged\n"
        )
        assert _count_changed_lines(diff) == 3

    def test_excludes_header_lines(self):
        diff = "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n"
        assert _count_changed_lines(diff) == 2

    def test_empty_diff_is_zero(self):
        assert _count_changed_lines("") == 0

    def test_context_lines_not_counted(self):
        diff = "--- a\n+++ b\n@@ -1,3 +1,3 @@\n context\n-old\n+new\n context\n"
        assert _count_changed_lines(diff) == 2


# ===========================================================================
# TestApply  — core replacement function
# ===========================================================================

class TestApply:
    # --- success path -------------------------------------------------------

    def test_applies_simple_replacement(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\ny = 2\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 99\n")
        assert result.applied is True
        assert "x = 99" in p.read_text()

    def test_file_is_actually_modified(self, tmp_path):
        p = _write(tmp_path, "f.py", "a = 1\n")
        apply(p, old_text="a = 1\n", new_text="a = 2\n")
        assert p.read_text() == "a = 2\n"

    def test_returns_patch_result(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert isinstance(result, PatchResult)

    def test_diff_is_non_empty_on_success(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert result.diff != ""

    def test_diff_contains_minus_line(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert "-x = 1" in result.diff

    def test_diff_contains_plus_line(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert "+x = 2" in result.diff

    def test_original_source_preserved(self, tmp_path):
        src = "x = 1\n"
        p = _write(tmp_path, "f.py", src)
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert result.original_source == src

    def test_patched_source_contains_new_text(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert "x = 2" in result.patched_source

    def test_failure_reason_empty_on_success(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert result.failure_reason == ""

    def test_lines_changed_positive_on_success(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert result.lines_changed > 0

    def test_file_path_is_absolute(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert result.file_path.is_absolute()

    def test_unrelated_lines_unchanged(self, tmp_path):
        p = _write(tmp_path, "f.py", "a = 1\nb = 2\nc = 3\n")
        apply(p, old_text="b = 2\n", new_text="b = 99\n")
        content = p.read_text()
        assert "a = 1" in content
        assert "c = 3" in content

    def test_multiline_old_text(self, tmp_path):
        src = "def foo():\n    x = 1\n    return x\n"
        p = _write(tmp_path, "f.py", src)
        result = apply(p, old_text="    x = 1\n    return x\n",
                       new_text="    x = 2\n    return x\n")
        assert result.applied is True
        assert "x = 2" in p.read_text()

    # --- failure: not found -------------------------------------------------

    def test_not_found_applied_false(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="no such text\n", new_text="new\n")
        assert result.applied is False

    def test_not_found_file_unchanged(self, tmp_path):
        src = "x = 1\n"
        p = _write(tmp_path, "f.py", src)
        apply(p, old_text="no such text\n", new_text="new\n")
        assert p.read_text() == src

    def test_not_found_failure_reason(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="ghost\n", new_text="x\n")
        assert "not found" in result.failure_reason

    def test_not_found_diff_empty(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="ghost\n", new_text="x\n")
        assert result.diff == ""

    # --- failure: ambiguous (>1 occurrence) ---------------------------------

    def test_ambiguous_applied_false(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\nx = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert result.applied is False

    def test_ambiguous_file_unchanged(self, tmp_path):
        src = "x = 1\nx = 1\n"
        p = _write(tmp_path, "f.py", src)
        apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert p.read_text() == src

    def test_ambiguous_failure_reason_mentions_count(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\nx = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n")
        assert "2" in result.failure_reason

    # --- failure: no change -------------------------------------------------

    def test_no_change_applied_false(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 1\n")
        assert result.applied is False

    def test_no_change_reason(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 1\n")
        assert "no change" in result.failure_reason

    # --- failure: missing file ----------------------------------------------

    def test_missing_file_applied_false(self, tmp_path):
        result = apply(tmp_path / "ghost.py", old_text="x\n", new_text="y\n")
        assert result.applied is False

    def test_missing_file_reason(self, tmp_path):
        result = apply(tmp_path / "ghost.py", old_text="x\n", new_text="y\n")
        assert "not found" in result.failure_reason

    def test_directory_applied_false(self, tmp_path):
        result = apply(tmp_path, old_text="x\n", new_text="y\n")
        assert result.applied is False

    def test_directory_reason(self, tmp_path):
        result = apply(tmp_path, old_text="x\n", new_text="y\n")
        assert "directory" in result.failure_reason

    # --- PatchError on bad args --------------------------------------------

    def test_empty_file_path_raises(self, tmp_path):
        with pytest.raises(PatchError):
            apply("", old_text="x\n", new_text="y\n")

    def test_empty_old_text_raises(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        with pytest.raises(PatchError):
            apply(p, old_text="", new_text="y\n")


# ===========================================================================
# TestApplyNullGuard
# ===========================================================================

class TestApplyNullGuard:
    # --- demo-style fixture -------------------------------------------------

    def _checkout(self, tmp_path: Path) -> Path:
        p = tmp_path / "checkout.py"
        p.write_text(_checkout_source(), encoding="utf-8")
        return p

    def test_applied_true_on_demo_pattern(self, tmp_path):
        result = apply_null_guard(self._checkout(tmp_path), "discount")
        assert result.applied is True

    def test_file_written_to_disk(self, tmp_path):
        p = self._checkout(tmp_path)
        original = p.read_text()
        apply_null_guard(p, "discount")
        assert p.read_text() != original

    def test_guard_line_present_after_patch(self, tmp_path):
        p = self._checkout(tmp_path)
        apply_null_guard(p, "discount")
        content = p.read_text()
        assert "order.discount or 0.0" in content

    def test_clamp_line_uses_local_var(self, tmp_path):
        p = self._checkout(tmp_path)
        apply_null_guard(p, "discount")
        content = p.read_text()
        assert "min(discount_rate" in content

    def test_buggy_line_removed(self, tmp_path):
        p = self._checkout(tmp_path)
        apply_null_guard(p, "discount")
        content = p.read_text()
        assert "min(order.discount" not in content

    def test_unrelated_lines_preserved(self, tmp_path):
        p = self._checkout(tmp_path)
        apply_null_guard(p, "discount")
        content = p.read_text()
        assert "return order.subtotal * discount_rate" in content
        assert "MAX_DISCOUNT_RATE = 1.0" in content

    def test_patched_file_is_valid_python(self, tmp_path):
        import ast
        p = self._checkout(tmp_path)
        apply_null_guard(p, "discount")
        ast.parse(p.read_text())  # must not raise

    def test_diff_shows_removal_of_buggy_line(self, tmp_path):
        result = apply_null_guard(self._checkout(tmp_path), "discount")
        assert "-" in result.diff
        assert "order.discount" in result.diff

    def test_diff_shows_guard_line_added(self, tmp_path):
        result = apply_null_guard(self._checkout(tmp_path), "discount")
        assert "+        discount_rate = order.discount or 0.0" in result.diff

    def test_failure_reason_empty_on_success(self, tmp_path):
        result = apply_null_guard(self._checkout(tmp_path), "discount")
        assert result.failure_reason == ""

    def test_lines_changed_is_4(self, tmp_path):
        # 1 removed + 3 added (guard, clamp, blank) = 4
        result = apply_null_guard(self._checkout(tmp_path), "discount")
        assert result.lines_changed == 4

    # --- failure: field not found ------------------------------------------

    def test_unknown_field_applied_false(self, tmp_path):
        result = apply_null_guard(self._checkout(tmp_path), "no_such_field")
        assert result.applied is False

    def test_unknown_field_file_unchanged(self, tmp_path):
        p = self._checkout(tmp_path)
        original = p.read_text()
        apply_null_guard(p, "no_such_field")
        assert p.read_text() == original

    def test_unknown_field_reason_mentions_field(self, tmp_path):
        result = apply_null_guard(self._checkout(tmp_path), "no_such_field")
        assert "no_such_field" in result.failure_reason

    # --- failure: missing file ---------------------------------------------

    def test_missing_file_applied_false(self, tmp_path):
        result = apply_null_guard(tmp_path / "ghost.py", "discount")
        assert result.applied is False

    def test_missing_file_reason(self, tmp_path):
        result = apply_null_guard(tmp_path / "ghost.py", "discount")
        assert "not found" in result.failure_reason

    # --- PatchError on bad args -------------------------------------------

    def test_empty_file_path_raises(self):
        with pytest.raises(PatchError):
            apply_null_guard("", "discount")

    def test_empty_field_token_raises(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        with pytest.raises(PatchError):
            apply_null_guard(p, "")

    # --- min-only variant (no outer max) -----------------------------------

    def test_min_only_pattern_patched(self, tmp_path):
        src = "class S:\n    def f(self, o):\n        rate = min(o.fee, cap)\n        return rate\n"
        p = _write(tmp_path, "s.py", src)
        result = apply_null_guard(p, "fee")
        assert result.applied is True
        content = p.read_text()
        assert "o.fee or 0.0" in content
        assert "min(rate" in content

    # --- alternative field names -------------------------------------------

    def test_surcharge_field(self, tmp_path):
        src = "class S:\n    CAP = 1.0\n    def calc(self, o):\n        r = max(0.0, min(o.surcharge, self.CAP))\n        return r\n"
        p = _write(tmp_path, "s.py", src)
        result = apply_null_guard(p, "surcharge")
        assert result.applied is True
        assert "o.surcharge or 0.0" in p.read_text()


# ===========================================================================
# TestBuildSuggestedFix
# ===========================================================================

def _make_result(applied: bool, tmp_path: Path, **kwargs) -> PatchResult:
    p = tmp_path / "f.py"
    p.write_text("x = 1\n")
    defaults = dict(
        applied=applied,
        file_path=p,
        diff="--- a\n+++ b\n-old\n+new\n" if applied else "",
        original_source="x = 1\n",
        patched_source="x = 2\n" if applied else "x = 1\n",
        failure_reason="" if applied else "old_text not found in file",
        lines_changed=2 if applied else 0,
    )
    defaults.update(kwargs)
    return PatchResult(**defaults)


class TestBuildSuggestedFix:
    def test_returns_suggested_fix(self, tmp_path):
        sf = build_suggested_fix(_make_result(True, tmp_path))
        assert isinstance(sf, SuggestedFix)

    def test_applied_patch_is_non_empty(self, tmp_path):
        sf = build_suggested_fix(_make_result(True, tmp_path))
        assert sf.patch != ""

    def test_applied_description_mentions_file(self, tmp_path):
        sf = build_suggested_fix(_make_result(True, tmp_path))
        assert "f.py" in sf.description

    def test_applied_description_mentions_lines_changed(self, tmp_path):
        sf = build_suggested_fix(_make_result(True, tmp_path))
        assert "2" in sf.description

    def test_failed_patch_is_empty(self, tmp_path):
        sf = build_suggested_fix(_make_result(False, tmp_path))
        assert sf.patch == ""

    def test_failed_description_mentions_reason(self, tmp_path):
        sf = build_suggested_fix(_make_result(False, tmp_path))
        assert "not found" in sf.description

    def test_failed_description_mentions_file(self, tmp_path):
        sf = build_suggested_fix(_make_result(False, tmp_path))
        assert "f.py" in sf.description

    def test_schema_accepts_applied_result(self, tmp_path):
        sf = build_suggested_fix(_make_result(True, tmp_path))
        assert isinstance(sf.description, str) and len(sf.description) > 0
        assert isinstance(sf.patch, str)

    def test_schema_accepts_failed_result(self, tmp_path):
        sf = build_suggested_fix(_make_result(False, tmp_path))
        assert isinstance(sf.description, str) and len(sf.description) > 0


# ===========================================================================
# TestPatchResult
# ===========================================================================

class TestPatchResult:
    def _ok(self, tmp_path: Path) -> PatchResult:
        p = _write(tmp_path, "f.py", "x = 1\n")
        return apply(p, old_text="x = 1\n", new_text="x = 2\n")

    def test_applied_is_bool(self, tmp_path):
        assert isinstance(self._ok(tmp_path).applied, bool)

    def test_file_path_is_path(self, tmp_path):
        assert isinstance(self._ok(tmp_path).file_path, Path)

    def test_diff_is_str(self, tmp_path):
        assert isinstance(self._ok(tmp_path).diff, str)

    def test_original_source_is_str(self, tmp_path):
        assert isinstance(self._ok(tmp_path).original_source, str)

    def test_patched_source_is_str(self, tmp_path):
        assert isinstance(self._ok(tmp_path).patched_source, str)

    def test_failure_reason_is_str(self, tmp_path):
        assert isinstance(self._ok(tmp_path).failure_reason, str)

    def test_lines_changed_is_int(self, tmp_path):
        assert isinstance(self._ok(tmp_path).lines_changed, int)

    def test_patched_source_ne_original_on_success(self, tmp_path):
        r = self._ok(tmp_path)
        assert r.patched_source != r.original_source

    def test_patched_source_eq_original_on_failure(self, tmp_path):
        p = _write(tmp_path, "f.py", "x = 1\n")
        r = apply(p, old_text="ghost\n", new_text="y\n")
        assert r.patched_source == r.original_source


# ===========================================================================
# TestRepoRoot
# ===========================================================================

class TestRepoRoot:
    def test_relative_path_resolved_via_repo_root(self, tmp_path):
        p = _write(tmp_path, "module.py", "x = 1\n")
        result = apply("module.py", old_text="x = 1\n", new_text="x = 2\n",
                       repo_root=tmp_path)
        assert result.applied is True

    def test_absolute_path_ignores_repo_root(self, tmp_path):
        p = _write(tmp_path, "module.py", "x = 1\n")
        result = apply(p, old_text="x = 1\n", new_text="x = 2\n",
                       repo_root="/nonexistent")
        assert result.applied is True

    def test_null_guard_repo_root(self, tmp_path):
        src = "class S:\n    def f(self, o):\n        r = max(0.0, min(o.fee, 1.0))\n        return r\n"
        _write(tmp_path, "svc.py", src)
        result = apply_null_guard("svc.py", "fee", repo_root=tmp_path)
        assert result.applied is True


# ===========================================================================
# TestDemoIntegration
# ===========================================================================

@pytest.mark.skipif(
    not DEMO_AVAILABLE,
    reason="Demo project not found at f:/GenAI/incident-replay-demo",
)
class TestDemoIntegration:
    """
    End-to-end: copy the demo checkout.py to a temp directory, apply the
    null-guard patch, and confirm that:
    - applied=True
    - the diff is correct
    - the patched file is valid Python
    - the patched file makes the regression test pass
    """

    @pytest.fixture
    def patched_checkout(self, tmp_path):
        """Copy checkout.py to tmp, apply the patch, yield (result, path)."""
        dest = tmp_path / "checkout.py"
        shutil.copy(DEMO_CHECKOUT, dest)
        result = apply_null_guard(dest, "discount", encoding="utf-8-sig")
        yield result, dest

    def test_applied_true(self, patched_checkout):
        result, _ = patched_checkout
        assert result.applied is True, f"Patch failed: {result.failure_reason}"

    def test_failure_reason_empty(self, patched_checkout):
        result, _ = patched_checkout
        assert result.failure_reason == ""

    def test_diff_removes_buggy_line(self, patched_checkout):
        result, _ = patched_checkout
        assert "-        discount_rate = max(0.0, min(order.discount" in result.diff

    def test_diff_adds_guard_line(self, patched_checkout):
        result, _ = patched_checkout
        assert "+        discount_rate = order.discount or 0.0" in result.diff

    def test_diff_adds_clamp_line(self, patched_checkout):
        result, _ = patched_checkout
        assert "+        discount_rate = max(0.0, min(discount_rate" in result.diff

    def test_patched_file_is_valid_python(self, patched_checkout):
        import ast
        _, dest = patched_checkout
        ast.parse(dest.read_text(encoding="utf-8-sig"))

    def test_patched_file_does_not_contain_buggy_line(self, patched_checkout):
        _, dest = patched_checkout
        content = dest.read_text(encoding="utf-8-sig")
        assert "min(order.discount" not in content

    def test_build_suggested_fix_from_result(self, patched_checkout):
        result, _ = patched_checkout
        sf = build_suggested_fix(result)
        assert isinstance(sf, SuggestedFix)
        assert sf.patch != ""
        assert "checkout.py" in sf.description

    def test_regression_test_passes_after_patch(self, tmp_path):
        """
        Full pipeline: copy the demo, patch checkout.py, run the regression
        test, confirm it now passes.
        """
        import sys
        sys.path.insert(0, str(DEMO_REPO))

        from incident_replay.agents import (
            code_agent, git_agent, log_agent, synthesis_agent, test_agent,
        )
        from incident_replay.execution.test_generator import generate
        from incident_replay.execution.test_runner import run_after_fix

        # Build a patched working copy of the demo project.
        patched_root = tmp_path / "demo_patched"
        shutil.copytree(str(DEMO_REPO), str(patched_root))
        result = apply_null_guard(
            patched_root / "app" / "checkout.py",
            "discount",
            encoding="utf-8-sig",
        )
        assert result.applied, f"Patch did not apply: {result.failure_reason}"

        # Run the synthesis pipeline.
        stack = (DEMO_REPO / "incident" / "stacktrace.txt").read_text()
        lf = log_agent.run(
            str(DEMO_REPO / "logs" / "production.log"), stack_trace=stack
        )
        gf = git_agent.run(
            str(DEMO_REPO),
            incident_time=lf.first_error_time,
            error_keywords=["discount", "null", "TypeError"],
            affected_files=["app/checkout.py"],
        )
        cf = code_agent.run(
            str(DEMO_REPO),
            stack_trace=stack,
            error_keywords=["discount", "null", "TypeError"],
            changed_files=["app/checkout.py"],
        )
        tf = test_agent.run(
            str(DEMO_REPO / "tests"),
            affected_functions=cf.affected_functions,
            error_keywords=["discount", "null", "TypeError"],
        )
        sr = synthesis_agent.run(lf, gf, cf, tf)

        # Generate the regression test into the patched project's tests dir.
        out = generate(
            sr,
            output_dir=patched_root / "tests",
            project_cwd=patched_root,
            run_against_project=False,
        )

        # Run after fix — should pass now.
        run_result = run_after_fix(out.written_path, cwd=patched_root)
        assert run_result.passed, (
            f"Regression test did not pass after fix.\n"
            f"pytest output:\n{run_result.output}"
        )
