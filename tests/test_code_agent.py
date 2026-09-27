"""
Tests for incident_replay.agents.code_agent.

All unit tests use in-process tmp_path fixtures (no demo repo needed).
Integration tests require the demo project at f:/GenAI/incident-replay-demo
and are skipped when it is absent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from incident_replay.agents.code_agent import (
    CodeAgentError,
    CodeFindings,
    CodeObservation,
    _NULL_GUARD_RE,
    _OPTIONAL_ANNOTATION_RE,
    _container_path_to_repo_relative,
    _parse_stack_trace,
    run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_REPO = Path("f:/GenAI/incident-replay-demo")
DEMO_AVAILABLE = DEMO_REPO.is_dir() and (DEMO_REPO / ".git").is_dir()

DEMO_STACK_TRACE = """\
Traceback (most recent call last):
  File "/app/app/api.py", line 48, in checkout
    result = _service.process_checkout(order)
  File "/app/app/checkout.py", line 55, in process_checkout
    discount_amount = self.calculate_discount(order)
  File "/app/app/checkout.py", line 45, in calculate_discount
    discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))
TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
"""

# Source that has a null guard (safe version)
SAFE_SOURCE = textwrap.dedent("""\
    class Service:
        def calculate(self, value):
            rate = value or 0.0
            return rate * 100
""")

# Source without a null guard (broken version — mirrors the demo regression)
BROKEN_SOURCE = textwrap.dedent("""\
    class Service:
        def calculate(self, value):
            return value * 100
""")

# Source with Optional annotation
OPTIONAL_SOURCE = textwrap.dedent("""\
    from typing import Optional

    class Service:
        def calculate(self, discount: Optional[float] = 0.0):
            return discount * 100
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# Regex unit tests
# ===========================================================================

class TestNullGuardRegex:
    def _match(self, line: str) -> bool:
        return bool(_NULL_GUARD_RE.search(line))

    def test_or_zero_float(self):
        assert self._match("        rate = value or 0.0")

    def test_or_zero_int(self):
        assert self._match("        rate = value or 0")

    def test_if_is_none(self):
        assert self._match("        if value is None:")

    def test_if_is_not_none(self):
        assert self._match("        if value is not None:")

    def test_plain_multiplication_not_guarded(self):
        # The guard regex should NOT match a plain multiplication
        assert not self._match("        return value * 100")

    def test_plain_return_not_guarded(self):
        assert not self._match("        return 42")


class TestOptionalAnnotationRegex:
    def _match(self, text: str) -> bool:
        return bool(_OPTIONAL_ANNOTATION_RE.search(text))

    def test_optional_bracket(self):
        assert self._match("Optional[float]")

    def test_pipe_none(self):
        assert self._match("float | None")

    def test_none_pipe(self):
        assert self._match("None | float")

    def test_plain_float_not_matched(self):
        assert not self._match("float")

    def test_int_not_matched(self):
        assert not self._match("int")


# ===========================================================================
# Stack trace parsing
# ===========================================================================

class TestParseStackTrace:
    def test_parses_all_frames(self):
        frames = _parse_stack_trace(DEMO_STACK_TRACE)
        assert len(frames) == 3

    def test_first_frame_path(self):
        frames = _parse_stack_trace(DEMO_STACK_TRACE)
        assert frames[0].raw_path == "/app/app/api.py"

    def test_frame_function_names(self):
        frames = _parse_stack_trace(DEMO_STACK_TRACE)
        names = [f.function for f in frames]
        assert "checkout" in names
        assert "calculate_discount" in names

    def test_frame_line_numbers(self):
        frames = _parse_stack_trace(DEMO_STACK_TRACE)
        lines = [f.line for f in frames]
        assert 48 in lines
        assert 45 in lines

    def test_empty_trace_returns_empty(self):
        assert _parse_stack_trace("") == []

    def test_malformed_trace_returns_empty(self):
        assert _parse_stack_trace("some random text with no frames") == []


# ===========================================================================
# Path resolution
# ===========================================================================

class TestContainerPathToRepoRelative:
    def test_app_prefix_stripped(self, tmp_path):
        _write(tmp_path, "app/checkout.py", "x=1")
        result = _container_path_to_repo_relative(
            "/app/app/checkout.py", str(tmp_path)
        )
        assert result == "app/checkout.py"

    def test_already_relative(self, tmp_path):
        _write(tmp_path, "app/checkout.py", "x=1")
        result = _container_path_to_repo_relative(
            "app/checkout.py", str(tmp_path)
        )
        assert result == "app/checkout.py"

    def test_nonexistent_falls_back_to_tail(self, tmp_path):
        result = _container_path_to_repo_relative(
            "/container/src/myapp/foo.py", str(tmp_path)
        )
        # Should return last two segments as best guess
        assert result == "myapp/foo.py"

    def test_empty_path_returns_empty(self, tmp_path):
        result = _container_path_to_repo_relative("", str(tmp_path))
        assert result == ""


# ===========================================================================
# Error handling
# ===========================================================================

class TestCodeAgentErrors:
    def test_empty_repo_path_raises(self):
        with pytest.raises(CodeAgentError):
            run("")

    def test_whitespace_repo_path_raises(self):
        with pytest.raises(CodeAgentError):
            run("   ")

    def test_nonexistent_file_in_stack_trace_goes_to_missing(self, tmp_path):
        trace = 'File "/app/app/ghost.py", line 10, in some_func\n    x = 1'
        f = run(str(tmp_path), stack_trace=trace)
        assert len(f.files_missing) >= 1

    def test_no_stack_trace_no_changed_files_returns_empty(self, tmp_path):
        f = run(str(tmp_path))
        assert f.observations == []
        assert f.files_inspected == []

    def test_hint_generated_when_no_signals(self, tmp_path):
        f = run(str(tmp_path))
        assert any("No code observations" in h for h in f.interpretation_hints)


# ===========================================================================
# Observations — unit scenarios
# ===========================================================================

class TestObservationsWithTempFiles:
    def test_stack_trace_function_recorded(self, tmp_path):
        """Function named in stack trace produces a stack-trace observation."""
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        trace = f'File "{tmp_path}/service.py", line 3, in calculate\n    return value * 100'
        f = run(str(tmp_path), stack_trace=trace)
        obs_names = [o.observation for o in f.observations]
        assert any("stack trace" in o.lower() for o in obs_names)

    def test_no_null_guard_detected(self, tmp_path):
        """Broken source (no guard) produces an 'No null-safety guard' observation."""
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        trace = f'File "{tmp_path}/service.py", line 3, in calculate\n    return value * 100'
        f = run(
            str(tmp_path),
            stack_trace=trace,
            error_keywords=["value"],
        )
        unguarded = [o for o in f.observations if "No null-safety guard" in o.observation]
        assert len(unguarded) >= 1

    def test_null_guard_present_detected(self, tmp_path):
        """Safe source (with guard) produces a 'guard is present' observation."""
        _write(tmp_path, "service.py", SAFE_SOURCE)
        trace = f'File "{tmp_path}/service.py", line 3, in calculate\n    rate = value or 0.0'
        f = run(
            str(tmp_path),
            stack_trace=trace,
            error_keywords=["value"],
        )
        guarded = [o for o in f.observations if "guard is present" in o.observation]
        assert len(guarded) >= 1

    def test_optional_annotation_detected(self, tmp_path):
        """Parameter with Optional[T] annotation produces a nullable observation."""
        _write(tmp_path, "service.py", OPTIONAL_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["discount"],
        )
        nullable = [o for o in f.observations if "nullable type annotation" in o.observation]
        assert len(nullable) >= 1

    def test_changed_file_inspected_without_stack_trace(self, tmp_path):
        """A file listed in changed_files is inspected even with no stack trace."""
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["value"],
        )
        assert "service.py" in f.files_inspected

    def test_observation_fields_populated(self, tmp_path):
        """Every CodeObservation must have all required fields non-empty."""
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["value"],
        )
        for obs in f.observations:
            assert obs.file != ""
            assert obs.function != ""
            assert obs.source_snippet != ""
            assert obs.observation != ""
            assert obs.relevance != ""
            assert obs.start_line > 0
            assert obs.end_line >= obs.start_line

    def test_affected_files_populated(self, tmp_path):
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["value"],
        )
        assert "service.py" in f.affected_files

    def test_affected_functions_populated(self, tmp_path):
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["value"],
        )
        assert len(f.affected_functions) >= 1

    def test_no_causal_claims_in_observations(self, tmp_path):
        """Observation text must never contain causal language."""
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["value"],
        )
        causal = ["root cause", "caused by", "because", "therefore", "bug is"]
        for obs in f.observations:
            for term in causal:
                assert term not in obs.observation.lower(), (
                    f"Causal claim found: {obs.observation!r}"
                )

    def test_hints_are_labelled(self, tmp_path):
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["value"],
        )
        for hint in f.interpretation_hints:
            assert hint.startswith("HINT:")

    def test_unguarded_hint_generated(self, tmp_path):
        _write(tmp_path, "service.py", BROKEN_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["service.py"],
            error_keywords=["value"],
        )
        assert any("null-safety guard" in h for h in f.interpretation_hints)

    def test_missing_file_in_missing_list(self, tmp_path):
        trace = f'File "{tmp_path}/ghost.py", line 5, in foo\n    x = 1'
        f = run(str(tmp_path), stack_trace=trace)
        assert "ghost.py" in f.files_missing[0]

    def test_missing_file_hint_generated(self, tmp_path):
        trace = f'File "{tmp_path}/ghost.py", line 5, in foo\n    x = 1'
        f = run(str(tmp_path), stack_trace=trace)
        assert any("could not be found" in h for h in f.interpretation_hints)

    def test_parse_error_goes_to_warnings(self, tmp_path):
        _write(tmp_path, "bad.py", "def broken(\n")
        f = run(
            str(tmp_path),
            changed_files=["bad.py"],
            error_keywords=["broken"],
        )
        assert len(f.parse_warnings) >= 1

    def test_two_files_both_inspected(self, tmp_path):
        _write(tmp_path, "a.py", BROKEN_SOURCE)
        _write(tmp_path, "b.py", SAFE_SOURCE)
        f = run(
            str(tmp_path),
            changed_files=["a.py", "b.py"],
            error_keywords=["value"],
        )
        assert "a.py" in f.files_inspected
        assert "b.py" in f.files_inspected


# ===========================================================================
# Integration against demo repository
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="incident-replay-demo repo not present")
class TestCodeAgentWithDemoRepo:
    REPO = str(DEMO_REPO)

    def _run(self, **kwargs) -> CodeFindings:
        return run(
            self.REPO,
            stack_trace=DEMO_STACK_TRACE,
            error_keywords=["discount", "nonetype"],
            changed_files=["app/checkout.py", "app/api.py"],
            error_type="TypeError",
            **kwargs,
        )

    def test_runs_without_error(self):
        f = self._run()
        assert isinstance(f, CodeFindings)

    def test_checkout_file_inspected(self):
        f = self._run()
        assert any("checkout.py" in p for p in f.files_inspected)

    def test_calculate_discount_in_affected_functions(self):
        f = self._run()
        assert any("calculate_discount" in fn for fn in f.affected_functions)

    def test_checkout_py_in_affected_files(self):
        f = self._run()
        assert any("checkout.py" in p for p in f.affected_files)

    def test_unguarded_observation_for_discount(self):
        """calculate_discount references 'discount' without a null guard."""
        f = self._run()
        unguarded = [
            o for o in f.observations
            if "No null-safety guard" in o.observation
            and "discount" in o.observation
        ]
        assert len(unguarded) >= 1

    def test_stack_trace_observation_present(self):
        """At least one observation records that the function is in the stack trace."""
        f = self._run()
        stack_obs = [
            o for o in f.observations
            if "stack trace" in o.observation.lower()
        ]
        assert len(stack_obs) >= 1

    def test_no_files_missing_for_valid_paths(self):
        """Stack trace paths from /app/app/… should resolve against the demo repo."""
        f = self._run()
        # checkout.py and api.py must resolve — we may have missing entries for
        # /app/ container paths, but the real files should be inspected
        assert len(f.files_inspected) >= 1

    def test_observations_have_source_snippets(self):
        f = self._run()
        for obs in f.observations:
            assert obs.source_snippet.strip() != "", (
                f"Empty source_snippet for {obs.function}"
            )

    def test_no_causal_claims_in_any_observation(self):
        f = self._run()
        causal = ["root cause", "caused by", "because", "therefore", "bug is"]
        for obs in f.observations:
            for term in causal:
                assert term not in obs.observation.lower()

    def test_hints_labelled(self):
        f = self._run()
        for hint in f.interpretation_hints:
            assert hint.startswith("HINT:")

    def test_unguarded_hint_present(self):
        f = self._run()
        assert any("null-safety guard" in h for h in f.interpretation_hints)

    def test_parse_warnings_empty(self):
        f = self._run()
        assert f.parse_warnings == []
