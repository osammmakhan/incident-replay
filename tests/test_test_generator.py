"""
Tests for incident_replay.execution.test_generator and test_runner.

Structure
---------
TestExtractFieldToken      — unit tests for _extract_field_token
TestExtractIncidentValue   — unit tests for _extract_incident_value
TestBuildTestSource        — unit tests for _build_test_source
TestValidateSyntax         — unit tests for _validate_syntax
TestModuleFromFile         — unit tests for _module_from_file
TestGenerate               — generate() with tmp_path (no subprocess)
TestGenerateErrors         — generate() raises GeneratorError on bad input
TestRunTestFile            — run_test_file() against real pytest files
TestDemoIntegration        — end-to-end against f:/GenAI/incident-replay-demo
                             (skipped when the demo project is absent)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from incident_replay.execution.test_generator import (
    GeneratedTest,
    GeneratorError,
    _build_test_source,
    _extract_field_token,
    _extract_incident_value,
    _module_from_file,
    _validate_syntax,
    generate,
)
from incident_replay.execution.test_runner import run_test_file
from incident_replay.models.schemas import Evidence, RegressionTest

# ---------------------------------------------------------------------------
# Demo-project availability guard
# ---------------------------------------------------------------------------

DEMO_REPO = Path("f:/GenAI/incident-replay-demo")
DEMO_LOG = DEMO_REPO / "logs" / "production.log"
DEMO_AVAILABLE = DEMO_LOG.is_file() and (DEMO_REPO / ".git").is_dir()


# ---------------------------------------------------------------------------
# Minimal duck-typed SynthesisResult builder
# ---------------------------------------------------------------------------

def _ev(source="log", location="a.log", observation="", relevance=""):
    return Evidence(source=source, location=location,
                    observation=observation, relevance=relevance)


@dataclass
class _FakeSynthesisResult:
    root_cause: str = ""
    affected_files: list = field(default_factory=list)
    affected_functions: list = field(default_factory=list)
    suspicious_commit: str | None = None
    supporting_evidence: list = field(default_factory=list)
    conflicting_evidence: list = field(default_factory=list)
    confidence: float = 0.70
    confidence_reasons: list = field(default_factory=list)
    uncertainty: list = field(default_factory=list)
    used_llm: bool = False


def _checkout_synthesis(**overrides) -> _FakeSynthesisResult:
    """A synthesis result that mirrors the demo checkout incident."""
    defaults = dict(
        root_cause=(
            "Hypothesis: the field 'discount' supplied as null is the cause "
            "of the TypeError failure on POST /checkout. "
            "The failure surfaces in CheckoutService.calculate_discount (app/checkout.py)."
        ),
        affected_files=["app/checkout.py"],
        affected_functions=["CheckoutService.calculate_discount"],
        supporting_evidence=[
            _ev("log", "logs/production.log",
                "dominant error signature: TypeError: unsupported operand; "
                "discount=null pattern observed."),
            _ev("git", "commit abc1234",
                "Commit abc1234: message \"refactor: simplify discount\"; "
                "changed files: app/checkout.py."),
        ],
    )
    defaults.update(overrides)
    return _FakeSynthesisResult(**defaults)


# ===========================================================================
# TestExtractFieldToken
# ===========================================================================

class TestExtractFieldToken:
    def test_extracts_from_root_cause_field_quoted(self):
        sr = _FakeSynthesisResult(
            root_cause="Hypothesis: the field 'discount' supplied as null causes TypeError."
        )
        assert _extract_field_token(sr) == "discount"

    def test_extracts_from_root_cause_double_quoted(self):
        sr = _FakeSynthesisResult(
            root_cause='the field "surcharge" supplied as null'
        )
        assert _extract_field_token(sr) == "surcharge"

    def test_falls_back_to_kv_pattern_in_root_cause(self):
        sr = _FakeSynthesisResult(root_cause="discount=null caused the crash")
        assert _extract_field_token(sr) == "discount"

    def test_extracts_from_log_evidence_observation(self):
        sr = _FakeSynthesisResult(
            root_cause="no field named",
            supporting_evidence=[
                _ev("log", "a.log", "discount=null pattern observed."),
            ],
        )
        assert _extract_field_token(sr) == "discount"

    def test_field_quoted_in_evidence_wins(self):
        sr = _FakeSynthesisResult(
            root_cause="",
            supporting_evidence=[
                _ev("code", "app/checkout.py:45",
                    "field 'discount' used without null-safety guard."),
            ],
        )
        assert _extract_field_token(sr) == "discount"

    def test_returns_none_when_no_field_found(self):
        sr = _FakeSynthesisResult(root_cause="generic error", supporting_evidence=[])
        assert _extract_field_token(sr) is None

    def test_none_root_cause_does_not_raise(self):
        sr = _FakeSynthesisResult(root_cause=None)
        # Should not raise; returns None or a value from evidence
        result = _extract_field_token(sr)
        assert result is None or isinstance(result, str)


# ===========================================================================
# TestExtractIncidentValue
# ===========================================================================

class TestExtractIncidentValue:
    def test_supplied_as_null_returns_None_literal(self):
        sr = _FakeSynthesisResult(
            root_cause="the field 'discount' supplied as null causes TypeError."
        )
        assert _extract_incident_value(sr, "discount") == "None"

    def test_supplied_as_none_case_insensitive(self):
        sr = _FakeSynthesisResult(
            root_cause="field 'x' supplied as None."
        )
        assert _extract_incident_value(sr, "x") == "None"

    def test_falls_back_to_kv_null_in_evidence(self):
        sr = _FakeSynthesisResult(
            root_cause="",
            supporting_evidence=[
                _ev("log", "a.log", "discount=null observed"),
            ],
        )
        assert _extract_incident_value(sr, "discount") == "None"

    def test_returns_none_when_no_value_found(self):
        sr = _FakeSynthesisResult(root_cause="generic failure", supporting_evidence=[])
        assert _extract_incident_value(sr, "discount") is None


# ===========================================================================
# TestModuleFromFile
# ===========================================================================

class TestModuleFromFile:
    def test_simple_relative_path(self):
        assert _module_from_file("app/checkout.py") == "app.checkout"

    def test_nested_path(self):
        assert _module_from_file("services/billing/invoice.py") == "services.billing.invoice"

    def test_strips_py_extension(self):
        result = _module_from_file("app/models.py")
        assert not result.endswith(".py")

    def test_container_path_deduplicated(self):
        # /app/app/checkout.py → app.checkout (leading /app/app → app)
        result = _module_from_file("/app/app/checkout.py")
        assert result == "app.checkout"


# ===========================================================================
# TestBuildTestSource
# ===========================================================================

class TestBuildTestSource:
    def _build(self, **overrides):
        defaults = dict(
            module_path="app.checkout",
            class_name="CheckoutService",
            method_name="calculate_discount",
            field_token="discount",
            incident_value_literal="None",
            test_function_name="test_discount_none_does_not_raise",
        )
        defaults.update(overrides)
        return _build_test_source(**defaults)

    def test_source_is_string(self):
        assert isinstance(self._build(), str)

    def test_contains_import(self):
        src = self._build()
        assert "from app.checkout import CheckoutService" in src

    def test_contains_test_function_name(self):
        src = self._build()
        assert "def test_discount_none_does_not_raise():" in src

    def test_contains_field_assignment(self):
        src = self._build()
        assert "discount = None" in src

    def test_contains_call(self):
        src = self._build()
        assert "svc.calculate_discount(order)" in src

    def test_contains_assert(self):
        src = self._build()
        assert "assert result is not None" in src

    def test_module_level_function_no_class(self):
        src = self._build(class_name="", method_name="calculate_discount")
        assert "from app.checkout import calculate_discount" in src
        assert "svc" not in src

    def test_source_is_valid_python(self):
        import ast
        src = self._build()
        ast.parse(src)  # must not raise


# ===========================================================================
# TestValidateSyntax
# ===========================================================================

class TestValidateSyntax:
    def test_valid_source_does_not_raise(self):
        _validate_syntax("def test_foo(): pass\n", "test_foo")

    def test_invalid_source_raises_generator_error(self):
        with pytest.raises(GeneratorError, match="syntax error"):
            _validate_syntax("def test_foo(: pass\n", "test_foo")

    def test_empty_source_is_valid(self):
        _validate_syntax("", "empty")


# ===========================================================================
# TestGenerate — no subprocess; file is written to tmp_path
# ===========================================================================

class TestGenerate:
    def _sr(self, **overrides):
        return _checkout_synthesis(**overrides)

    def test_returns_generated_test(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert isinstance(out, GeneratedTest)

    def test_regression_test_schema(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        rt = out.regression_test
        assert isinstance(rt, RegressionTest)
        assert rt.language == "python"
        assert rt.name.startswith("test_")
        assert len(rt.code) > 0

    def test_written_path_exists(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert out.written_path.is_file()

    def test_written_path_is_in_output_dir(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert out.written_path.parent == tmp_path

    def test_written_path_is_py_file(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert out.written_path.suffix == ".py"

    def test_file_content_matches_code(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert out.written_path.read_text(encoding="utf-8") == out.regression_test.code

    def test_no_run_when_run_against_project_false(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert out.initially_failing is None
        assert out.run_output == ""

    def test_output_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "new_subdir"
        assert not nested.exists()
        generate(self._sr(), output_dir=nested, run_against_project=False)
        assert nested.is_dir()

    def test_generated_code_is_syntactically_valid(self, tmp_path):
        import ast
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        ast.parse(out.regression_test.code)  # must not raise

    def test_name_contains_field_token(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert "discount" in out.regression_test.name

    def test_file_name_contains_field_token(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert "discount" in out.written_path.name

    def test_different_field_token(self, tmp_path):
        sr = _checkout_synthesis(
            root_cause="the field 'surcharge' supplied as null causes TypeError.",
            affected_files=["app/pricing.py"],
            affected_functions=["PricingService.apply_surcharge"],
            supporting_evidence=[
                _ev("log", "a.log", "surcharge=null observed"),
            ],
        )
        out = generate(sr, output_dir=tmp_path, run_against_project=False)
        assert "surcharge" in out.regression_test.name
        assert "surcharge" in out.regression_test.code

    def test_repeated_generation_overwrites_file(self, tmp_path):
        sr = self._sr()
        out1 = generate(sr, output_dir=tmp_path, run_against_project=False)
        out2 = generate(sr, output_dir=tmp_path, run_against_project=False)
        assert out1.written_path == out2.written_path
        assert out2.written_path.is_file()

    def test_code_contains_affected_class(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert "CheckoutService" in out.regression_test.code

    def test_code_contains_method_call(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert "calculate_discount" in out.regression_test.code

    def test_code_sets_field_to_none(self, tmp_path):
        out = generate(self._sr(), output_dir=tmp_path, run_against_project=False)
        assert "discount = None" in out.regression_test.code


# ===========================================================================
# TestGenerateErrors
# ===========================================================================

class TestGenerateErrors:
    def test_raises_when_no_field_token(self, tmp_path):
        sr = _FakeSynthesisResult(
            root_cause="generic TypeError failure",
            affected_files=["app/checkout.py"],
            affected_functions=["CheckoutService.calculate_discount"],
            supporting_evidence=[],
        )
        with pytest.raises(GeneratorError, match="field token"):
            generate(sr, output_dir=tmp_path, run_against_project=False)

    def test_raises_when_no_affected_file_or_function(self, tmp_path):
        sr = _FakeSynthesisResult(
            root_cause="the field 'discount' supplied as null",
            affected_files=[],
            affected_functions=[],
            supporting_evidence=[],
        )
        with pytest.raises(GeneratorError):
            generate(sr, output_dir=tmp_path, run_against_project=False)

    def test_raises_when_affected_file_empty_string(self, tmp_path):
        sr = _FakeSynthesisResult(
            root_cause="the field 'discount' supplied as null",
            affected_files=[""],
            affected_functions=[""],
            supporting_evidence=[],
        )
        with pytest.raises(GeneratorError):
            generate(sr, output_dir=tmp_path, run_against_project=False)


# ===========================================================================
# TestRunTestFile
# ===========================================================================

class TestRunTestFile:
    def _write(self, tmp_path: Path, name: str, source: str) -> Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(source), encoding="utf-8")
        return p

    def test_passing_test_returns_passed_true(self, tmp_path):
        p = self._write(tmp_path, "test_pass.py", """\
            def test_ok():
                assert 1 + 1 == 2
        """)
        result = run_test_file(p)
        assert result.passed is True

    def test_failing_test_returns_passed_false(self, tmp_path):
        p = self._write(tmp_path, "test_fail.py", """\
            def test_boom():
                assert False, "intentional failure"
        """)
        result = run_test_file(p)
        assert result.passed is False

    def test_error_test_returns_passed_false(self, tmp_path):
        p = self._write(tmp_path, "test_err.py", """\
            def test_raises():
                raise RuntimeError("unexpected error")
        """)
        result = run_test_file(p)
        assert result.passed is False

    def test_output_is_string(self, tmp_path):
        p = self._write(tmp_path, "test_str.py", """\
            def test_x():
                assert True
        """)
        result = run_test_file(p)
        assert isinstance(result.output, str)

    def test_output_contains_pytest_info(self, tmp_path):
        p = self._write(tmp_path, "test_info.py", """\
            def test_dummy():
                assert 2 + 2 == 4
        """)
        result = run_test_file(p)
        # pytest -q output should mention "passed"
        assert "passed" in result.output

    def test_failing_output_mentions_assertion(self, tmp_path):
        p = self._write(tmp_path, "test_assert.py", """\
            def test_msg():
                assert False, "unique_marker_xyz"
        """)
        result = run_test_file(p)
        assert "unique_marker_xyz" in result.output

    def test_syntax_error_returns_passed_false(self, tmp_path):
        p = self._write(tmp_path, "test_syntax.py", "def test(: pass")
        result = run_test_file(p)
        assert result.passed is False

    def test_timeout_returns_passed_false(self, tmp_path):
        p = self._write(tmp_path, "test_sleep.py", """\
            import time
            def test_slow():
                time.sleep(10)
        """)
        result = run_test_file(p, timeout=2.0)
        assert result.passed is False

    def test_extra_args_forwarded(self, tmp_path):
        p = self._write(tmp_path, "test_extra.py", """\
            def test_a():
                assert True
            def test_b():
                assert False
        """)
        # -k test_a selects only test_a which passes
        result = run_test_file(p, extra_args=["-k", "test_a"])
        assert result.passed is True


# ===========================================================================
# TestDemoIntegration — end-to-end, skipped when demo project is absent
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="Demo project not found at f:/GenAI/incident-replay-demo")
class TestDemoIntegration:
    """
    Run the full pipeline against the real demo project and confirm that
    the generated regression test:

    1. Is syntactically valid Python.
    2. Initially FAILS against the buggy code (initially_failing=True).
    3. Is saved inside the demo project's tests/ directory.
    4. Targets discount=None (the specific bug).
    5. Is representable as a RegressionTest schema object.
    """

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Remove any generated test file after the test so the demo stays clean."""
        generated: list[Path] = []
        yield generated
        for p in generated:
            p.unlink(missing_ok=True)

    def _run_pipeline(self):
        import sys
        sys.path.insert(0, str(DEMO_REPO))
        from incident_replay.agents import (
            code_agent, git_agent, log_agent, synthesis_agent, test_agent,
        )

        log_f = str(DEMO_LOG)
        stack = (DEMO_REPO / "incident" / "stacktrace.txt").read_text()

        lf = log_agent.run(log_f, stack_trace=stack)
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
        return synthesis_agent.run(lf, gf, cf, tf)

    def test_initially_failing_against_buggy_code(self, _cleanup):
        sr = self._run_pipeline()
        out_dir = DEMO_REPO / "tests"
        out = generate(
            sr,
            output_dir=out_dir,
            project_cwd=DEMO_REPO,
            run_against_project=True,
        )
        _cleanup.append(out.written_path)

        assert out.initially_failing is True, (
            f"Expected test to fail against buggy code; "
            f"pytest output:\n{out.run_output}"
        )

    def test_targets_discount_none(self, _cleanup):
        sr = self._run_pipeline()
        out_dir = DEMO_REPO / "tests"
        out = generate(
            sr,
            output_dir=out_dir,
            project_cwd=DEMO_REPO,
            run_against_project=False,
        )
        _cleanup.append(out.written_path)

        assert "discount" in out.regression_test.name
        assert "None" in out.regression_test.code
        assert "discount = None" in out.regression_test.code

    def test_written_inside_demo_tests_dir(self, _cleanup):
        sr = self._run_pipeline()
        out_dir = DEMO_REPO / "tests"
        out = generate(
            sr,
            output_dir=out_dir,
            project_cwd=DEMO_REPO,
            run_against_project=False,
        )
        _cleanup.append(out.written_path)

        assert out.written_path.parent == out_dir
        assert out.written_path.is_file()

    def test_regression_test_schema_is_valid(self, _cleanup):
        sr = self._run_pipeline()
        out_dir = DEMO_REPO / "tests"
        out = generate(
            sr,
            output_dir=out_dir,
            project_cwd=DEMO_REPO,
            run_against_project=False,
        )
        _cleanup.append(out.written_path)

        rt = out.regression_test
        assert isinstance(rt, RegressionTest)
        assert rt.language == "python"
        assert rt.name.startswith("test_")
        assert len(rt.code) > 10

    def test_generated_code_is_valid_python(self, _cleanup):
        import ast
        sr = self._run_pipeline()
        out_dir = DEMO_REPO / "tests"
        out = generate(
            sr,
            output_dir=out_dir,
            project_cwd=DEMO_REPO,
            run_against_project=False,
        )
        _cleanup.append(out.written_path)

        ast.parse(out.regression_test.code)  # must not raise

    def test_written_file_is_executable_with_pytest(self, _cleanup):
        """The generated file must be importable by pytest (even if the test fails)."""
        sr = self._run_pipeline()
        out_dir = DEMO_REPO / "tests"
        out = generate(
            sr,
            output_dir=out_dir,
            project_cwd=DEMO_REPO,
            run_against_project=False,
        )
        _cleanup.append(out.written_path)

        # Run pytest --collect-only: exit code 0 means collectible.
        # Note: collect-only never reports "N passed", so we check exit_code
        # directly rather than result.passed (which requires a real test run).
        result = run_test_file(
            out.written_path,
            cwd=DEMO_REPO,
            extra_args=["--collect-only", "-q"],
        )
        assert result.exit_code == 0, (
            f"pytest could not collect generated test:\n{result.output}"
        )
