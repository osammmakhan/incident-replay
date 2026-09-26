"""
Tests for incident_replay.agents.test_agent.

All unit tests use in-process tmp_path fixtures — no external repo needed.
Integration tests target f:/GenAI/incident-replay-demo/tests and are skipped
when that directory is absent.

Design mirror: same layering used by test_code_agent.py and test_git_agent.py.
  - Unit: internal helpers (field extraction, scenario description, gap detection)
  - Functional: run() with synthesised test files
  - Integration: run() against the real demo test suite
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from incident_replay.agents.test_agent import (
    TestAgentError,
    TestCase,
    TestFileFindings,
    TestInvestigatorFindings,
    _bare_name,
    _class_part,
    _detect_missing_scenarios,
    _discover_test_files,
    _extract_field_values,
    _rate_relevance,
    _scenario_description,
    _value_is_null,
    run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_TEST_DIR = Path("f:/GenAI/incident-replay-demo/tests")
DEMO_AVAILABLE = DEMO_TEST_DIR.is_dir()

# Affected functions from the demo incident stack trace
DEMO_AFFECTED_FUNCTIONS = [
    "CheckoutService.calculate_discount",
    "CheckoutService.process_checkout",
]

# Error keywords drawn from the demo error signature
DEMO_ERROR_KEYWORDS = ["discount", "nonetype"]

# The exact field=value that triggered the incident
DEMO_INCIDENT_FIELDS = {"discount": "None"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Minimal test file content used in multiple test classes
# ---------------------------------------------------------------------------

# Mirrors the structure of the demo test suite: numeric discount values only.
CHECKOUT_TESTS_NUMERIC_ONLY = """\
    import pytest
    from app.checkout import Order, CheckoutService

    @pytest.fixture
    def service():
        return CheckoutService()

    class TestCalculateDiscount:
        def test_no_discount(self, service):
            order = Order(order_id="X", customer_id="Y", discount=0.0)
            assert service.calculate_discount(order) == 0.0

        def test_ten_percent(self, service):
            order = Order(order_id="X", customer_id="Y", discount=0.10)
            assert service.calculate_discount(order) > 0

        def test_full_discount(self, service):
            order = Order(order_id="X", customer_id="Y", discount=1.0)
            assert service.calculate_discount(order) > 0

    class TestProcessCheckout:
        def test_confirmed_status(self, service):
            order = Order(order_id="X", customer_id="Y", discount=0.0)
            result = service.process_checkout(order)
            assert result["status"] == "confirmed"
"""

# Test file that exercises discount=None explicitly.
CHECKOUT_TESTS_WITH_NONE = """\
    import pytest
    from app.checkout import Order, CheckoutService

    @pytest.fixture
    def service():
        return CheckoutService()

    class TestCalculateDiscount:
        def test_none_discount(self, service):
            order = Order(order_id="X", customer_id="Y")
            order.discount = None
            with pytest.raises(TypeError):
                service.calculate_discount(order)

        def test_numeric_discount(self, service):
            order = Order(order_id="X", customer_id="Y", discount=0.10)
            assert service.calculate_discount(order) > 0
"""

# Unrelated test file (no checkout references).
UNRELATED_TESTS = """\
    class TestSomethingElse:
        def test_addition(self):
            assert 1 + 1 == 2

        def test_string_join(self):
            assert " ".join(["a", "b"]) == "a b"
"""


# ===========================================================================
# Unit tests — internal helpers
# ===========================================================================

class TestValueIsNull:
    def test_none_string(self):
        assert _value_is_null("none") is True

    def test_None_capitalised(self):
        assert _value_is_null("None") is True

    def test_null_string(self):
        assert _value_is_null("null") is True

    def test_nil_string(self):
        assert _value_is_null("nil") is True

    def test_zero_not_null(self):
        assert _value_is_null("0") is False

    def test_zero_float_not_null(self):
        assert _value_is_null("0.0") is False

    def test_numeric_discount_not_null(self):
        assert _value_is_null("0.1") is False

    def test_empty_string_not_null(self):
        assert _value_is_null("") is False


class TestBareNameHelper:
    def test_qualified_name(self):
        assert _bare_name("CheckoutService.calculate_discount") == "calculate_discount"

    def test_bare_name(self):
        assert _bare_name("calculate_discount") == "calculate_discount"

    def test_already_lowercase(self):
        assert _bare_name("foo") == "foo"

    def test_lowercases_result(self):
        assert _bare_name("Service.CalcDiscount") == "calcdiscount"


class TestClassPartHelper:
    def test_qualified_returns_class(self):
        assert _class_part("TestCalculateDiscount.test_no_discount") == "TestCalculateDiscount"

    def test_bare_returns_self(self):
        assert _class_part("test_something") == "test_something"


class TestExtractFieldValues:
    def test_attribute_assignment(self):
        snippet = "order.discount = 0.10"
        result = _extract_field_values(snippet, ["discount"])
        assert "discount" in result
        assert "0.10" in result["discount"]

    def test_keyword_argument(self):
        snippet = "Order(order_id='X', customer_id='Y', discount=0.20)"
        result = _extract_field_values(snippet, ["discount"])
        assert "discount" in result
        assert "0.20" in result["discount"]

    def test_none_assignment(self):
        snippet = "order.discount = None"
        result = _extract_field_values(snippet, ["discount"])
        assert "discount" in result
        assert "none" in result["discount"]

    def test_multiple_values_collected(self):
        snippet = (
            "order.discount = 0.0\n"
            "order.discount = 0.5\n"
            "order.discount = 1.0"
        )
        result = _extract_field_values(snippet, ["discount"])
        assert len(result["discount"]) == 3

    def test_untracked_field_not_returned(self):
        snippet = "order.quantity = 5"
        result = _extract_field_values(snippet, ["discount"])
        assert result == {}

    def test_case_insensitive_field_match(self):
        snippet = "order.Discount = 0.15"
        result = _extract_field_values(snippet, ["discount"])
        assert "discount" in result

    def test_no_duplicates(self):
        snippet = "order.discount = 0.1\norder.discount = 0.1"
        result = _extract_field_values(snippet, ["discount"])
        assert result["discount"].count("0.1") == 1


class TestScenarioDescription:
    """Tests for _scenario_description via a lightweight AST node stub."""

    def _parse_func(self, source: str):
        import ast
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node
        raise ValueError("no function found")

    def test_docstring_used_when_present(self):
        node = self._parse_func("""\
            def test_none_discount(self):
                \"\"\"discount=None raises TypeError.\"\"\"
                pass
        """)
        desc = _scenario_description(node, "")
        assert desc == "discount=None raises TypeError."

    def test_name_tokens_used_without_docstring(self):
        node = self._parse_func("""\
            def test_ten_percent_discount(self):
                pass
        """)
        desc = _scenario_description(node, "")
        assert "ten percent discount" in desc

    def test_test_prefix_stripped(self):
        node = self._parse_func("""\
            def test_no_discount(self):
                pass
        """)
        desc = _scenario_description(node, "")
        assert not desc.startswith("test_")


class TestDetectMissingScenarios:
    def _make_tc(self, qualified: str, field_vals: dict) -> TestCase:
        return TestCase(
            name=qualified.split(".")[-1],
            qualified_name=qualified,
            file="test_x.py",
            start_line=1,
            end_line=5,
            source_snippet="",
            field_values_tested=field_vals,
            covers_affected_function=True,
            scenario_description="placeholder",
        )

    def test_none_not_tested_returns_missing(self):
        tests = [
            self._make_tc("TC.test_zero", {"discount": ["0.0"]}),
            self._make_tc("TC.test_ten", {"discount": ["0.10"]}),
        ]
        gaps = _detect_missing_scenarios(tests, {"discount": "None"})
        assert len(gaps) == 1
        assert "none/null" in gaps[0].lower()

    def test_none_tested_returns_no_missing(self):
        tests = [
            self._make_tc("TC.test_none", {"discount": ["none"]}),
            self._make_tc("TC.test_ten", {"discount": ["0.10"]}),
        ]
        gaps = _detect_missing_scenarios(tests, {"discount": "None"})
        assert gaps == []

    def test_null_tested_returns_no_missing(self):
        tests = [
            self._make_tc("TC.test_null", {"discount": ["null"]}),
        ]
        gaps = _detect_missing_scenarios(tests, {"discount": "None"})
        assert gaps == []

    def test_field_never_set_returns_missing(self):
        # Tests exist but never set the tracked field
        tests = [
            self._make_tc("TC.test_something", {}),
        ]
        gaps = _detect_missing_scenarios(tests, {"discount": "None"})
        assert len(gaps) == 1
        assert "not passed explicitly" in gaps[0]

    def test_no_incident_fields_returns_empty(self):
        tests = [self._make_tc("TC.test_x", {"discount": ["0.1"]})]
        gaps = _detect_missing_scenarios(tests, {})
        assert gaps == []

    def test_non_null_incident_value_exact_missing(self):
        tests = [self._make_tc("TC.test_ten", {"discount": ["0.10"]})]
        gaps = _detect_missing_scenarios(tests, {"discount": "0.99"})
        assert len(gaps) == 1
        assert "0.99" in gaps[0]

    def test_non_null_incident_value_covered(self):
        tests = [self._make_tc("TC.test_exact", {"discount": ["0.99"]})]
        gaps = _detect_missing_scenarios(tests, {"discount": "0.99"})
        assert gaps == []


class TestRateRelevance:
    def _make_tc(self, covers: bool) -> TestCase:
        return TestCase(
            name="test_x",
            qualified_name="TC.test_x",
            file="f.py",
            start_line=1,
            end_line=5,
            source_snippet="",
            field_values_tested={},
            covers_affected_function=covers,
            scenario_description="x",
        )

    def test_high_when_any_test_covers_function(self):
        assert _rate_relevance([self._make_tc(True)], ["calculate_discount"]) == "high"

    def test_low_when_no_tests(self):
        assert _rate_relevance([], ["calculate_discount"]) == "low"

    def test_medium_when_no_directly_covering_test(self):
        assert _rate_relevance([self._make_tc(False)], ["calculate_discount"]) == "medium"


# ===========================================================================
# Discovery
# ===========================================================================

class TestDiscoverTestFiles:
    def test_finds_test_prefix_files(self, tmp_path):
        (tmp_path / "test_foo.py").write_text("")
        (tmp_path / "test_bar.py").write_text("")
        found = _discover_test_files(str(tmp_path))
        names = [Path(f).name for f in found]
        assert "test_foo.py" in names
        assert "test_bar.py" in names

    def test_finds_suffix_test_files(self, tmp_path):
        (tmp_path / "foo_test.py").write_text("")
        found = _discover_test_files(str(tmp_path))
        assert any("foo_test.py" in f for f in found)

    def test_ignores_non_test_files(self, tmp_path):
        (tmp_path / "checkout.py").write_text("")
        (tmp_path / "conftest.py").write_text("")
        found = _discover_test_files(str(tmp_path))
        assert found == []

    def test_recurses_into_subdirectories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "test_nested.py").write_text("")
        found = _discover_test_files(str(tmp_path))
        assert any("test_nested.py" in f for f in found)

    def test_skips_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "test_cached.py").write_text("")
        found = _discover_test_files(str(tmp_path))
        assert not any("__pycache__" in f for f in found)

    def test_empty_directory_returns_empty(self, tmp_path):
        assert _discover_test_files(str(tmp_path)) == []


# ===========================================================================
# Error handling
# ===========================================================================

class TestRunErrors:
    def test_empty_test_root_raises(self):
        with pytest.raises(TestAgentError):
            run("")

    def test_whitespace_test_root_raises(self):
        with pytest.raises(TestAgentError):
            run("   ")

    def test_nonexistent_path_raises(self):
        with pytest.raises(TestAgentError):
            run("/nonexistent/path/to/tests")

    def test_file_path_instead_of_dir_raises(self, tmp_path):
        f = tmp_path / "something.py"
        f.write_text("")
        with pytest.raises(TestAgentError):
            run(str(f))

    def test_empty_directory_returns_low_relevance(self, tmp_path):
        result = run(str(tmp_path))
        assert result.relevance_to_incident == "low"
        assert result.test_files_found == []


# ===========================================================================
# Functional tests with synthesised test files
# ===========================================================================

class TestRunWithSyntheticFiles:
    """run() against in-process tmp files — no external repo needed."""

    def test_returns_test_investigator_findings(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(str(tmp_path))
        assert isinstance(result, TestInvestigatorFindings)

    def test_discovers_test_file(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(str(tmp_path))
        assert "test_checkout.py" in result.test_files_found

    def test_relevant_tests_found_for_affected_function(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        assert len(result.all_relevant_tests) >= 1

    def test_qualified_names_in_relevant_test_names(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        names = result.relevant_test_names
        assert any("TestCalculateDiscount" in n for n in names)

    def test_covered_scenarios_populated(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        assert len(result.covered_scenarios) >= 1

    def test_unrelated_tests_excluded(self, tmp_path):
        _write(tmp_path, "test_other.py", UNRELATED_TESTS)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        assert result.all_relevant_tests == []

    def test_unrelated_file_in_found_but_not_relevant(self, tmp_path):
        _write(tmp_path, "test_other.py", UNRELATED_TESTS)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        assert "test_other.py" in result.test_files_found
        assert "test_other.py" not in result.test_files_relevant

    def test_missing_none_scenario_detected(self, tmp_path):
        """Core behaviour: discount=None is absent → appears in missing_scenarios."""
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
            incident_field_values={"discount": "None"},
        )
        assert len(result.missing_scenarios) >= 1
        missing_text = " ".join(result.missing_scenarios).lower()
        assert "none" in missing_text or "null" in missing_text

    def test_no_missing_when_none_is_tested(self, tmp_path):
        """When a test explicitly sets discount=None, no gap is reported."""
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_WITH_NONE)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
            incident_field_values={"discount": "None"},
        )
        assert result.missing_scenarios == []

    def test_relevance_high_when_function_matched(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        assert result.relevance_to_incident == "high"

    def test_relevance_low_when_no_relevant_tests(self, tmp_path):
        _write(tmp_path, "test_other.py", UNRELATED_TESTS)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        assert result.relevance_to_incident == "low"

    def test_hints_are_labelled(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
            incident_field_values={"discount": "None"},
        )
        for hint in result.interpretation_hints:
            assert hint.startswith("HINT:")

    def test_hint_mentions_missing_scenario(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
            incident_field_values={"discount": "None"},
        )
        assert any("Missing scenario" in h for h in result.interpretation_hints)

    def test_test_case_fields_populated(self, tmp_path):
        """Every TestCase in all_relevant_tests must have the required fields."""
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
        )
        for tc in result.all_relevant_tests:
            assert tc.name != ""
            assert tc.qualified_name != ""
            assert tc.file != ""
            assert tc.start_line > 0
            assert tc.end_line >= tc.start_line
            assert tc.source_snippet.strip() != ""
            assert tc.scenario_description != ""

    def test_file_findings_per_file(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        _write(tmp_path, "test_other.py", UNRELATED_TESTS)
        result = run(str(tmp_path))
        ff_files = [ff.file for ff in result.file_findings]
        assert "test_checkout.py" in ff_files
        assert "test_other.py" in ff_files

    def test_total_tests_counted_per_file(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(str(tmp_path))
        checkout_ff = next(ff for ff in result.file_findings if "checkout" in ff.file)
        # CHECKOUT_TESTS_NUMERIC_ONLY has 4 test functions
        assert checkout_ff.total_tests == 4

    def test_no_affected_functions_all_tests_relevant(self, tmp_path):
        """When no affected_functions filter is provided, all tests are relevant."""
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(str(tmp_path))
        assert len(result.all_relevant_tests) == 4

    def test_syntax_error_recorded_as_warning(self, tmp_path):
        bad = tmp_path / "test_broken.py"
        bad.write_text("def test_bad(\n", encoding="utf-8")
        result = run(str(tmp_path))
        broken_ff = next(ff for ff in result.file_findings if "broken" in ff.file)
        assert len(broken_ff.parse_warnings) >= 1

    def test_field_values_tested_numeric_discount(self, tmp_path):
        _write(tmp_path, "test_checkout.py", CHECKOUT_TESTS_NUMERIC_ONLY)
        result = run(
            str(tmp_path),
            affected_functions=["CheckoutService.calculate_discount"],
            incident_field_values={"discount": "None"},
        )
        all_discount_values: list[str] = []
        for tc in result.all_relevant_tests:
            all_discount_values.extend(tc.field_values_tested.get("discount", []))
        # Tests use 0.0, 0.10, 1.0 — none should be None
        assert len(all_discount_values) >= 1
        assert not any(_value_is_null(v) for v in all_discount_values)


# ===========================================================================
# Integration tests against the real demo test suite
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="incident-replay-demo tests not present")
class TestRunWithDemoTestSuite:
    DIR = str(DEMO_TEST_DIR)

    def _run(self, **kwargs) -> TestInvestigatorFindings:
        return run(
            self.DIR,
            affected_functions=DEMO_AFFECTED_FUNCTIONS,
            error_keywords=DEMO_ERROR_KEYWORDS,
            incident_field_values=DEMO_INCIDENT_FIELDS,
            **kwargs,
        )

    def test_runs_without_error(self):
        result = self._run()
        assert isinstance(result, TestInvestigatorFindings)

    def test_discovers_checkout_test_file(self):
        result = self._run()
        assert any("test_checkout" in f for f in result.test_files_found)

    def test_checkout_file_is_relevant(self):
        result = self._run()
        assert any("test_checkout" in f for f in result.test_files_relevant)

    def test_calculate_discount_tests_found(self):
        result = self._run()
        # Tests live in TestCalculateDiscount; match on class name or function name
        names = result.relevant_test_names
        assert any(
            "calculate_discount" in n.lower() or "TestCalculateDiscount" in n
            for n in names
        )

    def test_numeric_discount_scenarios_covered(self):
        result = self._run()
        assert len(result.covered_scenarios) >= 1

    def test_none_discount_identified_as_missing(self):
        """
        The real test suite has numeric discount cases only.
        The agent must identify discount=None as a missing scenario.
        """
        result = self._run()
        assert len(result.missing_scenarios) >= 1
        combined = " ".join(result.missing_scenarios).lower()
        assert "none" in combined or "null" in combined

    def test_missing_scenario_mentions_discount(self):
        result = self._run()
        combined = " ".join(result.missing_scenarios).lower()
        assert "discount" in combined

    def test_relevance_high(self):
        result = self._run()
        assert result.relevance_to_incident == "high"

    def test_hints_labelled(self):
        result = self._run()
        for hint in result.interpretation_hints:
            assert hint.startswith("HINT:")

    def test_hint_mentions_missing_scenario(self):
        result = self._run()
        assert any("Missing scenario" in h for h in result.interpretation_hints)

    def test_no_causal_claims_in_scenarios(self):
        result = self._run()
        causal = ["root cause", "caused by", "therefore", "bug is"]
        for scenario in result.covered_scenarios:
            for term in causal:
                assert term not in scenario.lower(), (
                    f"Causal claim in scenario: {scenario!r}"
                )

    def test_all_relevant_tests_have_source_snippets(self):
        result = self._run()
        for tc in result.all_relevant_tests:
            assert tc.source_snippet.strip() != "", (
                f"Empty snippet for {tc.qualified_name}"
            )

    def test_file_findings_match_found_files(self):
        result = self._run()
        ff_files = {ff.file for ff in result.file_findings}
        found_files = set(result.test_files_found)
        assert ff_files == found_files

    def test_total_tests_reasonable(self):
        """The demo test file has at least 10 tests — verify the count is sane."""
        result = self._run()
        checkout_ff = next(
            (ff for ff in result.file_findings if "test_checkout" in ff.file), None
        )
        assert checkout_ff is not None
        assert checkout_ff.total_tests >= 10

    def test_no_parse_warnings_for_valid_files(self):
        result = self._run()
        for ff in result.file_findings:
            assert ff.parse_warnings == [], (
                f"Unexpected warnings in {ff.file}: {ff.parse_warnings}"
            )
