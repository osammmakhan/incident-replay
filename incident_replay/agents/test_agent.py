"""
Test Investigator agent.

Responsibility: inspect an existing test suite and return structured findings
about which tests cover the affected functionality, what normal cases are
already tested, whether the incident scenario is covered, and what important
scenario is missing.

Design rules
------------
OBSERVATION vs INTERPRETATION
  Every field in TestInvestigatorFindings describes what the test files
  *actually contain*.  The agent does not prescribe a fix or generate a
  regression test — that is the synthesis stage's job.

No hardcoded answers
  Coverage is determined by inspecting real test source with AST analysis and
  text matching.  There is no mention of "discount=None" anywhere in the
  detection logic; the conclusion emerges from comparing the parameter values
  tested for each function against the error keywords supplied by the caller.

No LLM required
  All analysis is deterministic AST + regex on the test source files.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single test function identified in a test file."""

    name: str
    """Bare function name, e.g. ``test_ten_percent_discount``."""

    qualified_name: str
    """Class-qualified name, e.g. ``TestCalculateDiscount.test_ten_percent_discount``."""

    file: str
    """Path to the file that defines this test (relative to *test_root*)."""

    start_line: int
    end_line: int

    source_snippet: str
    """Verbatim source of the test function body."""

    # Values that appear literally in the test body for the tracked fields.
    # e.g. {"discount": ["0.0", "0.10", "1.5"]}
    field_values_tested: dict[str, list[str]]

    # True when this test exercises a function that appears in affected_functions
    covers_affected_function: bool

    # Human-readable statement of what scenario this test exercises.
    # Derived from the test's name tokens + any docstring.
    scenario_description: str


@dataclass
class TestFileFindings:
    """Observations about a single test file."""

    file: str                        # path relative to test_root
    total_tests: int
    relevant_tests: list[TestCase]   # tests that touch the affected functions
    parse_warnings: list[str]


@dataclass
class TestInvestigatorFindings:
    """
    Structured observations from inspecting the test suite.

    All fields describe what the tests *contain* — not what should be added.
    Suggestions for missing coverage are labelled as interpretation hints.
    """

    test_root: str
    """Root directory that was scanned."""

    # --- Coverage summary ---
    test_files_found: list[str]
    """All test files discovered under test_root."""

    test_files_relevant: list[str]
    """Files that contain at least one test covering an affected function."""

    relevant_test_names: list[str]
    """Qualified names of tests that cover the affected functions."""

    # --- Scenario coverage ---
    covered_scenarios: list[str]
    """Human-readable descriptions of the scenarios already tested."""

    # --- Gap detection ---
    # A missing scenario is a combination of (function_name, field, value) where
    # the field value that appeared in the incident is never used in any test.
    # Each entry is a plain string, e.g.
    #   "calculate_discount: 'discount' is never passed as None/null"
    missing_scenarios: list[str]

    # --- Incident relevance ---
    # How tightly the existing tests relate to the incident signature.
    # "high"   — tests cover the exact function that failed
    # "medium" — tests cover the module / class but not the specific function
    # "low"    — no relevant tests found
    relevance_to_incident: str

    # --- Interpretation hints ---
    interpretation_hints: list[str]

    # --- Per-file detail ---
    file_findings: list[TestFileFindings]

    # --- All test cases found (across all relevant files) ---
    all_relevant_tests: list[TestCase]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class TestAgentError(ValueError):
    """Raised when the test root cannot be scanned."""


def run(
    test_root: str,
    affected_functions: Optional[list[str]] = None,
    error_keywords: Optional[list[str]] = None,
    incident_field_values: Optional[dict[str, str]] = None,
) -> TestInvestigatorFindings:
    """Inspect the test suite under *test_root* and return findings.

    Parameters
    ----------
    test_root:
        Directory to scan for test files (``test_*.py`` / ``*_test.py``).
    affected_functions:
        Qualified function names from the incident stack trace, e.g.
        ``["CheckoutService.calculate_discount", "process_checkout"]``.
        Used to decide which tests are *relevant*.
    error_keywords:
        Words extracted from the error signature (e.g. ``["discount",
        "NoneType"]``).  Used to narrow field tracking inside test bodies.
    incident_field_values:
        A mapping of ``{field_name: value_observed_in_incident}`` taken
        from the incident report, e.g. ``{"discount": "None"}``.
        The agent checks whether any relevant test exercises a function
        with that specific value.

    Returns
    -------
    TestInvestigatorFindings

    Raises
    ------
    TestAgentError
        When *test_root* is empty, does not exist, or is not a directory.
    """
    if not test_root or not test_root.strip():
        raise TestAgentError("test_root must not be empty.")

    if not os.path.exists(test_root):
        raise TestAgentError(f"test_root does not exist: {test_root!r}")

    if not os.path.isdir(test_root):
        raise TestAgentError(f"test_root is not a directory: {test_root!r}")

    affected_functions = [f.lower() for f in (affected_functions or [])]
    error_keywords = [k.lower() for k in (error_keywords or [])]
    incident_field_values = {
        k.lower(): str(v).lower()
        for k, v in (incident_field_values or {}).items()
    }

    # --- Discover test files ---
    test_files = _discover_test_files(test_root)

    if not test_files:
        return _empty_findings(test_root, warning="No test files found under test_root.")

    # --- Parse each file ---
    file_findings: list[TestFileFindings] = []
    for tf in test_files:
        ff = _parse_test_file(
            path=tf,
            test_root=test_root,
            affected_functions=affected_functions,
            tracked_fields=list(incident_field_values.keys()) + error_keywords,
        )
        file_findings.append(ff)

    # --- Collect relevant tests across all files ---
    all_relevant: list[TestCase] = []
    for ff in file_findings:
        all_relevant.extend(ff.relevant_tests)

    relevant_files = [ff.file for ff in file_findings if ff.relevant_tests]
    relevant_names = [tc.qualified_name for tc in all_relevant]
    covered_scenarios = [tc.scenario_description for tc in all_relevant]

    # --- Detect missing scenarios ---
    missing = _detect_missing_scenarios(all_relevant, incident_field_values)

    # --- Relevance rating ---
    relevance = _rate_relevance(all_relevant, affected_functions)

    # --- Interpretation hints ---
    hints = _derive_hints(
        relevant_tests=all_relevant,
        missing=missing,
        relevance=relevance,
        incident_field_values=incident_field_values,
    )

    rel_paths = [os.path.relpath(tf, test_root).replace("\\", "/") for tf in test_files]

    return TestInvestigatorFindings(
        test_root=test_root,
        test_files_found=rel_paths,
        test_files_relevant=relevant_files,
        relevant_test_names=relevant_names,
        covered_scenarios=covered_scenarios,
        missing_scenarios=missing,
        relevance_to_incident=relevance,
        interpretation_hints=hints,
        file_findings=file_findings,
        all_relevant_tests=all_relevant,
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_test_files(test_root: str) -> list[str]:
    """Return all test files under *test_root* sorted by path."""
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(test_root):
        # Skip hidden directories and __pycache__
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for fn in sorted(filenames):
            if fn.startswith("test_") and fn.endswith(".py"):
                result.append(os.path.join(dirpath, fn))
            elif fn.endswith("_test.py"):
                result.append(os.path.join(dirpath, fn))
    return sorted(result)


# ---------------------------------------------------------------------------
# AST-based test file parsing
# ---------------------------------------------------------------------------

# Pattern for extracting literal argument values from a call expression.
# We capture string, int, float, and None literals.
_LITERAL_NONE_RE = re.compile(r"\bNone\b")
_LITERAL_NULL_RE = re.compile(r"\bnull\b")


def _parse_test_file(
    path: str,
    test_root: str,
    affected_functions: list[str],
    tracked_fields: list[str],
) -> TestFileFindings:
    """Parse one test file and return its :class:`TestFileFindings`."""
    rel = os.path.relpath(path, test_root).replace("\\", "/")
    warnings: list[str] = []

    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            source = fh.read()
    except OSError as exc:
        return TestFileFindings(
            file=rel,
            total_tests=0,
            relevant_tests=[],
            parse_warnings=[f"Cannot read file: {exc}"],
        )

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        warnings.append(f"Syntax error — file could not be parsed: {exc}")
        return TestFileFindings(
            file=rel,
            total_tests=0,
            relevant_tests=[],
            parse_warnings=warnings,
        )

    lines = source.splitlines()
    all_tests = _extract_test_cases(tree, lines, rel, tracked_fields)

    # A test is "relevant" when its source body references any of the
    # affected function name tokens (bare name, not fully qualified).
    affected_bare = {_bare_name(fn) for fn in affected_functions}

    relevant: list[TestCase] = []
    for tc in all_tests:
        body_lower = tc.source_snippet.lower()
        if affected_bare and any(bare in body_lower for bare in affected_bare):
            tc.covers_affected_function = True
            relevant.append(tc)
        elif not affected_functions:
            # No filter — every test is "relevant"
            tc.covers_affected_function = True
            relevant.append(tc)

    return TestFileFindings(
        file=rel,
        total_tests=len(all_tests),
        relevant_tests=relevant,
        parse_warnings=warnings,
    )


def _extract_test_cases(
    tree: ast.Module,
    lines: list[str],
    file_rel: str,
    tracked_fields: list[str],
) -> list[TestCase]:
    """Walk the AST and return a :class:`TestCase` for every test function."""
    results: list[TestCase] = []

    def _process_func(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qualifier: str,
    ) -> None:
        if not node.name.startswith("test"):
            return
        start = node.lineno
        end = node.end_lineno or start
        snippet = "\n".join(lines[start - 1 : end])
        qualified = f"{qualifier}.{node.name}" if qualifier else node.name

        field_vals = _extract_field_values(snippet, tracked_fields)
        desc = _scenario_description(node, snippet)

        results.append(TestCase(
            name=node.name,
            qualified_name=qualified,
            file=file_rel,
            start_line=start,
            end_line=end,
            source_snippet=snippet,
            field_values_tested=field_vals,
            covers_affected_function=False,  # filled in later
            scenario_description=desc,
        ))

    # Top-level functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _process_func(node, "")
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _process_func(child, node.name)

    return results


def _extract_field_values(snippet: str, tracked_fields: list[str]) -> dict[str, list[str]]:
    """
    Scan *snippet* for assignments/calls that set *tracked_fields* to a value.

    Looks for patterns like:
      order.discount = 0.1
      discount=0.1
      discount=None
    Returns a dict mapping field name → list of distinct string representations
    of the values used.
    """
    result: dict[str, list[str]] = {}
    for field_name in tracked_fields:
        fn_lower = field_name.lower()
        # Match  <field> = <value>  or  <field>=<value>  (assignment or kwarg)
        pattern = re.compile(
            rf"\b{re.escape(fn_lower)}\s*=\s*([^\s,\)\n]+)",
            re.IGNORECASE,
        )
        values: list[str] = []
        for m in pattern.finditer(snippet.lower()):
            val = m.group(1).strip().rstrip(",)")
            if val and val not in values:
                values.append(val)
        if values:
            result[fn_lower] = values
    return result


# Tokens that imply None/null semantics in Python test code.
_NONE_TOKENS = frozenset({"none", "null", "nil"})


def _value_is_null(value: str) -> bool:
    """Return True if *value* represents a None/null literal."""
    return value.lower() in _NONE_TOKENS


def _scenario_description(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    snippet: str,
) -> str:
    """
    Produce a human-readable scenario description from the test function.

    Priority:
      1. First line of the docstring, if present.
      2. Name tokens (snake_case → words).
    """
    # Docstring
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        doc = node.body[0].value.value.strip().splitlines()[0]
        if doc:
            return doc

    # Name tokens
    tokens = node.name.replace("test_", "").replace("_", " ").strip()
    return tokens if tokens else node.name


# ---------------------------------------------------------------------------
# Missing scenario detection
# ---------------------------------------------------------------------------

def _detect_missing_scenarios(
    relevant_tests: list[TestCase],
    incident_field_values: dict[str, str],
) -> list[str]:
    """
    Compare the values exercised in relevant tests against the values that
    appeared in the incident.  Return a list of plain-English gap descriptions.

    E.g. if the incident had ``discount=None`` but every test only uses numeric
    discount values, return:
      ["calculate_discount: 'discount' is never tested with a None/null value"]
    """
    if not incident_field_values:
        return []

    missing: list[str] = []

    for field_name, incident_value in incident_field_values.items():
        incident_is_null = _value_is_null(incident_value)

        # Collect all values for this field across all relevant tests
        all_tested_values: list[str] = []
        for tc in relevant_tests:
            all_tested_values.extend(tc.field_values_tested.get(field_name, []))

        if not all_tested_values:
            # Field never appears in any relevant test at all
            missing.append(
                f"Field '{field_name}' is not passed explicitly in any relevant test "
                f"(incident value was {incident_value!r})."
            )
            continue

        # Check whether the incident value class is covered
        if incident_is_null:
            tested_null = any(_value_is_null(v) for v in all_tested_values)
            if not tested_null:
                # Find which functions the relevant tests cover
                covered_fns = _distinct_ordered([
                    _class_part(tc.qualified_name) for tc in relevant_tests
                    if field_name in tc.field_values_tested
                ])
                fn_label = ", ".join(covered_fns) if covered_fns else "affected function"
                missing.append(
                    f"'{field_name}' is tested with numeric values in {fn_label} "
                    f"but is never passed as None/null — the value observed in the incident."
                )
        else:
            # For non-null incident values, check exact match (best-effort)
            tested_exact = any(
                v.strip("'\"") == incident_value.strip("'\"")
                for v in all_tested_values
            )
            if not tested_exact:
                missing.append(
                    f"'{field_name}' with value {incident_value!r} is not exercised "
                    f"in any relevant test."
                )

    return missing


# ---------------------------------------------------------------------------
# Relevance rating
# ---------------------------------------------------------------------------

def _rate_relevance(
    relevant_tests: list[TestCase],
    affected_functions: list[str],
) -> str:
    """Return "high", "medium", or "low" relevance to the incident."""
    if not relevant_tests:
        return "low"

    # "high" — at least one test directly covers an affected function
    directly_covered = [tc for tc in relevant_tests if tc.covers_affected_function]
    if directly_covered:
        return "high"

    return "medium"


# ---------------------------------------------------------------------------
# Interpretation hints
# ---------------------------------------------------------------------------

def _derive_hints(
    relevant_tests: list[TestCase],
    missing: list[str],
    relevance: str,
    incident_field_values: dict[str, str],
) -> list[str]:
    hints: list[str] = []

    if not relevant_tests:
        hints.append(
            "HINT: No relevant tests were found for the affected functions. "
            "Consider whether the test files are co-located with the source."
        )
        return hints

    if missing:
        for gap in missing:
            hints.append(f"HINT: Missing scenario detected — {gap}")
    else:
        hints.append(
            "HINT: All incident field values appear to be covered by the existing "
            "test suite.  Verify that tests use realistic input ranges."
        )

    if relevance == "high":
        hints.append(
            f"HINT: {len(relevant_tests)} test(s) directly exercise the affected "
            f"function(s).  These are the primary candidates for a regression test."
        )

    return hints


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bare_name(qualified: str) -> str:
    """Return the unqualified function name from 'Class.method' or 'method'."""
    return qualified.rsplit(".", 1)[-1].lower()


def _class_part(qualified: str) -> str:
    """Return the class prefix from 'Class.method', or the full name if none."""
    parts = qualified.rsplit(".", 1)
    return parts[0] if len(parts) == 2 else qualified


def _distinct_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _empty_findings(test_root: str, warning: str = "") -> TestInvestigatorFindings:
    return TestInvestigatorFindings(
        test_root=test_root,
        test_files_found=[],
        test_files_relevant=[],
        relevant_test_names=[],
        covered_scenarios=[],
        missing_scenarios=[],
        relevance_to_incident="low",
        interpretation_hints=[
            f"HINT: {warning}" if warning else "HINT: No test files found."
        ],
        file_findings=[],
        all_relevant_tests=[],
    )
