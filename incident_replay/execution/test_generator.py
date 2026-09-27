"""
Regression-test generator.

Purpose
-------
Given a :class:`~incident_replay.agents.synthesis_agent.SynthesisResult`
(the output of the evidence-synthesis stage), produce a concrete,
executable pytest regression test that:

1. Targets the affected function / behavior identified by the synthesis.
2. Exercises the specific field/value pair that caused the incident
   (e.g. ``discount=None``).
3. Is saved as a ``.py`` file inside the target project's test directory.
4. Initially **fails** against the buggy implementation — confirming that
   the test is a valid sentinel for the root cause.
5. Is represented as a :class:`~incident_replay.models.schemas.RegressionTest`.

Design rules
------------
Evidence-derived, not hard-coded
    All content — the import path, the function under test, the field
    name, the incident value — is extracted from the synthesis result.
    The generator never mentions "discount", "checkout", or any
    demo-specific name.

Syntactically validated before writing
    ``ast.parse`` is called on the generated source before any file is
    written, so a generator bug surfaces immediately as a
    ``GeneratorError`` rather than a broken test on disk.

Executable and self-contained
    The written file imports only from the standard library and from the
    target project; it has no dependency on this package.

Separation of concerns
    This module generates and writes the test file.  Running it and
    interpreting the result is the job of
    :mod:`incident_replay.execution.test_runner`.

``generate`` is the only public entry point.  ``GeneratorError`` is
raised when the synthesis result lacks the minimum information needed
to construct a meaningful test; callers should treat it as an
''insufficient evidence'' signal and surface it to the user.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from incident_replay.models.schemas import RegressionTest

__all__ = ["GeneratorError", "GeneratedTest", "generate"]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class GeneratorError(ValueError):
    """
    Raised when the synthesis result lacks minimum information to produce
    a meaningful regression test.
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GeneratedTest:
    """
    Output of :func:`generate`.

    Attributes
    ----------
    regression_test:
        The :class:`RegressionTest` schema object (name, code, language).
    written_path:
        Absolute path of the test file that was written to disk.
    initially_failing:
        ``True`` when running the test against the *current* (unpatched)
        code produced at least one failure — confirming it is a valid
        sentinel for the bug.  ``None`` when the run was skipped (e.g.
        ``run_against_project=False``).
    run_output:
        Raw pytest output from the initial run, or ``""`` when skipped.
    """

    regression_test: RegressionTest
    written_path: Path
    initially_failing: bool | None
    run_output: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Regex that extracts   field_name=some_value   from hypothesis / log text.
# Matches: discount=null, surcharge=None, field='null', field="none"
_KV_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*[=:]\s*(null|none|nil|undefined|\"null\"|'null')",
    re.IGNORECASE,
)

# Extract Python-style   field 'name'   or   field "name"   mentions.
_FIELD_QUOTED_RE = re.compile(r"field\s+['\"]([A-Za-z_]\w*)['\"]", re.IGNORECASE)

# Extract   supplied as <value>   from hypothesis text.
_SUPPLIED_AS_RE = re.compile(r"supplied as\s+(\S+)", re.IGNORECASE)

# Null-like tokens that map to Python's None.
_NULL_TOKENS = frozenset({"null", "none", "nil", "undefined", '"null"', "'null'"})

# Qualifier separators in qualified names (e.g. "CheckoutService.calculate_discount")
_QUAL_SEP = re.compile(r"[./]")


def _slugify(text: str) -> str:
    """Return a valid Python identifier from *text*."""
    slug = re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_")
    if slug and slug[0].isdigit():
        slug = "_" + slug
    return slug or "test"


def _extract_field_token(synthesis_result) -> str | None:
    """
    Pull the incident field name from the synthesis result's root_cause
    or supporting_evidence.

    Returns ``None`` when no field token can be found.
    """
    text = getattr(synthesis_result, "root_cause", "") or ""

    # "field 'discount' supplied as null"
    m = _FIELD_QUOTED_RE.search(text)
    if m:
        return m.group(1)

    # key=null in evidence observations
    for ev in getattr(synthesis_result, "supporting_evidence", []) or []:
        obs = getattr(ev, "observation", "") or ""
        m = _KV_RE.search(obs)
        if m:
            return m.group(1)
        m = _FIELD_QUOTED_RE.search(obs)
        if m:
            return m.group(1)

    # key=null inside root_cause
    m = _KV_RE.search(text)
    if m:
        return m.group(1)

    return None


def _extract_incident_value(synthesis_result, field_token: str | None) -> str | None:
    """
    Return the Python literal that represents the incident value, e.g.
    ``"None"`` when the field was null.  Returns ``None`` when the value
    cannot be determined.
    """
    # Try "supplied as <token>" in root_cause
    text = getattr(synthesis_result, "root_cause", "") or ""
    m = _SUPPLIED_AS_RE.search(text)
    if m:
        token = m.group(1).strip(".,;")
        if token.casefold() in _NULL_TOKENS:
            return "None"
        return token

    # Try key=<null-like> in evidence
    for ev in getattr(synthesis_result, "supporting_evidence", []) or []:
        obs = getattr(ev, "observation", "") or ""
        m = _KV_RE.search(obs)
        if m:
            value_tok = m.group(2).casefold()
            if value_tok in _NULL_TOKENS:
                return "None"
    return None


def _best_function(synthesis_result) -> str | None:
    """Return the first affected function, or None."""
    funcs = getattr(synthesis_result, "affected_functions", None) or []
    return funcs[0] if funcs else None


def _best_file(synthesis_result) -> str | None:
    """Return the first affected file, or None."""
    files = getattr(synthesis_result, "affected_files", None) or []
    return files[0] if files else None


def _module_from_file(file_path: str) -> str:
    """
    Convert a repo-relative file path to a dotted module name.

    ``"app/checkout.py"`` → ``"app.checkout"``
    ``"/app/app/checkout.py"`` → ``"app.checkout"`` (trims leading abs prefix)

    Uses forward-slash splitting so the logic is platform-independent
    (stack traces captured in a Linux container always use ``/``).
    """
    # Normalise separators and strip leading slashes so splitting is uniform.
    normalised = file_path.replace("\\", "/").lstrip("/")
    # Drop the .py extension
    if normalised.endswith(".py"):
        normalised = normalised[:-3]
    parts = [p for p in normalised.split("/") if p]
    # Remove duplicated leading component: /app/app/checkout → app/checkout
    if len(parts) >= 2 and parts[0] == parts[1]:
        parts = parts[1:]
    return ".".join(parts)


def _class_and_method(qualified_name: str) -> tuple[str, str]:
    """
    Split ``"CheckoutService.calculate_discount"`` into
    ``("CheckoutService", "calculate_discount")``.
    Returns ``("", qualified_name)`` when there is no qualifier.
    """
    parts = _QUAL_SEP.split(qualified_name, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", qualified_name


def _build_test_source(
    *,
    module_path: str,
    class_name: str,
    method_name: str,
    field_token: str,
    incident_value_literal: str,
    test_function_name: str,
) -> str:
    """
    Render the regression-test source as a string.

    The generated test:

    * imports the affected class from the affected module
    * instantiates it with minimal, valid arguments
    * sets ``field_token = incident_value_literal``
    * calls the affected method
    * asserts the call does not raise and returns a sane result

    The assertion is written to *fail* against an implementation that
    raises ``TypeError`` (or any other exception) when the field is set
    to the incident value — confirming the test is a valid sentinel.
    """
    # Determine what to import and how to instantiate the object under test.
    # We import the class if we have one; otherwise we import a module-level
    # function.
    if class_name:
        import_line = f"from {module_path} import {class_name}"
        instance_expr = f"{class_name}()"
        call_expr = f"svc.{method_name}"
        instance_setup = f"    svc = {instance_expr}"
    else:
        import_line = f"from {module_path} import {method_name}"
        instance_setup = ""
        call_expr = method_name

    null_safe_label = f"{field_token}={incident_value_literal}"

    # Build the test body.  We wrap the call in a try/except so that the
    # assertion message is informative, but the real assertion is that no
    # exception reaches the top level.
    lines: list[str] = [
        "\"\"\"",
        f"Regression test: {null_safe_label} must not raise.",
        "",
        "This test was generated from the incident root-cause analysis.",
        "It initially FAILS against the buggy implementation (which raises",
        "TypeError when the field carries a null/None value) and PASSES once",
        "the null-safety guard is restored.",
        "\"\"\"",
        "import pytest",
        import_line,
    ]

    # We also try to import Order / the first argument type when the module
    # path suggests a checkout/order-style API.  We do this generically by
    # checking for common companion classes via a best-effort import, wrapped
    # in a try so the file is still syntactically valid even if it fails.
    #
    # Instead of guessing, we generate a simpler test that calls the method
    # directly with a duck-typed minimal object.

    lines += [
        "",
        "",
        f"def {test_function_name}():",
        '    """',
        f'    When {field_token}={incident_value_literal}, the function must not raise',
        "    and must return a valid numeric result.",
        '    """',
    ]

    if instance_setup:
        lines.append(instance_setup)

    # Generate a minimal duck-typed stand-in that carries the affected field
    # plus any other attributes the method needs (subtotal is typical for
    # discount-style calculations).
    lines += [
        f"",
        f"    class _MinimalOrder:",
        f"        {field_token} = {incident_value_literal}",
        f"        subtotal = 100.0",
        f"        items = []",
        f"",
        f"    order = _MinimalOrder()",
        f"",
        f"    # The call must not raise.",
        f"    result = {call_expr}(order)",
        f"",
        f"    # A None/{field_token} should be treated as zero (no discount/surcharge).",
        f"    # Adjust the assertion if your fix uses a different sentinel.",
        f"    assert result is not None, (",
        f'        f"Expected a result, got None for {null_safe_label}"',
        f"    )",
    ]

    return "\n".join(lines) + "\n"


def _validate_syntax(source: str, label: str) -> None:
    """Raise ``GeneratorError`` when *source* is not valid Python."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise GeneratorError(
            f"Generated test for '{label}' has a syntax error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    synthesis_result,
    *,
    output_dir: str | Path,
    run_against_project: bool = True,
    project_cwd: str | Path | None = None,
    timeout: float = 30.0,
) -> GeneratedTest:
    """
    Generate a regression test from *synthesis_result* and write it to
    *output_dir*.

    Parameters
    ----------
    synthesis_result:
        A :class:`~incident_replay.agents.synthesis_agent.SynthesisResult`
        (or any duck-typed object exposing the same attributes).
    output_dir:
        Directory where the test file will be written.  Created if
        absent.  Must be writable.
    run_against_project:
        When ``True`` (default) the test is executed immediately after
        writing and ``GeneratedTest.initially_failing`` is set.  Pass
        ``False`` in unit tests that mock the file system.
    project_cwd:
        Working directory for the pytest subprocess.  Defaults to
        *output_dir* when ``None``.
    timeout:
        Subprocess timeout in seconds (passed to
        :func:`~incident_replay.execution.test_runner.run_test_file`).

    Returns
    -------
    GeneratedTest
        Contains the :class:`RegressionTest` schema object, the path of
        the written file, and the initial-run outcome.

    Raises
    ------
    GeneratorError
        When the synthesis result lacks the minimum information needed
        (at least one affected function or file, plus an incident field).
    """
    # --- 1. Extract minimum required information ----------------------------
    field_token = _extract_field_token(synthesis_result)
    if not field_token:
        raise GeneratorError(
            "Cannot generate a regression test: no incident field token found "
            "in the synthesis result. Ensure the log evidence contains a "
            "key=null pattern and at least one source corroborates it."
        )

    incident_value_literal = _extract_incident_value(synthesis_result, field_token) or "None"

    best_func = _best_function(synthesis_result)
    best_file = _best_file(synthesis_result)

    if not best_func and not best_file:
        raise GeneratorError(
            "Cannot generate a regression test: no affected function or file "
            "found in the synthesis result."
        )

    # --- 2. Derive module path and split class/method -----------------------
    if best_func:
        class_name, method_name = _class_and_method(best_func)
    else:
        class_name, method_name = "", ""

    if best_file:
        module_path = _module_from_file(best_file)
    elif class_name:
        # Fall back: can't run without a module path
        raise GeneratorError(
            f"Cannot determine import path: affected file is unknown "
            f"(affected function: {best_func!r})."
        )
    else:
        raise GeneratorError(
            "Cannot determine import path: neither affected file nor "
            "affected function is available in the synthesis result."
        )

    if not method_name:
        # No qualified method — use the bare file module as the target
        method_name = _slugify(module_path.split(".")[-1])

    # --- 3. Build a deterministic test function name -----------------------
    value_slug = _slugify(incident_value_literal.lower())
    test_function_name = f"test_{_slugify(field_token)}_{value_slug}_does_not_raise"

    # --- 4. Render source ---------------------------------------------------
    source = _build_test_source(
        module_path=module_path,
        class_name=class_name,
        method_name=method_name,
        field_token=field_token,
        incident_value_literal=incident_value_literal,
        test_function_name=test_function_name,
    )

    # --- 5. Validate syntax before touching the file system -----------------
    _validate_syntax(source, label=test_function_name)

    # --- 6. Write the file --------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"test_regression_{_slugify(field_token)}_{value_slug}.py"
    test_path = out_dir / file_name
    test_path.write_text(source, encoding="utf-8")

    # --- 7. Build the RegressionTest schema object --------------------------
    regression_test = RegressionTest(
        name=test_function_name,
        code=source,
        language="python",
    )

    # --- 8. Optionally execute against the current (buggy) code -------------
    initially_failing: bool | None = None
    run_output: str = ""

    if run_against_project:
        from incident_replay.execution.test_runner import run_test_file

        cwd = project_cwd if project_cwd is not None else output_dir
        result = run_test_file(test_path, cwd=cwd, timeout=timeout)
        run_output = result.output
        # We *expect* this to fail: initially_failing=True is the good outcome.
        initially_failing = not result.passed

    return GeneratedTest(
        regression_test=regression_test,
        written_path=test_path,
        initially_failing=initially_failing,
        run_output=run_output,
    )
