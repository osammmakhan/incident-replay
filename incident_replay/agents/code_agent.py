"""
Code Investigator agent.

Responsibility: inspect source files relevant to an incident and return
structured, factual observations about the code — without declaring a
root cause.

Design rules
------------
OBSERVATION vs INTERPRETATION
  Every :class:`CodeObservation` field describes what the source code
  *actually contains*.  The agent identifies patterns, parameter types,
  guard conditions, and call paths — it does not assert that any of these
  *caused* the incident.  Causal judgement belongs to the synthesis stage.

No hardcoded answers
  The agent derives findings from three evidence sources that callers
  supply:
    1. Stack trace text — parsed to extract file paths, function names,
       and the line numbers at which execution was active.
    2. Git findings — changed files from the suspicious commit narrow
       which source files to inspect.
    3. Error keywords — field names / type names from the error signature
       are used to score code patterns as relevant.

  All source analysis is deterministic AST + regex on the actual files in
  the repository at their current state.

No LLM required
  All analysis uses Python's ``ast`` module and regular expressions.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from incident_replay.utils.file_utils import (
    FileReadError,
    FunctionInfo,
    extract_function,
    file_exists,
    find_functions_in_file,
    read_file,
    read_section,
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeObservation:
    """
    A single factual observation about a piece of source code.

    Observations describe what the code *contains or lacks*, not why it
    fails.  The ``relevance`` field explains which evidence signal led to
    this observation being recorded.
    """

    file: str               # Relative or absolute path as supplied / found
    function: str           # Qualified name, e.g. "CheckoutService.calculate_discount"
    start_line: int
    end_line: int
    source_snippet: str     # Verbatim lines from the file

    # What the code contains — a factual statement
    observation: str

    # Which input evidence led to this observation being recorded
    relevance: str


@dataclass
class CodeFindings:
    """
    Structured code observations for one incident investigation.

    All fields describe what the source code *contains* — not what
    caused the incident.
    """

    repo_path: str
    files_inspected: list[str]
    files_missing: list[str]        # Paths that could not be read
    parse_warnings: list[str]       # Non-fatal issues (e.g. parse errors)

    observations: list[CodeObservation]

    # Flat lists for easy consumption by the synthesis stage
    affected_files: list[str]       # Deduplicated file paths with observations
    affected_functions: list[str]   # Deduplicated qualified function names

    # Interpretation hints — explicitly labelled, never causal claims
    interpretation_hints: list[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class CodeAgentError(ValueError):
    """Raised when the agent cannot proceed (e.g. repo_path is empty)."""


def run(
    repo_path: str,
    stack_trace: Optional[str] = None,
    error_keywords: Optional[list[str]] = None,
    changed_files: Optional[list[str]] = None,
    error_type: Optional[str] = None,
) -> CodeFindings:
    """Inspect source files in *repo_path* and return :class:`CodeFindings`.

    Parameters
    ----------
    repo_path:
        Absolute path to the repository root.  All relative file paths
        (from the stack trace and *changed_files*) are resolved against
        this directory.
    stack_trace:
        Raw stack trace text.  File paths and function names are parsed
        from ``File "…", line N, in function_name`` frames.
    error_keywords:
        Words from the error signature (e.g. ``["discount", "NoneType"]``).
        Used to score patterns inside function bodies.
    changed_files:
        File paths (relative to repo root) from the suspicious commit.
        These are inspected even if not in the stack trace.
    error_type:
        The exception class name (e.g. ``"TypeError"``).  Recorded in
        observations for context.

    Returns
    -------
    CodeFindings

    Raises
    ------
    CodeAgentError
        When *repo_path* is empty or blank.
    """
    if not repo_path or not repo_path.strip():
        raise CodeAgentError("repo_path must not be empty.")

    error_keywords = [k.lower() for k in (error_keywords or [])]
    changed_files = list(changed_files or [])

    warnings: list[str] = []
    missing: list[str] = []
    observations: list[CodeObservation] = []

    # 1. Parse stack trace frames → candidate (file, function, line) tuples
    frames = _parse_stack_trace(stack_trace or "")

    # 2. Build the full set of files to inspect
    #    Priority: stack trace frames first, then changed_files
    candidate_paths = _resolve_candidate_paths(
        repo_path, frames, changed_files
    )

    inspected: list[str] = []

    for rel_path, abs_path in candidate_paths:
        if not file_exists(abs_path):
            missing.append(rel_path)
            continue

        inspected.append(rel_path)

        try:
            file_obs, file_warns = _inspect_file(
                abs_path=abs_path,
                rel_path=rel_path,
                frames=frames,
                error_keywords=error_keywords,
                error_type=error_type,
            )
        except (FileReadError, ValueError) as exc:
            warnings.append(f"Could not inspect {rel_path!r}: {exc}")
            continue

        observations.extend(file_obs)
        warnings.extend(file_warns)

    affected_files = _distinct_ordered([o.file for o in observations])
    affected_functions = _distinct_ordered([o.function for o in observations])
    hints = _derive_hints(observations, missing, error_keywords)

    return CodeFindings(
        repo_path=repo_path,
        files_inspected=inspected,
        files_missing=missing,
        parse_warnings=warnings,
        observations=observations,
        affected_files=affected_files,
        affected_functions=affected_functions,
        interpretation_hints=hints,
    )


# ---------------------------------------------------------------------------
# Stack trace parsing
# ---------------------------------------------------------------------------

# Matches:   File "/app/app/checkout.py", line 45, in calculate_discount
_FRAME_RE = re.compile(
    r'File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>\S+)'
)


@dataclass
class _StackFrame:
    raw_path: str       # As written in the traceback (may be /app/… or relative)
    line: int
    function: str


def _parse_stack_trace(text: str) -> list[_StackFrame]:
    """Extract (path, line, function) tuples from a Python traceback."""
    frames: list[_StackFrame] = []
    for m in _FRAME_RE.finditer(text):
        frames.append(_StackFrame(
            raw_path=m.group("path"),
            line=int(m.group("line")),
            function=m.group("func"),
        ))
    return frames


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_candidate_paths(
    repo_path: str,
    frames: list[_StackFrame],
    changed_files: list[str],
) -> list[tuple[str, str]]:
    """
    Return (relative_path, absolute_path) pairs for all candidate files,
    preserving priority order and deduplicating.

    Stack trace paths like ``/app/app/checkout.py`` are mapped to repo
    equivalents (``app/checkout.py``) by stripping common container prefixes
    and then scanning ``repo_path`` for a matching filename.
    """
    seen: set[str] = set()
    results: list[tuple[str, str]] = []

    def _add(rel: str) -> None:
        if rel and rel not in seen:
            seen.add(rel)
            results.append((rel, os.path.join(repo_path, rel)))

    # Stack trace frames first — resolve container paths → repo-relative
    for frame in frames:
        rel = _container_path_to_repo_relative(frame.raw_path, repo_path)
        if rel:
            _add(rel)

    # Changed files from git agent
    for cf in changed_files:
        _add(cf)

    return results


def _container_path_to_repo_relative(raw: str, repo_path: str) -> str:
    """
    Map a container-style absolute path (``/app/app/checkout.py``) to a
    path relative to *repo_path*.

    Strategy:
      1. If *raw* is already relative and the file exists under repo_path → use it.
      2. Try stripping common container prefixes (``/app/``, ``/src/``, etc.).
      3. Walk suffix tails from right to left and check existence.
      4. Return the first match, or the best-guess tail if nothing exists.
    """
    # Normalise to forward slashes
    raw_norm = raw.replace("\\", "/")
    repo_norm = repo_path.replace("\\", "/").rstrip("/")

    # Already relative?
    candidate = os.path.join(repo_path, raw_norm)
    if os.path.isfile(candidate):
        return raw_norm

    # Strip known container prefixes
    for prefix in ("/app/", "/src/", "/code/", "/workspace/", "/project/"):
        if raw_norm.startswith(prefix):
            tail = raw_norm[len(prefix):]
            candidate = os.path.join(repo_path, tail)
            if os.path.isfile(candidate):
                return tail

    # Walk suffix tails (handles /app/app/checkout.py → app/checkout.py)
    parts = [p for p in raw_norm.split("/") if p]
    for start in range(len(parts)):
        tail = "/".join(parts[start:])
        candidate = os.path.join(repo_path, tail)
        if os.path.isfile(candidate):
            return tail

    # Nothing found — return best-guess tail (last two segments)
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    if parts:
        return parts[-1]
    return ""


# ---------------------------------------------------------------------------
# Per-file inspection
# ---------------------------------------------------------------------------

# Patterns that indicate a null-safety guard is present on a parameter
_NULL_GUARD_RE = re.compile(
    r"(?:"
    r"\bor\s+0(?:\.0+)?\b"         # val or 0 / val or 0.0
    r"|\bor\s+['\"][\s]*['\"]"     # val or ""
    r"|\bor\s+\[\]"                # val or []
    r"|\bif\s+\w+\s+is\s+None\b"   # if x is None
    r"|\bif\s+\w+\s+is\s+not\s+None\b"
    r"|\bif\s+\w+\s*(?:!=|==)\s*None\b"
    r"|\w+\s+if\s+\w+\s+is\s+not\s+None\b"  # ternary
    r")"
)

# Patterns that indicate a value is used directly without guarding
_DIRECT_USE_RE = re.compile(
    r"(?:"
    r"\bmin\s*\("         # min(x, ...)
    r"|\bmax\s*\("        # max(x, ...)
    r"|\*\s*\w"           # multiplication
    r"|\w\s*\*"
    r"|\+\s*\w"           # addition
    r"|\w\s*\+"
    r")"
)

# Optional[T] or T | None in type annotations
_OPTIONAL_ANNOTATION_RE = re.compile(
    r"Optional\[|"
    r"\|\s*None\b|"
    r"None\s*\|"
)


def _inspect_file(
    abs_path: str,
    rel_path: str,
    frames: list[_StackFrame],
    error_keywords: list[str],
    error_type: Optional[str],
) -> tuple[list[CodeObservation], list[str]]:
    """
    Produce observations for a single file.

    Returns (observations, warnings).
    """
    observations: list[CodeObservation] = []
    warnings: list[str] = []

    try:
        all_funcs = find_functions_in_file(abs_path)
    except (FileReadError, ValueError) as exc:
        return [], [f"Could not parse {rel_path!r}: {exc}"]

    # Identify which functions to inspect:
    # - Functions named in the stack trace for this file
    # - All functions whose body contains any error keyword
    frame_funcs: set[str] = set()
    frame_lines: dict[str, int] = {}  # function_name → line from traceback

    raw_path_tail = rel_path.replace("\\", "/")
    for frame in frames:
        frame_tail = frame.raw_path.replace("\\", "/")
        # Match if any suffix of the frame path equals the rel_path
        if frame_tail.endswith(raw_path_tail) or raw_path_tail.endswith(
            frame_tail.lstrip("/")
        ):
            frame_funcs.add(frame.function)
            frame_lines[frame.function] = frame.line

    for func in all_funcs:
        in_stack = func.name in frame_funcs or func.qualified_name in frame_funcs
        keyword_hit = _contains_any(func.source.lower(), error_keywords)

        if not (in_stack or keyword_hit):
            continue

        obs_list = _observe_function(
            func=func,
            rel_path=rel_path,
            error_keywords=error_keywords,
            error_type=error_type,
            in_stack=in_stack,
            stack_line=frame_lines.get(func.name) or frame_lines.get(func.qualified_name),
        )
        observations.extend(obs_list)

    return observations, warnings


def _observe_function(
    func: FunctionInfo,
    rel_path: str,
    error_keywords: list[str],
    error_type: Optional[str],
    in_stack: bool,
    stack_line: Optional[int],
) -> list[CodeObservation]:
    """
    Generate observations for a single function.

    Each observation is a factual statement about the code content.
    """
    observations: list[CodeObservation] = []
    source = func.source
    source_lower = source.lower()

    # ---- Observation: function is in the stack trace ----
    if in_stack:
        relevance = (
            f"Function '{func.qualified_name}' appears in the stack trace"
            + (f" at line {stack_line}" if stack_line else "")
            + "."
        )
        observations.append(CodeObservation(
            file=rel_path,
            function=func.qualified_name,
            start_line=func.start_line,
            end_line=func.end_line,
            source_snippet=source,
            observation=(
                f"Function '{func.qualified_name}' is present in the stack trace. "
                f"It spans lines {func.start_line}–{func.end_line}."
            ),
            relevance=relevance,
        ))

    # ---- Observation: Optional / nullable parameter annotation ----
    # Look for parameters annotated as Optional[T] or T | None
    try:
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg in node.args.args:
                if arg.annotation is None:
                    continue
                ann_src = ast.unparse(arg.annotation)
                if _OPTIONAL_ANNOTATION_RE.search(ann_src):
                    observations.append(CodeObservation(
                        file=rel_path,
                        function=func.qualified_name,
                        start_line=func.start_line,
                        end_line=func.end_line,
                        source_snippet=source,
                        observation=(
                            f"Parameter '{arg.arg}' has a nullable type annotation "
                            f"({ann_src}), indicating it may legally be None."
                        ),
                        relevance=(
                            f"Nullable annotation on '{arg.arg}' in "
                            f"'{func.qualified_name}' may allow None to propagate."
                        ),
                    ))
    except SyntaxError:
        pass

    # ---- Observation: keyword-named parameter used directly (no guard) ----
    for kw in error_keywords:
        if kw not in source_lower:
            continue
        # Check whether there is a null guard for this keyword
        kw_lines = [
            l for l in source.splitlines()
            if kw in l.lower() and not l.strip().startswith("#")
        ]
        has_guard = any(_NULL_GUARD_RE.search(l) for l in kw_lines)
        has_direct = any(_DIRECT_USE_RE.search(l) for l in kw_lines
                         if kw in l.lower())

        if kw_lines:
            guard_note = (
                "A null-safety guard is present for this value."
                if has_guard
                else "No null-safety guard was found for this value in this function."
            )
            direct_note = (
                " The value is used directly in an arithmetic/comparison expression."
                if has_direct
                else ""
            )
            observations.append(CodeObservation(
                file=rel_path,
                function=func.qualified_name,
                start_line=func.start_line,
                end_line=func.end_line,
                source_snippet="\n".join(kw_lines),
                observation=(
                    f"Field '{kw}' is referenced in '{func.qualified_name}'. "
                    f"{guard_note}{direct_note}"
                ),
                relevance=(
                    f"'{kw}' appears in the error signature and is present "
                    f"in the body of '{func.qualified_name}'."
                ),
            ))

    # ---- Observation: error_type mentioned in source (e.g. in a comment) ----
    if error_type and error_type.lower() in source_lower:
        observations.append(CodeObservation(
            file=rel_path,
            function=func.qualified_name,
            start_line=func.start_line,
            end_line=func.end_line,
            source_snippet=source,
            observation=(
                f"The exception type '{error_type}' is referenced in the "
                f"source of '{func.qualified_name}'."
            ),
            relevance=(
                f"'{error_type}' is the exception type from the incident "
                f"and appears in the function body."
            ),
        ))

    return observations


# ---------------------------------------------------------------------------
# Interpretation hints
# ---------------------------------------------------------------------------

def _derive_hints(
    observations: list[CodeObservation],
    missing: list[str],
    error_keywords: list[str],
) -> list[str]:
    hints: list[str] = []

    if not observations and not missing:
        hints.append(
            "HINT: No code observations were produced. Supply a stack trace "
            "or changed_files to guide the investigator."
        )
        return hints

    if missing:
        hints.append(
            f"HINT: {len(missing)} file(s) referenced in the stack trace "
            f"could not be found in the repository: {', '.join(missing)}. "
            f"Verify the repo_path and stack trace paths are compatible."
        )

    # Surface functions that lack a null guard for a keyword-named field
    unguarded = [
        o for o in observations
        if "No null-safety guard" in o.observation
    ]
    if unguarded:
        funcs = _distinct_ordered(o.function for o in unguarded)
        hints.append(
            f"HINT: The following function(s) reference a keyword-named field "
            f"without a detected null-safety guard: {', '.join(funcs)}. "
            f"Investigate whether a None value for that field can reach this code."
        )

    nullable_params = [
        o for o in observations
        if "nullable type annotation" in o.observation
    ]
    if nullable_params:
        params = _distinct_ordered(
            re.search(r"Parameter '(\w+)'", o.observation).group(1)
            for o in nullable_params
            if re.search(r"Parameter '(\w+)'", o.observation)
        )
        hints.append(
            f"HINT: Parameter(s) {', '.join(repr(p) for p in params)} are "
            f"declared nullable. Verify that all callers handle or default "
            f"these before use in arithmetic."
        )

    return hints


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def _distinct_ordered(items) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
