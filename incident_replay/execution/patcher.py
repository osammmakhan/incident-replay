"""
Code patcher for the Incident Replay workflow.

Purpose
-------
Given a :class:`~incident_replay.models.schemas.SuggestedFix` and the
path(s) of the affected file(s), apply the minimal code change that
resolves the root cause, verify the change was actually made, and
return a structured result containing the unified diff and the patch
status.

Design rules
------------
No silent success
    :func:`apply` returns a :class:`PatchResult` with ``applied=False``
    and a ``failure_reason`` whenever the change cannot be made safely.
    It never reports ``applied=True`` unless the file on disk was
    actually modified.

Exact-match replacement only
    The patcher searches for the *exact* old text in the file.  If the
    text is not found, or is found more than once, the patch is rejected.
    This is intentionally conservative: a false negative (patch rejected)
    is far safer than a false positive (wrong text modified).

One file, one change
    Each :func:`apply` call targets exactly one file.  It will not touch
    any other file.  Callers that need to patch multiple files must call
    it once per file.

Atomic write
    The patched content is written in a single ``write_text`` call.  If
    the write fails the original file is left unchanged (the in-memory
    replacement is discarded).

No external tools
    The patcher uses only the standard library (``difflib``, ``pathlib``,
    ``re``).  No ``git apply``, no ``patch``, no subprocess.

Separation of concerns
    This module writes the fix.  Running the regression test to verify
    the fix is the job of :mod:`incident_replay.execution.test_runner`.

``SuggestedFix.patch``  format
-------------------------------
The ``patch`` field on :class:`~incident_replay.models.schemas.SuggestedFix`
is a *display* diff stored for the UI.  The patcher does **not** parse it
to drive the replacement; instead it derives the old and new text from
``old_lines`` / ``new_lines`` that the caller supplies explicitly.  The
``patch`` field is populated (or overwritten) by :func:`apply` with the
real unified diff that was produced.

Public API
----------
``PatchResult``
    Dataclass capturing whether the patch was applied, the path of the
    modified file, the unified diff, the original source, and the patched
    source.  When ``applied=False`` the file is unchanged and
    ``failure_reason`` explains why.

``apply(file_path, old_text, new_text, *, repo_root, encoding)``
    Core function.  Replaces *old_text* with *new_text* in *file_path*,
    confirms the replacement happened exactly once, and returns a
    ``PatchResult``.

``apply_null_guard(file_path, field_token, *, repo_root, encoding)``
    Higher-level helper: given the name of the nullable field (e.g.
    ``"discount"``), finds the unguarded line of the form::

        <variable> = max(0.0, min(<obj>.<field>, ...))

    or::

        <variable> = min(<obj>.<field>, ...)

    and replaces it with two lines that first guard ``<field>`` against
    ``None``::

        <variable> = <obj>.<field> or 0.0
        <variable> = max(0.0, min(<variable>, ...))

    Returns a ``PatchResult``.  If the pattern is not found, returns
    ``applied=False`` without touching the file.

``build_suggested_fix(patch_result)``
    Convert a ``PatchResult`` to the
    :class:`~incident_replay.models.schemas.SuggestedFix` schema object
    consumed by :class:`~incident_replay.models.schemas.IncidentReport`.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from incident_replay.models.schemas import SuggestedFix

__all__ = [
    "PatchError",
    "PatchResult",
    "apply",
    "apply_null_guard",
    "build_suggested_fix",
]


# ---------------------------------------------------------------------------
# Regex for the unguarded-null pattern
# ---------------------------------------------------------------------------

# Matches lines of the form (with arbitrary leading whitespace):
#   <var> = max(0.0, min(<obj>.<field>, ...))
#   <var> = min(<obj>.<field>, ...)
#   <var> = max(<obj>.<field>, ...)
# Capture groups:
#   1 - leading whitespace
#   2 - variable name (lhs)
#   3 - entire rhs (for reconstruction)
#   4 - object prefix, e.g. "order."
#   5 - field token, e.g. "discount"
_UNGUARDED_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*"
    r"(?P<rhs>"
    r"(?:max\s*\(\s*0(?:\.0+)?\s*,\s*)?"       # optional max(0.0, ...
    r"min\s*\(\s*"
    r"(?P<obj_prefix>[\w.]+\.)"                  # e.g. "order."
    r"(?P<field>\w+)"                            # e.g. "discount"
    r"(?P<rest>[^)]*\))"                         # rest of min(...)
    r"(?:\s*\))?"                                # closing ) of max(
    r")"
)


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class PatchError(RuntimeError):
    """
    Raised only for structural mistakes in caller arguments (e.g. empty
    ``file_path``).  Logical patch failures — pattern not found, file not
    readable, ambiguous match — are returned as ``PatchResult(applied=False)``
    so the pipeline can handle them gracefully.
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    """
    Outcome of a single patch attempt.

    Attributes
    ----------
    applied:
        ``True`` only when the file on disk was actually modified.
    file_path:
        Absolute path of the targeted file (whether modified or not).
    diff:
        Unified diff string (``fromfile`` / ``tofile`` headers included).
        Empty string when ``applied`` is ``False``.
    original_source:
        Full content of the file *before* the patch.  Empty string when
        the file could not be read.
    patched_source:
        Full content of the file *after* the patch.  Equal to
        ``original_source`` when ``applied`` is ``False``.
    failure_reason:
        Short human-readable explanation of why the patch failed.
        Empty string when ``applied`` is ``True``.
    lines_changed:
        Number of lines that differ between original and patched source.
        0 when ``applied`` is ``False``.
    """

    applied: bool
    file_path: Path
    diff: str
    original_source: str
    patched_source: str
    failure_reason: str
    lines_changed: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read(file_path: Path, encoding: str) -> tuple[str, str]:
    """
    Return ``(source, error_reason)``.  On success ``error_reason`` is ``""``.
    """
    try:
        return file_path.read_text(encoding=encoding), ""
    except FileNotFoundError:
        return "", f"file not found: {file_path}"
    except IsADirectoryError:
        return "", f"path is a directory, not a file: {file_path}"
    except PermissionError:
        return "", f"permission denied reading: {file_path}"
    except OSError as exc:
        return "", f"cannot read {file_path}: {exc}"


def _unified_diff(original: str, patched: str, file_path: Path) -> str:
    """Return a unified diff between *original* and *patched* as a single string."""
    label = str(file_path)
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"{label} (before)",
        tofile=f"{label} (after)",
    )
    return "".join(diff_lines)


def _count_changed_lines(diff: str) -> int:
    """Count lines that start with ``+`` or ``-`` (excluding the ``---``/``+++`` headers)."""
    count = 0
    for line in diff.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("---", "+++")):
            count += 1
    return count


def _failed(
    file_path: Path,
    original_source: str,
    reason: str,
) -> PatchResult:
    return PatchResult(
        applied=False,
        file_path=file_path,
        diff="",
        original_source=original_source,
        patched_source=original_source,
        failure_reason=reason,
        lines_changed=0,
    )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def apply(
    file_path: str | Path,
    old_text: str,
    new_text: str,
    *,
    repo_root: str | Path | None = None,
    encoding: str = "utf-8",
) -> PatchResult:
    """
    Replace *old_text* with *new_text* in *file_path* and return a
    :class:`PatchResult`.

    The replacement is accepted only when *old_text* appears **exactly
    once** in the file.  Zero or multiple matches both produce
    ``applied=False`` — the file is never modified.

    Parameters
    ----------
    file_path:
        Path of the file to patch.  Resolved relative to *repo_root*
        when *repo_root* is supplied and *file_path* is not absolute.
    old_text:
        The exact text to search for.  Must match literally (no regex).
    new_text:
        The replacement text.
    repo_root:
        Optional base directory used to resolve a relative *file_path*.
    encoding:
        File encoding.  Defaults to ``"utf-8"``.

    Returns
    -------
    PatchResult
        ``applied=True`` iff the file was modified.

    Raises
    ------
    PatchError
        When *file_path* or *old_text* is empty.
    """
    if not file_path:
        raise PatchError("file_path must not be empty")
    if not old_text:
        raise PatchError("old_text must not be empty")

    p = Path(file_path)
    if repo_root is not None and not p.is_absolute():
        p = Path(repo_root) / p
    p = p.resolve()

    original, err = _read(p, encoding)
    if err:
        return _failed(p, "", err)

    count = original.count(old_text)
    if count == 0:
        return _failed(p, original, "old_text not found in file")
    if count > 1:
        return _failed(
            p, original,
            f"old_text found {count} times — refusing to patch ambiguous match",
        )

    patched = original.replace(old_text, new_text, 1)

    # Safety: confirm the replacement actually changed the content.
    if patched == original:
        return _failed(p, original, "replacement produced no change")

    try:
        p.write_text(patched, encoding=encoding)
    except PermissionError as exc:
        return _failed(p, original, f"permission denied writing {p}: {exc}")
    except OSError as exc:
        return _failed(p, original, f"cannot write {p}: {exc}")

    diff = _unified_diff(original, patched, p)
    return PatchResult(
        applied=True,
        file_path=p,
        diff=diff,
        original_source=original,
        patched_source=patched,
        failure_reason="",
        lines_changed=_count_changed_lines(diff),
    )


# ---------------------------------------------------------------------------
# Higher-level helper: null-guard insertion
# ---------------------------------------------------------------------------

def apply_null_guard(
    file_path: str | Path,
    field_token: str,
    *,
    repo_root: str | Path | None = None,
    encoding: str = "utf-8",
) -> PatchResult:
    """
    Find the line that uses *field_token* inside ``min(...)`` without a
    null-safety guard and replace it with two lines that guard the field
    first.

    Example — given ``field_token="discount"`` and a file containing::

        discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))

    the result is::

        discount_rate = order.discount or 0.0
        discount_rate = max(0.0, min(discount_rate, self.MAX_DISCOUNT_RATE))

    Only the first matching line is patched.  If no line matches, or more
    than one line matches, ``applied=False`` is returned without touching
    the file.

    Parameters
    ----------
    file_path:
        Path of the file to inspect and patch.
    field_token:
        The bare field name (e.g. ``"discount"``), matched case-sensitively.
    repo_root:
        Optional base directory used to resolve a relative *file_path*.
    encoding:
        File encoding.

    Returns
    -------
    PatchResult
    """
    if not file_path:
        raise PatchError("file_path must not be empty")
    if not field_token:
        raise PatchError("field_token must not be empty")

    p = Path(file_path)
    if repo_root is not None and not p.is_absolute():
        p = Path(repo_root) / p
    p = p.resolve()

    original, err = _read(p, encoding)
    if err:
        return _failed(p, "", err)

    # Find all lines that match the unguarded pattern for this field_token.
    matching_lines: list[tuple[int, re.Match]] = []
    for i, line in enumerate(original.splitlines(keepends=True)):
        m = _UNGUARDED_RE.match(line)
        if m and m.group("field") == field_token:
            matching_lines.append((i, m))

    if not matching_lines:
        return _failed(p, original, f"unguarded pattern for field '{field_token}' not found")

    if len(matching_lines) > 1:
        return _failed(
            p, original,
            f"unguarded pattern for field '{field_token}' found on "
            f"{len(matching_lines)} lines — refusing to patch ambiguous match",
        )

    _lineno, m = matching_lines[0]
    indent = m.group("indent")
    var = m.group("var")
    obj_prefix = m.group("obj_prefix")   # e.g. "order."
    rest = m.group("rest")               # e.g. ", self.MAX_DISCOUNT_RATE)"
    rhs = m.group("rhs")

    # Reconstruct the old single line (exact match for the apply() call).
    old_line = m.group(0)

    # Detect whether the rhs wraps min(...) inside max(0.0, ...).
    # If so the clamp line needs an extra closing ) for the outer max(.
    has_outer_max = rhs.lstrip().startswith("max")

    # Build the two replacement lines.
    # Line 1: guard the field — treat None as 0.0 (no effect).
    guard_line = f"{indent}{var} = {obj_prefix}{field_token} or 0.0\n"
    # Line 2: the original expression but now using the local variable.
    # rest already closes min(...); append ) only when max(...) wraps it.
    close = ")" if has_outer_max else ""
    clamp_line = f"{indent}{var} = max(0.0, min({var}{rest}{close}\n"

    new_text = guard_line + clamp_line

    return apply(
        p,
        old_text=old_line,
        new_text=new_text,
        encoding=encoding,
        # repo_root already absorbed into p
    )


# ---------------------------------------------------------------------------
# Schema bridge
# ---------------------------------------------------------------------------

def build_suggested_fix(patch_result: PatchResult) -> SuggestedFix:
    """
    Convert a :class:`PatchResult` to the
    :class:`~incident_replay.models.schemas.SuggestedFix` schema object
    consumed by :class:`~incident_replay.models.schemas.IncidentReport`.

    When ``patch_result.applied`` is ``True`` the description states what
    was changed and the ``patch`` field contains the unified diff.

    When ``patch_result.applied`` is ``False`` the description states why
    the automatic patch could not be applied and ``patch`` is empty.
    """
    if patch_result.applied:
        description = (
            f"Null-safety guard applied to {patch_result.file_path.name}: "
            f"{patch_result.lines_changed} line(s) changed."
        )
        return SuggestedFix(description=description, patch=patch_result.diff)
    else:
        reason = patch_result.failure_reason or "unknown reason"
        description = (
            f"Automatic patch could not be applied to "
            f"{patch_result.file_path.name}: {reason}. "
            "Apply the fix manually using the diff shown in the report."
        )
        return SuggestedFix(description=description, patch="")
