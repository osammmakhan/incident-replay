"""
File inspection utilities for incident-replay agents.

Provides safe, deterministic helpers for reading source files and
extracting relevant sections by line range or function name.
No UI dependencies.  All I/O errors are surfaced as typed exceptions.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class FileReadError(OSError):
    """Raised when a file cannot be read (missing, permission denied, etc.)."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileSection:
    """A contiguous slice of a source file."""

    path: str
    start_line: int   # 1-based, inclusive
    end_line: int     # 1-based, inclusive
    content: str


@dataclass
class FunctionInfo:
    """Location and source of a top-level or class-level function/method."""

    name: str
    qualified_name: str   # e.g. "ClassName.method_name"
    start_line: int
    end_line: int
    source: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Return the full text of *path*.

    Raises :class:`FileReadError` if the file does not exist or cannot be
    read.  Normalises line endings to ``\\n``.
    """
    if not os.path.exists(path):
        raise FileReadError(f"File not found: {path!r}")
    if not os.path.isfile(path):
        raise FileReadError(f"Path is not a regular file: {path!r}")
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            return fh.read().replace("\r\n", "\n").replace("\r", "\n")
    except OSError as exc:
        raise FileReadError(f"Cannot read {path!r}: {exc}") from exc


def read_lines(path: str) -> list[str]:
    """Return the file as a list of lines (newline stripped)."""
    return read_file(path).splitlines()


def read_section(path: str, start_line: int, end_line: int) -> FileSection:
    """Return lines *start_line* to *end_line* (1-based, inclusive).

    Clamps the range to the actual file length rather than raising.
    """
    lines = read_lines(path)
    total = len(lines)
    s = max(1, start_line)
    e = min(total, end_line)
    snippet = "\n".join(lines[s - 1 : e])
    return FileSection(path=path, start_line=s, end_line=e, content=snippet)


def extract_function(path: str, function_name: str) -> Optional[FunctionInfo]:
    """Return the source and line numbers for *function_name* in *path*.

    Searches top-level functions and methods of top-level classes.
    Returns ``None`` if the function is not found.  Returns the *first*
    match when the name is ambiguous.

    Raises :class:`FileReadError` if the file cannot be read.
    Raises :class:`ValueError` if the file cannot be parsed as Python.
    """
    source = read_file(path)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse {path!r} as Python: {exc}") from exc

    lines = source.splitlines()

    def _extract(node: ast.FunctionDef | ast.AsyncFunctionDef,
                 qualifier: str) -> FunctionInfo:
        qualified = f"{qualifier}.{node.name}" if qualifier else node.name
        start = node.lineno
        end = node.end_lineno or start
        body = "\n".join(lines[start - 1 : end])
        return FunctionInfo(
            name=node.name,
            qualified_name=qualified,
            start_line=start,
            end_line=end,
            source=body,
        )

    func_node_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    # Search top-level functions first
    for node in ast.walk(tree):
        if isinstance(node, func_node_types) and node.name == function_name:
            # Determine qualifier (class name if nested under a ClassDef)
            qualifier = _find_class_qualifier(tree, node)
            return _extract(node, qualifier)

    return None


def find_functions_in_file(path: str) -> list[FunctionInfo]:
    """Return all top-level functions and class methods defined in *path*.

    Useful for building a map of the file before targeted extraction.
    Raises :class:`FileReadError` or :class:`ValueError` on bad input.
    """
    source = read_file(path)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse {path!r} as Python: {exc}") from exc

    lines = source.splitlines()
    results: list[FunctionInfo] = []
    func_node_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, func_node_types):
            start, end = node.lineno, node.end_lineno or node.lineno
            results.append(FunctionInfo(
                name=node.name,
                qualified_name=node.name,
                start_line=start,
                end_line=end,
                source="\n".join(lines[start - 1 : end]),
            ))
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, func_node_types):
                    start, end = child.lineno, child.end_lineno or child.lineno
                    qualified = f"{node.name}.{child.name}"
                    results.append(FunctionInfo(
                        name=child.name,
                        qualified_name=qualified,
                        start_line=start,
                        end_line=end,
                        source="\n".join(lines[start - 1 : end]),
                    ))

    return results


def file_exists(path: str) -> bool:
    """Return ``True`` if *path* exists and is a regular file."""
    return os.path.isfile(path)


def safe_read_file(path: str, default: str = "") -> str:
    """Like :func:`read_file` but returns *default* instead of raising."""
    try:
        return read_file(path)
    except FileReadError:
        return default


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_class_qualifier(tree: ast.Module,
                          target: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return the class name if *target* is a direct child of a ClassDef."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if child is target:
                    return node.name
    return ""
