"""
Incident Replay orchestration layer.

Public API
----------
``replay_incident(incident_description, repo_path, log_path, stack_trace)``
    → :class:`~incident_replay.models.schemas.IncidentReport`

    Run all four investigator agents, build a timeline, synthesize evidence,
    calculate confidence, generate a regression test, attempt an automatic
    patch, and return a single, fully-populated ``IncidentReport`` that
    conforms exactly to :mod:`incident_replay.models.schemas`.

Design rules
------------
No Streamlit in this module
    The orchestrator is a pure backend function.  The UI imports and calls
    ``replay_incident``; it never contains investigation logic itself.

No silent failures
    When an investigator raises an unexpected exception it is caught,
    wrapped in a human-readable error message, and surfaced in the report
    (via ``incident_summary`` or the relevant finding list) rather than
    silently swallowed.  The pipeline always produces a structured
    ``IncidentReport`` even when every investigator fails.

Agent isolation
    Each investigator runs in its own try/except block.  A failure in the
    log agent does not prevent the git agent from running.

Minimal coupling
    The orchestrator imports findings types only to pass them between
    pipeline stages.  It does not duplicate any logic that already lives
    in an agent or analysis module.
"""

from __future__ import annotations

import os
from typing import Optional

from incident_replay.agents import code_agent, git_agent, log_agent, synthesis_agent, test_agent
from incident_replay.analysis import evidence as evidence_module
from incident_replay.analysis import timeline as timeline_module
from incident_replay.execution import patcher, test_generator, test_runner
from incident_replay.models.schemas import (
    Evidence,
    IncidentReport,
    RegressionTest,
    SuggestedFix,
    TimelineEvent,
    VerificationResult,
)

__all__ = ["OrchestratorError", "replay_incident"]


# ---------------------------------------------------------------------------
# Public error type
# ---------------------------------------------------------------------------

class OrchestratorError(RuntimeError):
    """
    Raised only for structurally invalid arguments (e.g. both *repo_path*
    and *log_path* are empty).

    Individual investigator failures are *not* raised — they are embedded
    in the returned :class:`~incident_replay.models.schemas.IncidentReport`
    so the UI can display partial results rather than a hard crash.
    """


# ---------------------------------------------------------------------------
# Investigator failure placeholder helpers
# ---------------------------------------------------------------------------

def _failure_evidence(source: str, location: str, reason: str) -> Evidence:
    """Build a single evidence entry that records an investigator failure."""
    return Evidence(
        source=source,  # type: ignore[arg-type]
        location=location,
        observation=f"Investigator failed: {reason}",
        relevance="unavailable — investigator did not complete",
    )


def _failure_timeline_event(investigator: str, reason: str) -> TimelineEvent:
    return TimelineEvent(
        timestamp="(time unknown)",
        event="error",
        description=f"{investigator} investigator failed: {reason}",
    )


# ---------------------------------------------------------------------------
# Keyword / hint extraction helpers
# ---------------------------------------------------------------------------

def _keywords_from_log(log_findings) -> list[str]:
    """Extract error-type keywords from log findings for downstream agents."""
    if log_findings is None:
        return []
    kw: list[str] = []
    sig = getattr(log_findings, "dominant_error_signature", None) or ""
    if sig:
        # "TypeError: unsupported operand" → ["TypeError", "operand"]
        for part in sig.replace(":", " ").split():
            if len(part) > 3:
                kw.append(part)
    for pat in getattr(log_findings, "suspicious_input_patterns", []) or []:
        # "discount=null" → "discount"
        field = pat.split("=")[0]
        if field:
            kw.append(field)
    return list(dict.fromkeys(kw))  # deduplicate preserving order


def _affected_files_from_stack(stack_trace: Optional[str]) -> list[str]:
    """Pull bare file paths from a stack trace for the git agent."""
    if not stack_trace:
        return []
    import re
    return re.findall(r'File "([^"]+)"', stack_trace)


def _test_root_from_repo(repo_path: str) -> Optional[str]:
    """Return the first existing tests/ or test/ directory under *repo_path*."""
    for candidate in ("tests", "test"):
        full = os.path.join(repo_path, candidate)
        if os.path.isdir(full):
            return full
    return None


def _incident_field_values_from_log(log_findings) -> dict[str, str]:
    """Build ``{field: value}`` from suspicious input patterns."""
    result: dict[str, str] = {}
    for pat in getattr(log_findings, "suspicious_input_patterns", []) or []:
        if "=" in pat:
            k, v = pat.split("=", 1)
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Schema assembly helpers
# ---------------------------------------------------------------------------

def _make_empty_regression_test(reason: str) -> RegressionTest:
    return RegressionTest(
        name="test_regression_unavailable",
        code=f"# Regression test could not be generated: {reason}\n",
        language="python",
    )


def _make_empty_suggested_fix(reason: str) -> SuggestedFix:
    return SuggestedFix(
        description=f"Automatic fix unavailable: {reason}",
        patch="",
    )


def _make_empty_verification(reason: str) -> VerificationResult:
    return VerificationResult(
        passed=False,
        output=f"Verification skipped: {reason}",
    )


def _incident_summary(
    synthesis_result,
    error_notes: list[str],
) -> str:
    """Compose the incident_summary field from synthesis + any error notes."""
    parts: list[str] = []
    root = getattr(synthesis_result, "root_cause", "") or ""
    if root:
        parts.append(root)
    if error_notes:
        parts.append("Investigator errors: " + "; ".join(error_notes))
    return "\n\n".join(parts) if parts else "Incident analysis could not be completed."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def replay_incident(
    incident_description: str,
    repo_path: str,
    log_path: str,
    stack_trace: Optional[str] = None,
) -> IncidentReport:
    """Run all investigators and return a fully-populated :class:`IncidentReport`.

    Parameters
    ----------
    incident_description:
        Free-text description of the incident provided by the operator.
        Stored in the report and passed as context to synthesis; does not
        drive any agent logic directly.
    repo_path:
        Absolute or relative path to the git repository root to inspect.
    log_path:
        Path to the application log file for the incident window.
    stack_trace:
        Optional raw stack trace text (e.g. from a ``stacktrace.txt``
        file).  When ``None`` the log agent attempts to extract an inline
        traceback from the log.

    Returns
    -------
    IncidentReport
        Always returned, even when every investigator fails.  Failures are
        represented as low-confidence findings with explanatory messages.

    Raises
    ------
    OrchestratorError
        When both *repo_path* and *log_path* are empty/blank — there is
        nothing to investigate.
    """
    # ------------------------------------------------------------------ #
    # 0. Pre-flight checks                                                #
    # ------------------------------------------------------------------ #
    if (not repo_path or not repo_path.strip()) and (not log_path or not log_path.strip()):
        raise OrchestratorError(
            "At least one of repo_path or log_path must be supplied."
        )

    error_notes: list[str] = []
    all_evidence: list[Evidence] = []
    timeline_events: list[TimelineEvent] = []

    log_findings = None
    git_findings = None
    code_findings = None
    test_findings = None

    # ------------------------------------------------------------------ #
    # 1. Log Investigator                                                 #
    # ------------------------------------------------------------------ #
    try:
        log_findings = log_agent.run(log_path or "", stack_trace=stack_trace)
    except Exception as exc:  # noqa: BLE001
        reason = _format_exc(exc)
        error_notes.append(f"LogAgent: {reason}")
        all_evidence.append(_failure_evidence("log", log_path or "<no path>", reason))
        timeline_events.append(_failure_timeline_event("Log", reason))

    # ------------------------------------------------------------------ #
    # 2. Git Investigator                                                 #
    # ------------------------------------------------------------------ #
    error_kw = _keywords_from_log(log_findings)
    affected_files_hint = _affected_files_from_stack(
        stack_trace or (getattr(log_findings, "stack_trace", None) if log_findings else None)
    )
    incident_time = getattr(log_findings, "first_error_time", None) if log_findings else None

    if repo_path and repo_path.strip():
        try:
            git_findings = git_agent.run(
                repo_path=repo_path,
                incident_time=incident_time,
                error_keywords=error_kw,
                affected_files=affected_files_hint,
            )
        except Exception as exc:  # noqa: BLE001
            reason = _format_exc(exc)
            error_notes.append(f"GitAgent: {reason}")
            all_evidence.append(_failure_evidence("git", repo_path, reason))
            timeline_events.append(_failure_timeline_event("Git", reason))

    # ------------------------------------------------------------------ #
    # 3. Code Investigator                                                #
    # ------------------------------------------------------------------ #
    changed_files_hint: list[str] = []
    if git_findings and git_findings.top_suspect:
        changed_files_hint = git_findings.top_suspect.changed_files

    effective_stack = stack_trace or (
        getattr(log_findings, "stack_trace", None) if log_findings else None
    )

    if repo_path and repo_path.strip():
        try:
            code_findings = code_agent.run(
                repo_path=repo_path,
                stack_trace=effective_stack,
                error_keywords=error_kw,
                changed_files=changed_files_hint,
                error_type=_dominant_error_type(log_findings),
            )
        except Exception as exc:  # noqa: BLE001
            reason = _format_exc(exc)
            error_notes.append(f"CodeAgent: {reason}")
            all_evidence.append(_failure_evidence("code", repo_path, reason))
            timeline_events.append(_failure_timeline_event("Code", reason))

    # ------------------------------------------------------------------ #
    # 4. Test Investigator                                                #
    # ------------------------------------------------------------------ #
    test_root = _test_root_from_repo(repo_path) if repo_path and repo_path.strip() else None
    affected_funcs_hint: list[str] = []
    if code_findings:
        affected_funcs_hint = code_findings.affected_functions

    if test_root:
        try:
            test_findings = test_agent.run(
                test_root=test_root,
                affected_functions=affected_funcs_hint,
                error_keywords=error_kw,
                incident_field_values=_incident_field_values_from_log(log_findings),
            )
        except Exception as exc:  # noqa: BLE001
            reason = _format_exc(exc)
            error_notes.append(f"TestAgent: {reason}")
            all_evidence.append(_failure_evidence("test", test_root, reason))
            timeline_events.append(_failure_timeline_event("Test", reason))

    # ------------------------------------------------------------------ #
    # 5. Timeline construction                                            #
    # ------------------------------------------------------------------ #
    log_lines: list[str] = _read_log_lines(log_path)
    try:
        built_events = timeline_module.build_from_findings(
            log_findings=log_findings,
            git_findings=git_findings,
            log_lines=log_lines or None,
        )
        timeline_events = built_events + [
            e for e in timeline_events if e not in built_events
        ]
    except Exception as exc:  # noqa: BLE001
        reason = _format_exc(exc)
        error_notes.append(f"Timeline: {reason}")
        # Keep whatever failure events were already added

    # ------------------------------------------------------------------ #
    # 6. Evidence normalization                                           #
    # ------------------------------------------------------------------ #
    try:
        collected = evidence_module.collect(
            log_findings=log_findings,
            git_findings=git_findings,
            code_findings=code_findings,
            test_findings=test_findings,
        )
        all_evidence = collected + [
            e for e in all_evidence if e not in collected
        ]
    except Exception as exc:  # noqa: BLE001
        reason = _format_exc(exc)
        error_notes.append(f"Evidence: {reason}")

    # ------------------------------------------------------------------ #
    # 7. Evidence synthesis                                               #
    # ------------------------------------------------------------------ #
    try:
        synthesis_result = synthesis_agent.run(
            log_findings=log_findings,
            git_findings=git_findings,
            code_findings=code_findings,
            test_findings=test_findings,
            evidence=all_evidence or None,
            llm_backend=_make_llm_backend(),
        )
    except Exception as exc:  # noqa: BLE001
        reason = _format_exc(exc)
        error_notes.append(f"Synthesis: {reason}")
        # Fall back to a zero-confidence placeholder
        synthesis_result = _empty_synthesis_result(reason)

    # ------------------------------------------------------------------ #
    # 8. Regression test generation                                       #
    # ------------------------------------------------------------------ #
    regression_test: RegressionTest
    verification_result: VerificationResult
    suggested_fix: SuggestedFix

    # The written path is needed for the after-fix re-run in step 9.
    _generated_test_path = None
    _project_cwd_for_test = None

    gen_output_dir = _regression_test_dir(repo_path)
    if gen_output_dir:
        try:
            _project_cwd_for_test = (
                repo_path if repo_path and repo_path.strip() else gen_output_dir
            )
            generated = test_generator.generate(
                synthesis_result,
                output_dir=gen_output_dir,
                run_against_project=True,
                project_cwd=_project_cwd_for_test,
            )
            regression_test = generated.regression_test
            _generated_test_path = generated.written_path
            # Record the before-fix outcome for transparency; the final
            # VerificationResult will be updated after the patch is applied.
            _before_fix_output = generated.run_output
            _before_fix_failing = generated.initially_failing  # True = good
        except test_generator.GeneratorError as exc:
            regression_test = _make_empty_regression_test(str(exc))
            _before_fix_output = str(exc)
            _before_fix_failing = None
        except Exception as exc:  # noqa: BLE001
            reason = _format_exc(exc)
            error_notes.append(f"TestGenerator: {reason}")
            regression_test = _make_empty_regression_test(reason)
            _before_fix_output = reason
            _before_fix_failing = None
    else:
        msg = "no repo_path supplied; cannot determine output directory"
        regression_test = _make_empty_regression_test(msg)
        _before_fix_output = msg
        _before_fix_failing = None

    # ------------------------------------------------------------------ #
    # 9. Suggested fix (null-guard patch)                                 #
    # ------------------------------------------------------------------ #
    affected_files = synthesis_result.affected_files
    field_token = _field_token_from_synthesis(synthesis_result)

    if affected_files and field_token and repo_path and repo_path.strip():
        # Restore the target file to its committed (pre-fix) state so that
        # repeated pipeline runs are idempotent.  If the file was already
        # patched by a previous run, the patcher would report "pattern not
        # found" and set verification_result.passed=False, which would be
        # a false negative.  The git restore is a silent best-effort; if it
        # fails (no git, no commit) the patcher's own logic handles it.
        _git_restore_file(affected_files[0], repo_path)
        patch_result = patcher.apply_null_guard(
            file_path=affected_files[0],
            field_token=field_token,
            repo_root=repo_path,
        )
        suggested_fix = patcher.build_suggested_fix(patch_result)
    else:
        missing = []
        if not affected_files:
            missing.append("no affected file identified")
        if not field_token:
            missing.append("no incident field token identified")
        if not (repo_path and repo_path.strip()):
            missing.append("no repo_path supplied")
        suggested_fix = _make_empty_suggested_fix("; ".join(missing))
        patch_result = None

    # ------------------------------------------------------------------ #
    # 9b. Re-run the regression test after the patch                      #
    # Verification = "did the fix make the test pass?"                    #
    # ------------------------------------------------------------------ #
    if _generated_test_path is not None and _before_fix_failing is not None:
        # Only re-run when: (a) we have a real test file, AND
        # (b) the before-fix run actually exercised the code (not skipped).
        patch_was_applied = (
            patch_result is not None
            and getattr(patch_result, "applied", False)
        )
        if patch_was_applied:
            try:
                after_run = test_runner.run_test_file(
                    _generated_test_path,
                    cwd=_project_cwd_for_test,
                    phase="after",
                )
                verification_result = test_runner.to_verification_result(after_run)
            except Exception as exc:  # noqa: BLE001
                reason = _format_exc(exc)
                error_notes.append(f"AfterFixRun: {reason}")
                verification_result = _make_empty_verification(
                    f"after-fix re-run failed: {reason}"
                )
        else:
            # Patch was not applied — the unguarded pattern was not found in
            # the file (already fixed, or the synthesis picked the wrong file).
            # We cannot verify the fix, so passed=False to avoid a false PASS.
            verification_result = VerificationResult(
                passed=False,
                output=(
                    "[patch not applied — fix could not be verified] "
                    + (_before_fix_output or "no output")
                ),
            )
    else:
        # No test was generated or the before-fix run was skipped.
        verification_result = _make_empty_verification(
            _before_fix_output or "regression test was not generated"
        )

    # ------------------------------------------------------------------ #
    # 10. Assemble IncidentReport                                         #
    # ------------------------------------------------------------------ #
    suspicious_commit: Optional[str] = synthesis_result.suspicious_commit
    supporting_evidence: list[Evidence] = synthesis_result.supporting_evidence or []

    return IncidentReport(
        incident_summary=_incident_summary(synthesis_result, error_notes),
        timeline=timeline_events,
        root_cause=synthesis_result.root_cause,
        evidence=supporting_evidence or all_evidence,
        affected_files=synthesis_result.affected_files,
        affected_functions=synthesis_result.affected_functions,
        suspicious_commit=suspicious_commit,
        confidence=synthesis_result.confidence,
        regression_test=regression_test,
        suggested_fix=suggested_fix,
        verification_result=verification_result,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_restore_file(file_path: str, repo_root: str) -> None:
    """
    Restore *file_path* to its HEAD-committed state using ``git checkout``.

    This makes repeated pipeline runs idempotent: if a previous run already
    patched the file, this call reverts it so the patcher finds the unguarded
    pattern again.

    Failures are silenced — the patcher's own "pattern not found" logic will
    handle the case where the file cannot be restored.

    Parameters
    ----------
    file_path:
        Path to the file, resolved relative to *repo_root*.
    repo_root:
        Root of the git repository.
    """
    import subprocess
    from pathlib import Path as _Path

    try:
        abs_file = _Path(file_path)
        if not abs_file.is_absolute():
            abs_file = (_Path(repo_root) / file_path).resolve()
        repo = _Path(repo_root).resolve()
        rel = abs_file.relative_to(repo)
        subprocess.run(
            ["git", "checkout", "HEAD", "--", rel.as_posix()],
            cwd=str(repo),
            capture_output=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001
        pass  # silent — patcher will handle unrestorable state


def _make_llm_backend():
    """
    Return a Groq-backed llm_backend callable when GROQ_API_KEY is set,
    or ``None`` to keep the fully deterministic pipeline.

    The returned callable is passed to ``synthesis_agent.run()`` as
    ``llm_backend``.  It receives the structured evidence list and a context
    dict already assembled by the synthesis stage, so the model is given
    exactly the facts the deterministic pipeline found — nothing more.

    Model used: ``llama-3.3-70b-versatile`` (fast, free tier available on Groq).
    Override with the GROQ_MODEL environment variable.

    Safety guarantees (enforced by synthesis_agent, not here):
    - Every file, function, and commit the model names is checked against
      supporting evidence; untraceable claims are silently dropped.
    - Confidence is always computed from evidence alone; the model cannot
      inflate it.
    - If the call raises, times out, or returns unusable output, the
      deterministic hypothesis is used instead.
    """
    import os
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None  # no key → deterministic mode, no network call

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    def _backend(evidence_list, context: dict):
        try:
            from groq import Groq  # type: ignore[import]
        except ImportError:
            return None  # groq package not installed → fall back silently

        client = Groq(api_key=api_key)

        evidence_text = "\n".join(
            f"[{getattr(e, 'source', '?')}] {getattr(e, 'location', '')} — "
            f"{getattr(e, 'observation', '')}"
            for e in (evidence_list or [])
        )

        system_prompt = (
            "You are a senior reliability engineer performing root-cause analysis.\n"
            "You will be given structured evidence collected by automated agents "
            "(log analysis, git history, code inspection, test coverage).\n"
            "Your job is to write a clear, concise root_cause sentence and identify "
            "the affected files, functions, and the suspicious commit.\n\n"
            "Rules:\n"
            "1. Only name files, functions, and commits that appear in the evidence.\n"
            "2. The root_cause must start with 'Hypothesis: '.\n"
            "3. Be specific: name the field, the null/None value, and the code path.\n"
            "4. Return valid JSON only — no markdown fences, no extra keys."
        )

        user_prompt = (
            f"Evidence:\n{evidence_text or '(none)'}\n\n"
            f"Context:\n"
            f"  error_signature: {context.get('error_signature', '')}\n"
            f"  affected_files: {context.get('affected_files', [])}\n"
            f"  affected_functions: {context.get('affected_functions', [])}\n"
            f"  suspicious_commit: {context.get('suspicious_commit')}\n"
            f"  missing_scenarios: {context.get('missing_scenarios', [])}\n\n"
            "Return a JSON object with exactly these keys:\n"
            '  "root_cause": "<string starting with Hypothesis:>",\n'
            '  "affected_files": ["<file>", ...],\n'
            '  "affected_functions": ["<function>", ...],\n'
            '  "suspicious_commit": "<sha or null>"\n'
        )

        import json as _json
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=30,
        )
        return _json.loads(response.choices[0].message.content)

    return _backend


def _format_exc(exc: Exception) -> str:
    """Return a compact one-line error description."""
    return f"{type(exc).__name__}: {exc}"


def _dominant_error_type(log_findings) -> Optional[str]:
    if log_findings is None:
        return None
    sig = getattr(log_findings, "dominant_error_signature", None) or ""
    if ":" in sig:
        return sig.split(":")[0].strip()
    return None


def _read_log_lines(log_path: str) -> list[str]:
    """Read raw lines from the log file; return [] on any error."""
    if not log_path or not log_path.strip():
        return []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            return fh.readlines()
    except OSError:
        return []


def _regression_test_dir(repo_path: str) -> Optional[str]:
    """Return a writable directory for generated regression tests."""
    if not repo_path or not repo_path.strip():
        return None
    test_root = _test_root_from_repo(repo_path)
    if test_root:
        return os.path.join(test_root, "regression")
    return os.path.join(repo_path, "tests", "regression")


def _field_token_from_synthesis(synthesis_result) -> Optional[str]:
    """Extract the incident field token from the root_cause text."""
    import re
    text = getattr(synthesis_result, "root_cause", "") or ""
    # "field 'discount' supplied as null"
    m = re.search(r"field\s+['\"]([A-Za-z_]\w*)['\"]", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # "discount=null"
    m = re.search(
        r"\b([A-Za-z_]\w*)\s*=\s*(?:null|none|nil)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return None


def _empty_synthesis_result(reason: str):
    """Return a zero-confidence SynthesisResult placeholder."""
    from incident_replay.agents.synthesis_agent import SynthesisResult
    return SynthesisResult(
        root_cause=f"Synthesis failed: {reason}",
        affected_files=[],
        affected_functions=[],
        suspicious_commit=None,
        supporting_evidence=[],
        conflicting_evidence=[],
        confidence=0.0,
        confidence_reasons=[f"Synthesis stage failed: {reason}"],
        uncertainty=["All findings are unavailable due to synthesis failure."],
        used_llm=False,
    )
