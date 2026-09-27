import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from pathlib import Path
from incident_replay.agents import log_agent, git_agent, code_agent, synthesis_agent, test_agent
from incident_replay.analysis import evidence as evidence_module
from incident_replay.orchestrator import (
    _keywords_from_log, _affected_files_from_stack, _field_token_from_synthesis,
    _git_restore_file, _test_root_from_repo, _regression_test_dir,
    _incident_field_values_from_log, _dominant_error_type, _read_log_lines,
    _empty_synthesis_result, _make_empty_regression_test,
)
from incident_replay.execution import patcher, test_generator, test_runner
from incident_replay.analysis import timeline as timeline_module

demo = Path("tests/demo_project")
repo_path = str(demo)
log_path = str(demo / "logs" / "production.log")
stack_trace = (demo / "incident" / "stacktrace.txt").read_text()

log_findings = log_agent.run(log_path, stack_trace=stack_trace)
error_kw = _keywords_from_log(log_findings)
affected_files_hint = _affected_files_from_stack(stack_trace)
git_findings = git_agent.run(repo_path=repo_path, error_keywords=error_kw, affected_files=affected_files_hint)
code_findings = code_agent.run(repo_path=repo_path, stack_trace=stack_trace, error_keywords=error_kw)
all_evidence = evidence_module.collect(log_findings=log_findings, git_findings=git_findings, code_findings=code_findings)
synthesis_result = synthesis_agent.run(log_findings=log_findings, git_findings=git_findings, code_findings=code_findings, evidence=all_evidence)

print("synthesis affected_files:", synthesis_result.affected_files)
field_token = _field_token_from_synthesis(synthesis_result)
print("field_token:", field_token)

# Pre-restore
if synthesis_result.affected_files and field_token:
    print("pre-restoring:", synthesis_result.affected_files[0])
    _git_restore_file(synthesis_result.affected_files[0], repo_path)
    from pathlib import Path as P
    f = (P(repo_path) / synthesis_result.affected_files[0]).resolve()
    print("file after restore:", "BUGGY" if "or 0.0" not in f.read_text() else "FIXED")

# Patch
pr = patcher.apply_null_guard(
    file_path=synthesis_result.affected_files[0],
    field_token=field_token,
    repo_root=repo_path,
)
print("patch applied:", pr.applied, "| reason:", pr.failure_reason)
