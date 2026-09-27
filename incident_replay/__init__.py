"""
incident_replay — public package surface.

The UI and all external callers should import from this package or from
``incident_replay.api`` directly.  Internal sub-modules (agents, execution,
analysis) are implementation details and subject to change.
"""

from incident_replay.api import (
    OrchestratorError,
    apply_suggested_fix,
    get_regression_test,
    get_verification_result,
    replay_incident,
    run_regression_test,
    run_regression_test_after_fix,
)
from incident_replay.models.schemas import (
    Evidence,
    IncidentReport,
    RegressionTest,
    SuggestedFix,
    TimelineEvent,
    VerificationResult,
)

__all__ = [
    # Pipeline entry point
    "replay_incident",
    "OrchestratorError",
    # Proof-workflow helpers
    "get_regression_test",
    "run_regression_test",
    "apply_suggested_fix",
    "run_regression_test_after_fix",
    "get_verification_result",
    # Schema types
    "IncidentReport",
    "RegressionTest",
    "SuggestedFix",
    "VerificationResult",
    "Evidence",
    "TimelineEvent",
]
