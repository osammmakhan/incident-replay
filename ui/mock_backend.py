"""
Mock backend for UI development and demo purposes.

Returns a hardcoded IncidentReport so the UI can be built and styled
without a live backend. Swap the import in app.py once the real
replay_incident() is implemented.

Usage:
    from ui.mock_backend import replay_incident
"""

from incident_replay.models.schemas import (
    Evidence,
    IncidentReport,
    RegressionTest,
    SuggestedFix,
    TimelineEvent,
    VerificationResult,
)


def replay_incident(
    incident_description: str,
    repo_path: str,
    log_path: str = "",
    stack_trace: str = "",
) -> IncidentReport:
    """Return a canned IncidentReport for UI development and demo use."""

    return IncidentReport(
        incident_summary=(
            "Checkout endpoint returns HTTP 500 for all requests where a discount "
            "code is applied. The defect is a missing null-guard in apply_discount() "
            "introduced in commit d4e8f21, which reached production at 09:14 UTC. "
            "All affected requests fail with TypeError: unsupported operand type(s) "
            "for *: 'float' and 'NoneType'."
        ),
        timeline=[
            TimelineEvent(timestamp="09:11 UTC", description="Commit d4e8f21 merged to main: add promotional discount support."),
            TimelineEvent(timestamp="09:14 UTC", description="Deployment pipeline completes. New build live in production."),
            TimelineEvent(timestamp="09:17 UTC", description="Error rate on POST /api/checkout rises to 34%. Alerts fire."),
            TimelineEvent(timestamp="09:21 UTC", description="On-call engineer acknowledges. Triage begins."),
            TimelineEvent(timestamp="09:28 UTC", description="Root cause isolated to apply_discount() null path. Hotfix prepared."),
            TimelineEvent(timestamp="09:35 UTC", description="Hotfix deployed. Error rate returns to 0%."),
        ],
        root_cause=(
            "apply_discount() in payment/discount.py does not guard against a None "
            "discount value. When a checkout request carries no active promo code the "
            "discount field is None, causing `total * (1 - None)` to raise a TypeError "
            "that propagates uncaught through the API layer and returns HTTP 500."
        ),
        evidence=[
            Evidence(
                source="log",
                location="demo_project/logs/error.log:142",
                observation="TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
                relevance="Confirms the crash originates when discount is None, matching the reported HTTP 500.",
            ),
            Evidence(
                source="git",
                location="commit d4e8f21 — payment/discount.py",
                observation="apply_discount() added in this commit with no null-guard on the discount parameter.",
                relevance="The defect was introduced here; no defensive check was added before the arithmetic.",
            ),
            Evidence(
                source="code",
                location="payment/discount.py:12 — apply_discount()",
                observation="return total * (1 - discount)  # discount can be None when no promo is applied",
                relevance="Direct site of the TypeError; multiplying a float by None raises at runtime.",
            ),
            Evidence(
                source="test",
                location="tests/test_checkout.py",
                observation="No existing test passes discount=None to apply_discount() or the checkout endpoint.",
                relevance="Absence of this test case allowed the defect to reach production undetected.",
            ),
        ],
        affected_files=["payment/discount.py", "api/checkout.py"],
        affected_functions=["apply_discount", "post"],
        suspicious_commit="d4e8f21",
        confidence=0.94,
        regression_test=RegressionTest(
            name="test_apply_discount_null_discount",
            language="python",
            code=(
                "import pytest\n"
                "from payment.discount import apply_discount\n\n"
                "def test_apply_discount_null_discount():\n"
                "    \"\"\"apply_discount must raise ValueError when discount is None.\"\"\"\n"
                "    with pytest.raises(ValueError, match='discount must not be None'):\n"
                "        apply_discount(total=99.99, discount=None)\n"
            ),
        ),
        suggested_fix=SuggestedFix(
            description=(
                "Add a None-guard at the top of apply_discount() that raises ValueError "
                "before the arithmetic executes. Callers should pass 0.0 for no discount "
                "rather than None."
            ),
            patch=(
                "--- a/payment/discount.py\n"
                "+++ b/payment/discount.py\n"
                "@@ -9,6 +9,9 @@\n"
                " def apply_discount(total: float, discount) -> float:\n"
                "+    if discount is None:\n"
                "+        raise ValueError('discount must not be None; pass 0.0 for no discount')\n"
                "     return total * (1 - discount)\n"
            ),
        ),
        verification_result=VerificationResult(
            passed=True,
            output=(
                "============================= test session starts ==============================\n"
                "collected 1 item\n\n"
                "tests/test_checkout.py::test_apply_discount_null_discount PASSED        [100%]\n\n"
                "============================== 1 passed in 0.08s ==============================="
            ),
        ),
    )
