"""Policy stubs describing risk outcomes."""

from core.domain.decision import Decision

POLICY_NOTES = {
    Decision.PASS: "Prescription cleared with no regulatory or safety blocks.",
    Decision.WARN: "Requires pharmacist/admin review before dispensing.",
    Decision.FAIL: "Do not dispense; contains banned or unsafe items.",
}

__all__ = ["Decision", "POLICY_NOTES"]
