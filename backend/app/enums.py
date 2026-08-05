from __future__ import annotations

import enum


class QualityResult(str, enum.Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class InspectionStatus(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BatchStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class ReviewTaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"


class HumanDecision(str, enum.Enum):
    """Human review decision (5E). Final quality result is derived separately:
    PASS -> PASS; CONFIRM_DEFECT / CORRECT_DEFECT / OTHER_DEFECT -> FAIL."""

    PASS = "PASS"
    CONFIRM_DEFECT = "CONFIRM_DEFECT"
    CORRECT_DEFECT = "CORRECT_DEFECT"
    OTHER_DEFECT = "OTHER_DEFECT"
