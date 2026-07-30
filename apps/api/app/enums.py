from enum import StrEnum


class InspectionStatus(StrEnum):
    QUEUED = "queued"
    VISION_ANALYZING = "vision_analyzing"
    REASONING = "reasoning"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    MANUAL_REVIEW = "manual_review"
    FAILED = "failed"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Disposition(StrEnum):
    RELEASE = "release"
    MANUAL_REVIEW = "manual_review"
    REJECT = "reject"
    STOP_LINE = "stop_line"


class ActionType(StrEnum):
    RELEASE_PRODUCT = "release_product"
    MANUAL_REVIEW = "manual_review"
    REJECT_PRODUCT = "reject_product"
    CREATE_TICKET = "create_ticket"
    SEND_NOTIFICATION = "send_notification"
    REQUEST_LINE_STOP = "request_line_stop"
    EXECUTE_LINE_STOP = "execute_line_stop"


class ActionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    FAILED = "failed"
