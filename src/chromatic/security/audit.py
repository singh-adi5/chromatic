"""
Security audit logging.

Records security-relevant events without logging PII or biometric data.
Maps to OWASP ASVS 7 (Error Handling and Logging):
- 7.1.1 Do not log credentials or sensitive data
- 7.2.1 Log all authentication decisions
- 7.2.2 Log access control decisions
- 7.3.1 Log integrity events
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("chromatic.audit")


class AuditEventType(str, Enum):
    """Enumeration of auditable events."""

    DETECTION_STARTED = "detection.started"
    DETECTION_COMPLETED = "detection.completed"
    DETECTION_FAILED = "detection.failed"
    INPUT_REJECTED = "input.rejected"
    RATE_LIMIT_HIT = "security.rate_limit_exceeded"
    AUTH_FAILED = "security.auth_failed"
    AUTH_SUCCEEDED = "security.auth_succeeded"
    CONFIG_CHANGED = "system.config_changed"
    CALIBRATION_RESET = "system.calibration_reset"


@dataclass(frozen=True)
class AuditEvent:
    """A single audit event.

    Note the absence of any fields that could carry PII or biometric data.
    `principal_id` is intended to hold an opaque hash of the user identifier,
    never the raw identifier itself.
    """

    event_id: str
    event_type: AuditEventType
    timestamp: float
    principal_id: str | None = None       # hashed identifier only
    request_id: str | None = None
    outcome: str = "success"              # "success" | "failure"
    reason: str | None = None             # generic; never leaks internal state
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return json.dumps(d, separators=(",", ":"), sort_keys=True)


def audit(
    event_type: AuditEventType,
    *,
    principal_id: str | None = None,
    request_id: str | None = None,
    outcome: str = "success",
    reason: str | None = None,
    **metadata: Any,
) -> None:
    """Record a single audit event.

    The logger named "chromatic.audit" should be routed to a dedicated
    handler (e.g., a SIEM forwarder) in production. Operational logs and audit
    logs should be separated to allow different retention and access policies.
    """
    event = AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=time.time(),
        principal_id=principal_id,
        request_id=request_id,
        outcome=outcome,
        reason=reason,
        metadata=metadata,
    )
    logger.info(event.to_json())
