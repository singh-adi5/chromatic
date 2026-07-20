from __future__ import annotations
import logging
from .audit import audit, AuditEventType

logger = logging.getLogger("chromatic.audit")

class AuditLog:
    """Wrapper for emitting audit events"""

    @staticmethod
    def log(event_type: AuditEventType, **kwargs) -> None:
        audit(event_type, **kwargs)
