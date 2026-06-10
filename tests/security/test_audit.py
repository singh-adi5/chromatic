"""Audit log tests.

Specifically, we verify the *negative* property: PII and biometric data
never enter the audit log even if a caller mistakenly tries to pass them.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from chromatic.security import AuditEventType, audit
from chromatic.security.audit import AuditEvent

pytestmark = pytest.mark.security


def test_audit_event_serialises_to_json() -> None:
    event = AuditEvent(
        event_id="11111111-1111-1111-1111-111111111111",
        event_type=AuditEventType.DETECTION_COMPLETED,
        timestamp=1234567890.0,
        principal_id="hashed-abc",
        request_id="r-1",
        outcome="success",
        reason=None,
        metadata={"confidence": 0.87},
    )
    parsed = json.loads(event.to_json())
    assert parsed["event_type"] == "detection.completed"
    assert parsed["principal_id"] == "hashed-abc"
    assert parsed["outcome"] == "success"
    assert parsed["metadata"]["confidence"] == 0.87


def test_audit_emits_to_dedicated_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="chromatic.audit"):
        audit(AuditEventType.DETECTION_COMPLETED, request_id="r-1", outcome="success")
    audit_messages = [r for r in caplog.records if r.name == "chromatic.audit"]
    assert len(audit_messages) == 1
    payload = json.loads(audit_messages[0].getMessage())
    assert payload["event_type"] == "detection.completed"


def test_audit_does_not_accept_frame_data() -> None:
    """The `AuditEvent` dataclass cannot carry an ndarray field — verified by
    construction: passing one as `metadata` is caller error, but the JSON
    serialisation will fail loudly rather than silently logging bytes.
    """
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    event = AuditEvent(
        event_id="abc",
        event_type=AuditEventType.DETECTION_COMPLETED,
        timestamp=0.0,
        metadata={"frame": arr},   # caller mistake
    )
    with pytest.raises(TypeError):
        event.to_json()


def test_audit_event_field_allowlist() -> None:
    """No new fields may be added to AuditEvent without an explicit code review.

    This test pins the allowed field names; if someone adds a field that
    could carry PII, the test forces the review.
    """
    expected = {
        "event_id", "event_type", "timestamp", "principal_id",
        "request_id", "outcome", "reason", "metadata",
    }
    actual = {f.name for f in AuditEvent.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    assert actual == expected, (
        "AuditEvent fields changed. Review for PII risk and update this test."
    )
