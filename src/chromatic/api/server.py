"""
FastAPI server for Chromatic.

This module is intentionally thin: it wires HTTP/WebSocket transport to the
`LivenessDetector`, applies authentication and rate limiting, and translates
internal exceptions into the public error contract documented in API.md.

All security-relevant decisions (validation, audit, fail-closed responses)
happen here at the boundary, not in the detection core.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse

from chromatic import LivenessDetector, __version__
from chromatic.config import Settings, configure_logging, load_settings
from chromatic.exceptions import (
    AuthenticationError,
    FrameTooLargeError,
    InvalidFrameError,
    RateLimitExceededError,
)
from chromatic.security import AuditEventType, TokenBucketRateLimiter, audit

try:
    import jwt as pyjwt
except ImportError:  # pragma: no cover - api extras are optional
    pyjwt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Content types we accept on /detect.
_ALLOWED_MIME = {"image/jpeg", "image/png"}


# --- Lifespan: load settings, rate limiter, etc. ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(settings)
    app.state.settings = settings
    app.state.rate_limiter = TokenBucketRateLimiter(settings.rate_limit_per_minute)
    app.state.audit_salt = os.environ.get("CHROMATIC_AUDIT_HASH_SALT", "").encode()
    app.state.jwt_public_key = _load_jwt_public_key()
    app.state.auth_mode = os.environ.get("CHROMATIC_AUTH_MODE", "jwt")
    app.state.ready = False
    # Warm up a detector to load MediaPipe model so /ready becomes true.
    try:
        warm = LivenessDetector(settings)
        warm.close()
        app.state.ready = True
        logger.info("chromatic ready (version %s)", __version__)
    except Exception:  # pragma: no cover
        logger.exception("startup warm-up failed")
        raise
    try:
        yield
    finally:
        app.state.ready = False


def _load_jwt_public_key() -> str | None:
    path = os.environ.get("CHROMATIC_JWT_PUBLIC_KEY_PATH")
    inline = os.environ.get("CHROMATIC_JWT_PUBLIC_KEY")
    if inline:
        return inline
    if path:
        with Path(path).open(encoding="utf-8") as fh:
            return fh.read()
    return None


app = FastAPI(
    title="Chromatic",
    version=__version__,
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)


# --- Auth dependency --------------------------------------------------------

def _hash_principal(sub: str, salt: bytes) -> str:
    """BLAKE2b-160 hash of the principal — never log the raw `sub`."""
    return hashlib.blake2b(sub.encode(), digest_size=20, salt=salt[:16].ljust(16, b"0")).hexdigest()


async def authenticated_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_authenticated_principal: Annotated[str | None, Header()] = None,
) -> str:
    """Verify the caller and return the *hashed* principal ID.

    Returns a hashed ID — the raw `sub` is intentionally not propagated past
    this function, so downstream code cannot accidentally log it.
    """
    auth_mode: str = request.app.state.auth_mode
    salt: bytes = request.app.state.audit_salt

    if auth_mode == "trust_proxy":
        if not x_authenticated_principal:
            raise HTTPException(401, {"error": "unauthorized"})
        return _hash_principal(x_authenticated_principal, salt)

    if pyjwt is None:  # pragma: no cover
        raise HTTPException(501, {"error": "auth_not_configured"})

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, {"error": "unauthorized"})
    token = authorization.split(" ", 1)[1]

    public_key = request.app.state.jwt_public_key
    if not public_key:
        raise HTTPException(500, {"error": "auth_not_configured"})

    try:
        claims = pyjwt.decode(
            token,
            public_key,
            algorithms=["RS256", "ES256"],
            audience="chromatic",
            options={"require": ["iss", "aud", "exp", "iat", "sub"]},
        )
    except Exception:
        audit(AuditEventType.AUTH_FAILED, outcome="failure")
        raise HTTPException(401, {"error": "unauthorized"}) from None

    sub = str(claims["sub"])
    principal = _hash_principal(sub, salt)
    audit(AuditEventType.AUTH_SUCCEEDED, principal_id=principal)
    return principal


def _apply_rate_limit(request: Request, principal: str) -> None:
    limiter: TokenBucketRateLimiter = request.app.state.rate_limiter
    try:
        limiter.acquire(principal)
    except RateLimitExceededError:
        audit(AuditEventType.RATE_LIMIT_HIT, principal_id=principal, outcome="failure")
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limited"},
            headers={"Retry-After": "60"},
        ) from None


# --- Routes -----------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/ready")
async def ready(request: Request) -> Response:
    if not request.app.state.ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready"})


@app.post("/v1/detect")
async def detect(
    request: Request,
    principal: Annotated[str, Depends(authenticated_principal)],
    content_type: Annotated[str | None, Header()] = None,
    traceparent: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """Single-frame detection. Stateless — for streaming use the WS endpoint."""
    settings: Settings = request.app.state.settings
    request_id = traceparent or str(uuid.uuid4())

    _apply_rate_limit(request, principal)

    if content_type not in _ALLOWED_MIME:
        return _error(415, "unsupported_media_type", request_id)

    body = await request.body()
    if len(body) > settings.max_frame_bytes:
        audit(
            AuditEventType.INPUT_REJECTED,
            principal_id=principal, request_id=request_id, outcome="failure",
            reason="payload_too_large",
        )
        return _error(413, "payload_too_large", request_id)

    arr = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return _error(422, "decode_failed", request_id)

    try:
        # Single-frame mode: a fresh detector. The verdict cannot reach
        # is_live=True (sustained-frames requirement), but the layer scores
        # are useful for early-return UI on the client side.
        with LivenessDetector(settings) as detector:
            diag = detector.process_frame(frame)
    except (InvalidFrameError, FrameTooLargeError) as exc:
        audit(
            AuditEventType.INPUT_REJECTED,
            principal_id=principal, request_id=request_id, outcome="failure",
            reason=type(exc).__name__,
        )
        return _error(400, "invalid_input", request_id)
    except AuthenticationError:
        return _error(401, "unauthorized", request_id)
    except Exception:
        logger.exception("detection failed (request_id=%s)", request_id)
        audit(
            AuditEventType.DETECTION_FAILED,
            principal_id=principal, request_id=request_id, outcome="failure",
        )
        return _error(500, "internal_error", request_id)

    body_out: dict[str, Any] = {
        "is_live": diag.verdict.is_live,
        "confidence": round(diag.verdict.confidence, 4),
        "sustained_confidence": None,  # not meaningful in stateless mode
        "layer_scores": diag.verdict.layer_scores.as_dict(),
        "reasons": diag.verdict.reasons,
        "request_id": request_id,
    }
    return JSONResponse(body_out)


@app.websocket("/v1/ws/stream")
async def stream(websocket: WebSocket) -> None:  # pragma: no cover - WS hard to unit-test
    """Streaming detection — keeps detector state across frames."""
    await websocket.accept()
    settings: Settings = websocket.app.state.settings
    salt: bytes = websocket.app.state.audit_salt

    # Auth is performed via subprotocol header on connect.
    auth_header = websocket.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        await websocket.close(code=4401)
        return

    if pyjwt is None or not websocket.app.state.jwt_public_key:
        await websocket.close(code=4500)
        return

    try:
        claims = pyjwt.decode(
            auth_header.split(" ", 1)[1],
            websocket.app.state.jwt_public_key,
            algorithms=["RS256", "ES256"],
            audience="chromatic",
            options={"require": ["iss", "aud", "exp", "iat", "sub"]},
        )
    except Exception:
        await websocket.close(code=4401)
        return

    principal = _hash_principal(str(claims["sub"]), salt)
    detector = LivenessDetector(settings)
    session_started = time.monotonic()
    frame_count = 0

    try:
        # First message must be the init control.
        init = await websocket.receive_json()
        if init.get("type") != "init":
            await websocket.close(code=4400)
            return

        while True:
            # Enforce a max-session duration.
            if time.monotonic() - session_started > 60.0:
                await websocket.send_json({"type": "session_timeout"})
                break

            try:
                buf = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "idle_timeout"})
                break

            if len(buf) > settings.max_frame_bytes:
                await websocket.send_json({"type": "error", "error": "payload_too_large"})
                continue

            arr = np.frombuffer(buf, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"type": "error", "error": "decode_failed"})
                continue

            try:
                diag = detector.process_frame(frame)
            except (InvalidFrameError, FrameTooLargeError):
                await websocket.send_json({"type": "error", "error": "invalid_input"})
                continue

            frame_count += 1
            payload: dict[str, Any] = {
                "type": "diagnostic",
                "is_live": diag.verdict.is_live,
                "confidence": round(diag.verdict.confidence, 4),
                "sustained_confidence": round(diag.verdict.sustained_confidence, 4),
                "calibration_progress": round(diag.calibration_progress, 3),
                "rppg_progress": round(diag.rppg_progress, 3),
                "layer_scores": diag.verdict.layer_scores.as_dict(),
                "reasons": diag.verdict.reasons,
            }
            if diag.pulse is not None:
                payload["pulse"] = {
                    "bpm": round(diag.pulse.bpm, 1),
                    "snr_db": round(diag.pulse.snr_db, 1),
                }
            await websocket.send_json(payload)

            if diag.verdict.is_live:
                await websocket.send_json(
                    {
                        "type": "final",
                        "is_live": True,
                        "confidence": round(diag.verdict.confidence, 4),
                        "sustained_confidence": round(diag.verdict.sustained_confidence, 4),
                        "layer_scores": diag.verdict.layer_scores.as_dict(),
                        "reasons": diag.verdict.reasons,
                    }
                )
                break
    except WebSocketDisconnect:
        pass
    finally:
        detector.close()
        audit(
            AuditEventType.DETECTION_COMPLETED,
            principal_id=principal, outcome="success", frames=frame_count,
        )


# --- Helpers ----------------------------------------------------------------

def _error(status_code: int, code: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"error": code, "request_id": request_id, "detail": code},
        status_code=status_code,
    )
