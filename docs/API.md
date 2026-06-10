# REST API

> Audience: client developers integrating Chromatic into a product.

## 1. Base URL and versioning

```
https://<your-host>/v1
```

Breaking changes go in a new path prefix (`/v2`). The current major version
follows the package version's major component.

## 2. Authentication

All endpoints except `/health` and `/ready` require a bearer JWT.

```
Authorization: Bearer <token>
```

Token requirements:

| Claim | Required | Notes |
| --- | --- | --- |
| `iss` | yes | issuer; must match the deployment's expected issuer |
| `aud` | yes | audience; must match `chromatic` |
| `sub` | yes | principal identifier (any string) |
| `exp` | yes | recommended ≤ 5 min in the future |
| `iat` | yes | issued-at; rejected if more than 5 min skew |

Signature algorithm: `RS256`. The service does not accept `HS256` or `none`.

If `CHROMATIC_AUTH_MODE=trust_proxy` is set, the service trusts an upstream
gateway and reads the principal from `X-Authenticated-Principal`.

## 3. Endpoints

### 3.1 `POST /v1/detect`

Run liveness detection on a single frame. **Stateless** — each request runs a
new analyser. For session-aware detection (which is what we recommend for
real usage), use the WebSocket endpoint.

**Request**

```
POST /v1/detect HTTP/1.1
Content-Type: image/jpeg          # or image/png
Authorization: Bearer <token>

<binary frame data>
```

| Header | Required | Notes |
| --- | --- | --- |
| `Content-Type` | yes | `image/jpeg` or `image/png` |
| `Content-Length` | yes | ≤ `CHROMATIC_MAX_FRAME_BYTES` (default 16 MiB) |
| `Traceparent` | no | propagated to audit logs as `request_id` |
| `X-Session-Id` | no | groups requests for sustained-window scoring |

**Successful response (HTTP 200)**

```json
{
  "is_live": false,
  "confidence": 0.487,
  "sustained_confidence": null,
  "layer_scores": {
    "hardware": 0.78,
    "rppg":     0.0,
    "texture":  0.91,
    "geometry": 0.55,
    "motion":   0.50
  },
  "reasons": [
    "low rppg score (0.00)",
    "warming up — 28 more frames needed"
  ],
  "request_id": "e7b9c2bf-cb35-4f87-8b3d-2e8a4d8e4ce0"
}
```

A single-frame request cannot produce a positive `is_live` result — the
sustained-frames check requires multiple frames over time. For one-shot
flows, accept this and use the layer scores directly, **or** use the
WebSocket endpoint for a proper session.

### 3.2 `WebSocket /v1/ws/stream`

Stream frames over a WebSocket. The session keeps a single detector instance
warm so PRNU calibration, the rPPG sliding window, and the motion history
all accumulate across frames.

**Subprotocol**: none. The first message from the client must be a JSON
control message; subsequent messages are binary frames.

**Open**

```
GET /v1/ws/stream HTTP/1.1
Upgrade: websocket
Authorization: Bearer <token>
```

**First message — control (JSON, text frame)**

```json
{
  "type": "init",
  "format": "jpeg",
  "target_fps": 30
}
```

**Subsequent messages — frames (binary)**

Raw JPEG (or PNG) bytes per frame. The server responds with one JSON message
per processed frame:

```json
{
  "type": "diagnostic",
  "is_live": false,
  "confidence": 0.51,
  "sustained_confidence": 0.49,
  "calibration_progress": 1.0,
  "rppg_progress": 0.84,
  "layer_scores": { ... },
  "pulse": { "bpm": 72.0, "snr_db": 4.2 },
  "reasons": [...]
}
```

A `final` message is sent once the sustained-frames threshold is met or
when the session is closed:

```json
{
  "type": "final",
  "is_live": true,
  "confidence": 0.81,
  "sustained_confidence": 0.78,
  "layer_scores": { ... },
  "reasons": ["all signals consistent (confidence 0.78)"]
}
```

Either side may close at any time. The server enforces:

- Maximum frames per second per session (default 30).
- Maximum session duration (default 60 seconds).
- Frame size cap (`CHROMATIC_MAX_FRAME_BYTES`).

### 3.3 `GET /health`

Liveness probe for the orchestrator. Returns 200 with `{"status": "ok"}`
once the process is running. Does **not** require authentication.

### 3.4 `GET /ready`

Readiness probe. Returns 200 once the MediaPipe model graph is loaded and
the validator is initialised. Until then returns 503.

## 4. Errors

All error responses share the shape:

```json
{
  "error": "<error_code>",
  "request_id": "<uuid>",
  "detail": "<short, non-revealing message>"
}
```

| HTTP | `error` code | When |
| --- | --- | --- |
| 400 | `invalid_input` | Frame failed validation (size, shape, dtype, brightness). |
| 401 | `unauthorized` | Missing or invalid JWT. Never specifies which. |
| 413 | `payload_too_large` | Body exceeds `CHROMATIC_MAX_FRAME_BYTES`. |
| 415 | `unsupported_media_type` | `Content-Type` not in {`image/jpeg`, `image/png`}. |
| 422 | `decode_failed` | Frame bytes could not be decoded by OpenCV. |
| 429 | `rate_limited` | Per-principal token bucket exhausted. `Retry-After` is set. |
| 500 | `internal_error` | Unexpected failure. The `request_id` correlates to the audit log. |
| 503 | `not_ready` | Service not yet ready (model loading). |

Detail strings are deliberately generic. The full reason is in the audit log
under the same `request_id`.

## 5. Rate limits

Per-principal token-bucket: `CHROMATIC_RATE_LIMIT_PER_MIN` requests per minute
(default 60), burst of one minute's worth. `429` responses include:

```
Retry-After: 12          # seconds
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
```

WebSocket sessions are limited separately by a frame-rate cap per session,
not by the per-request bucket.

## 6. Client examples

### 6.1 Python (single-frame)

```python
import httpx

with open("face.jpg", "rb") as fh:
    response = httpx.post(
        "https://example/v1/detect",
        content=fh.read(),
        headers={
            "Content-Type": "image/jpeg",
            "Authorization": f"Bearer {token}",
        },
        timeout=5.0,
    )
response.raise_for_status()
print(response.json())
```

### 6.2 JavaScript (WebSocket stream)

```javascript
const ws = new WebSocket("wss://example/v1/ws/stream", [], {
  headers: { Authorization: `Bearer ${token}` },
});

ws.onopen = () => {
  ws.send(JSON.stringify({ type: "init", format: "jpeg", target_fps: 30 }));
  // ... pipe getUserMedia frames as JPEG blobs to ws.send(blob)
};

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "final") {
    console.log("verdict:", msg.is_live, "reasons:", msg.reasons);
    ws.close();
  }
};
```

## 7. Compatibility policy

- Adding a new field to a response is **non-breaking**.
- Removing or renaming a field is **breaking** and goes in `/v2`.
- Adding a new HTTP error code is **non-breaking** as long as existing codes
  keep their meaning.
- Tightening validation may reject inputs that previously succeeded; this is
  treated as breaking and goes in `/v2` unless it's plugging a security
  vulnerability, in which case it ships in a patch with a `security:`
  changelog entry.
