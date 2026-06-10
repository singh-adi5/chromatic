# Security

> Audience: security auditors and operators. This is the operational
> companion to [THREAT_MODEL.md](THREAT_MODEL.md) — it maps each claim there
> to the concrete code and config that enforces it.

## 1. Secure Development Practices

We follow these principles in every change:

1. **Least privilege.** Code only handles the data it needs; the audit logger
   does not see frames, the validator does not see the verdict.
2. **Fail closed.** Validation failures, missing models, and rate-limit hits
   all result in a denial, never a fallback that lets traffic through.
3. **Defence in depth.** Security relies on multiple controls; the system
   does not assume any single one is sufficient.
4. **Reproducibility.** Pinned dependencies, deterministic scoring,
   integrity-checked model artefacts.

## 2. OWASP ASVS 4.0 — Mapping

The system targets OWASP ASVS Level 2. Items not relevant to a stateless
biometric-control service (e.g., session management for stored credentials)
are marked **N/A**.

| ASVS § | Control | Where it lives |
| --- | --- | --- |
| 1.1 | SDLC documented | [SDLC.md](SDLC.md) |
| 1.4 | Trust boundaries documented | [ARCHITECTURE.md](ARCHITECTURE.md) §4, [THREAT_MODEL.md](THREAT_MODEL.md) §2 |
| 1.6 | Cryptographic architecture documented | This file, §6 |
| 2.x | Authentication | API layer; JWT verification (`api/server.py`); see §5 |
| 3.x | Session management | Stateless; no sessions persisted |
| 4.x | Access control | Per-principal rate limit (`security/rate_limit.py`) |
| **5.1.1** | Input validated to a schema | `security/validators.py:FrameValidator.validate` — shape + dtype + channels |
| **5.1.3** | Validate range, length, format | `FrameValidator` — H/W bounds, byte cap, brightness range |
| **5.1.4** | Enforce numeric and string bounds | `_get_int` / `_get_float` in `config/settings.py` |
| **5.1.5** | Reject NaN/Inf in numeric fields | `FrameValidator.validate` rejects via `np.isfinite` |
| 5.3 | Output encoding | JSON responses through FastAPI; no HTML rendering |
| **7.1.1** | Do not log credentials or sensitive data | `security/audit.py:AuditEvent` allow-lists fields; no frame data anywhere |
| **7.2.1** | Log all authentication decisions | `AuditEventType.AUTH_SUCCEEDED` / `.AUTH_FAILED` |
| **7.2.2** | Log access control decisions | `AuditEventType.RATE_LIMIT_HIT`, `INPUT_REJECTED` |
| **7.3.1** | Log integrity events | `AuditEventType.CONFIG_CHANGED`, `CALIBRATION_RESET` |
| 8.x | Data protection | No persistence; container has read-only root FS; see §3 |
| 9.x | Communications | TLS at the proxy; we do not terminate TLS in-process |
| 10.x | Malicious code | Dependencies pinned; SCA in CI; no `pickle` deserialisation |
| **11.1.4** | Resource consumption limits | Frame byte cap; per-principal token bucket |
| **11.1.5** | Anti-automation controls | Token-bucket rate limiter |
| 14.1 | Build pipeline documented | [SDLC.md](SDLC.md) §4 |
| 14.2 | Dependency management | `requirements.txt` pinned; Dependabot enabled |
| 14.3 | Unintended security disclosure | Generic error messages at API boundary (see §4) |

The bolded items above have **direct unit-test coverage** in
`tests/security/` so the control cannot regress silently.

## 3. OWASP Machine Learning Security Top 10 — Mapping

| Risk | What it is | How we address it |
| --- | --- | --- |
| ML01 — Input manipulation | Adversarial perturbations | Multi-modal fusion; sustained-frames check; the attacker must defeat orthogonal signals. |
| ML02 — Data poisoning | Tampering with training data | Not applicable to v0.1 — we use a pretrained MediaPipe model and do not train on user data. |
| ML03 — Model inversion | Reconstructing training data from queries | Not applicable — outputs are scalar scores; no probability vector exposes class membership. |
| ML04 — Membership inference | Determining whether a sample was in training | Same as ML03; outputs are normalised scores, not probabilities over a discrete label set. |
| ML05 — Model theft | Extracting model weights via queries | Score outputs are coarse-grained and rate-limited; the pretrained MediaPipe model is already public. |
| ML06 — AI supply chain | Trojaned dependencies | Models fetched via checksum-verified script; SBOM produced in CI (Syft); SCA via `pip-audit`. |
| ML07 — Transfer learning | Leakage from upstream model | We use MediaPipe's published face-landmarker model as a geometry primitive only — no domain-specific transfer learning. |
| ML08 — Model skewing | Manipulating fairness outcomes | The decision threshold is configurable and audit-logged; fusion weights are bounded and version-controlled. |
| ML09 — Output integrity | Tampering with predictions in flight | TLS at the proxy; verdicts logged at source; downstream consumers can require signed responses. |
| ML10 — Neural net DoS | Inputs that maximise inference time | Frame size capped before inference; MediaPipe latency bounded by input dimensions. |

## 4. Error-handling policy

The API boundary follows the **generic-message + correlation-ID** pattern.
Clients see a stable, non-revealing error code; full detail goes to the
audit log under the same `request_id`.

| Internal exception | HTTP response | Audit event |
| --- | --- | --- |
| `InvalidFrameError`, `FrameTooLargeError` | 400 `{"error": "invalid_input", "request_id": "..."}` | `INPUT_REJECTED` with `reason` field |
| `FaceNotFoundError` | 200 with `is_live=false`, `reasons=["no face detected"]` | `DETECTION_COMPLETED` outcome=failure |
| `RateLimitExceededError` | 429 `{"error": "rate_limited"}` + `Retry-After` | `RATE_LIMIT_HIT` |
| `AuthenticationError` | 401 `{"error": "unauthorized"}` | `AUTH_FAILED` |
| Any other exception | 500 `{"error": "internal_error", "request_id": "..."}` | `DETECTION_FAILED` with internal trace |

`AuthenticationError` is deliberately generic in its public form — it does
not reveal whether the principal does not exist, the signature is invalid,
or the token has expired.

## 5. Authentication and authorisation

The repo ships a reference FastAPI server in `src/chromatic/api/`. It
expects callers to present a **bearer JWT**. The defaults:

- Signature algorithm: `RS256` (asymmetric). The public key path comes from
  `CHROMATIC_JWT_PUBLIC_KEY_PATH`; the private key is **never** read by this
  service.
- Required claims: `iss`, `aud`, `exp`, `iat`, `sub`.
- `exp` window: ≤ 5 minutes recommended.
- The `sub` claim is hashed before use in audit logs.

If you front this service with an API gateway that already terminates
authentication (e.g., AWS API Gateway with a Cognito authoriser), set
`CHROMATIC_AUTH_MODE=trust_proxy` and pass the verified principal in
`X-Authenticated-Principal`. Do this only with a private network between the
gateway and the service.

## 6. Cryptographic architecture

Chromatic performs **no key generation, key storage, or encryption of
user data** itself. Its only cryptographic operation is JWT signature
verification (handled by `pyjwt`).

- Hashing for audit IDs uses **BLAKE2b-160** — fast, salted with a per-deployment
  application secret to prevent cross-deployment linkage.
- Model integrity: **SHA-256** verified by `scripts/download_models.sh`.

We rely on the operator's reverse proxy for TLS. Recommended minimum:
TLS 1.2+, ECDHE, AES-GCM or ChaCha20-Poly1305, HSTS with `max-age` ≥
31536000.

## 7. Logging and audit

Two loggers, two destinations:

- **Operational logs** — logger name `chromatic.*` (everything except
  `.audit`). Free-form, includes warnings and timings. Goes to stdout in
  the container. Acceptable retention: ≤ 30 days.
- **Audit logs** — logger name `chromatic.audit`. Strictly structured
  JSON (`AuditEvent`). Goes to its own handler. Acceptable retention:
  ≥ 1 year for regulated deployments, with WORM semantics.

Audit-event fields are an allow-list — adding new fields requires a code
change and review:

```
event_id        UUIDv4 — globally unique
event_type      AuditEventType enum value
timestamp       seconds since epoch (float)
principal_id    hashed identifier (str | null)
request_id      correlation ID with the API request (str | null)
outcome         "success" | "failure"
reason          generic string; must not leak internal state
metadata        small dict of operational counters
```

Specifically forbidden: raw frames, landmark coordinates, base64 blobs,
unhashed identifiers.

## 8. Secret management

- The application accepts secrets **only via environment variables** —
  there is no support for reading secrets from files committed to the repo.
- The Dockerfile does not bake any secrets.
- `.env` files are gitignored.
- In production, mount secrets from a secret manager (AWS Secrets Manager,
  GCP Secret Manager, Vault) into the container.

Pre-commit runs `gitleaks` on every commit; CI runs it on every PR.

## 9. Hardened container defaults

The container image is built from `python:3.12-slim` and:

- Runs as a non-root user (`chromatic`, UID 10001).
- Has `HEALTHCHECK` configured against `/health`.
- Sets `PYTHONUNBUFFERED=1` and `PYTHONDONTWRITEBYTECODE=1`.
- Excludes test files, `.git/`, and demo PNGs via `.dockerignore`.
- Pins the Python version, the base image digest, and every dependency.

The `docker-compose.yml` ships with recommended runtime hardening:

- `read_only: true`
- `tmpfs: ["/tmp:size=64m"]`
- `cap_drop: [ALL]`
- `security_opt: ["no-new-privileges:true"]`
- CPU and memory limits

## 10. CI/CD security gates

A change cannot reach `main` without:

- `ruff check` clean (no warnings).
- `mypy` clean.
- `bandit -r src` clean.
- `pytest` green, including all `tests/security/*` tests, with coverage
  ≥ 70 %.
- `trivy fs` clean for HIGH/CRITICAL findings.
- `gitleaks` clean.

Workflow definitions live under `.github/workflows/`. Branch protection
must be enabled on `main` so these gates are enforced — see
[DEPLOYMENT.md](DEPLOYMENT.md) §2 for the GitHub configuration checklist.

## 11. Privacy

This service processes biometric data (facial video). In most jurisdictions
this is a special category of personal data with stricter rules. The defaults
are designed to minimise that exposure:

- **No persistence.** Frames are processed in memory and discarded.
- **No identifiers in detection logs.** Only hashed principals appear in
  audit events, and only if the caller chose to pass one.
- **No outbound network calls** from the detection path. Everything happens
  locally inside the container.

This does not make the system automatically compliant with GDPR, HIPAA, or
similar — your downstream system still needs a lawful basis to process the
biometric data, retention policies, and data-subject access procedures. But
it ensures that *this component* contributes the minimum possible to your
overall compliance burden.
