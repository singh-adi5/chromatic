# Threat Model

> Audience: security architects, product security reviewers, and anyone
> approving this system for deployment in front of real users.

## 1. Scope

This threat model covers the Chromatic service: the detection pipeline
itself, its REST/WebSocket front door, the audit logger, and the deployment
defaults shipped in `docker/`. It explicitly does **not** cover:

- The downstream system that consumes the verdict (your application's
  authentication or onboarding logic).
- The client-side capture environment beyond what is observable in the
  payload (browser hardening, mobile attestation, etc.).
- Network controls outside the container (WAF rules, mTLS).

We use **STRIDE** for the structured pass and an **attack tree** for the
adversarial perspective on the highest-impact threat (deepfake bypass).

## 2. System diagram and trust boundaries

```
┌──── untrusted ────┐   ┌────── trust boundary ──────┐   ┌─ trusted ─┐
│                   │   │                            │   │           │
│   end-user        │──▶│   API gateway / TLS proxy  │──▶│  detector │
│   (browser/SDK)   │   │   (rate-limit, JWT verify) │   │  (in mem) │
│                   │   │                            │   │           │
└───────────────────┘   └────────────┬───────────────┘   └─────┬─────┘
                                     │                         │
                                     ▼                         ▼
                              ┌──────────────┐         ┌─────────────────┐
                              │  audit log   │         │  no persistence │
                              │  → SIEM      │         │  (frames in mem)│
                              └──────────────┘         └─────────────────┘
```

Trust boundaries (and the controls that span each):

- **Network ↔ container:** TLS at the proxy; per-IP and per-principal rate
  limit; JWT signature verification.
- **API ↔ detector:** `FrameValidator` (allow-list shape, dtype, byte size,
  brightness bounds). No untrusted data reaches the detector unvalidated.
- **Process ↔ filesystem:** model artefacts are read from a pinned path and
  verified against `models/CHECKSUMS.txt`. No write access to user data.

## 3. Assets

| Asset | Why it matters |
| --- | --- |
| The verdict | Downstream identity decisions depend on it; manipulating it = full bypass. |
| Audit log integrity | Required for incident response and regulatory audits. |
| Configuration (decision threshold, fusion weights) | Lowering thresholds degrades security globally. |
| Model artefact (`face_landmarker.task`) | Tampering changes detection behaviour silently. |
| Application secrets (JWT signing key, etc.) | Forging tokens grants free access. |

## 4. STRIDE analysis

### 4.1 Spoofing

| ID | Threat | Likelihood | Impact | Mitigations |
| -- | --- | --- | --- | --- |
| S-01 | Replay a captured biometric session (frame stream replay). | Medium | High | Server-side challenge nonce embedded in client capture context; PRNU calibration restarts per session; sustained-frames check makes static-image replays insufficient. |
| S-02 | Forge a JWT to impersonate another principal. | Medium | High | Asymmetric signatures (RS256 / ES256), short `exp`, mandatory `aud` and `iss` claims, key rotation. |
| S-03 | Spoof source IP to bypass per-IP rate limit. | Low | Medium | Bind rate limit to the authenticated principal, not the IP; deploy behind a proxy that sets `X-Forwarded-For` honestly. |
| S-04 | Camera spoof via OBS / virtual-cam injection. | High | High | PRNU layer scores virtual cameras low; motion + rPPG layers reject pre-rendered streams. |

### 4.2 Tampering

| ID | Threat | Likelihood | Impact | Mitigations |
| -- | --- | --- | --- | --- |
| T-01 | Modify request payload in flight. | Low | Medium | TLS at the proxy; integrity check on multipart frames. |
| T-02 | Tamper with the MediaPipe model on disk. | Low | High | Models verified against SHA-256 in `models/CHECKSUMS.txt` at startup; image runs as non-root; model directory mounted read-only. |
| T-03 | Tamper with audit log records to hide fraud. | Medium | High | Audit logger writes to a dedicated handler; in production, ship to an append-only SIEM with WORM retention. |
| T-04 | Modify configuration at runtime via env var injection. | Low | High | `Settings` is frozen (`dataclass(frozen=True)`); bounds-checked at startup; container runs with read-only root filesystem. |

### 4.3 Repudiation

| ID | Threat | Likelihood | Impact | Mitigations |
| -- | --- | --- | --- | --- |
| R-01 | A user denies having attempted authentication. | Medium | Medium | Every detection produces an `AuditEvent` with `event_id`, `principal_id` (hashed), `request_id`, outcome, and confidence; ship to a tamper-evident SIEM. |
| R-02 | An operator denies a config change that lowered security. | Low | High | Configuration changes flow through Git PRs; reject runtime mutation; log `CONFIG_CHANGED` events on service boot with the active settings hash. |

### 4.4 Information disclosure

| ID | Threat | Likelihood | Impact | Mitigations |
| -- | --- | --- | --- | --- |
| I-01 | Biometric data exfiltrated from logs. | Low | High | Audit logger is allow-listed: no raw frames, no landmark coordinates, no images. PII fields are hashed by the caller before being passed in. |
| I-02 | Verbose error messages leak internal state. | Medium | Medium | Exceptions are generic at the API boundary; detailed reasons stay in audit logs only. |
| I-03 | Side-channel timing leak from validation failures. | Low | Low | Validation rejects in constant order regardless of which check fails first; size check is cheap and runs first. |
| I-04 | Memory dump exposes frames. | Low | High | Container runs without core dumps; frames are not pinned in memory beyond the active analysis window. |

### 4.5 Denial of service

| ID | Threat | Likelihood | Impact | Mitigations |
| -- | --- | --- | --- | --- |
| D-01 | Submit oversized frames to exhaust memory. | High | Medium | `max_frame_bytes` (default 16 MiB) and `max_frame_width`/`max_frame_height` enforced **before** decoding; reject early. |
| D-02 | Submit frames containing NaN/Inf to crash analyzers. | Medium | Medium | `FrameValidator` rejects non-finite numeric input. |
| D-03 | Open many concurrent sessions to exhaust threads. | Medium | Medium | Per-principal token-bucket limiter; container CPU/memory limits at the runtime. |
| D-04 | Slowloris-style WebSocket connections. | Medium | Low | Idle-connection timeout on the proxy; per-connection frame-rate limit. |
| D-05 | Adversarial input that maximises model latency. | Low | Medium | MediaPipe runs in a deterministic graph; latency bounded by frame size, already capped. |

### 4.6 Elevation of privilege

| ID | Threat | Likelihood | Impact | Mitigations |
| -- | --- | --- | --- | --- |
| E-01 | Container escape via OpenCV/MediaPipe vulnerability. | Low | High | Run as non-root user; read-only root FS; seccomp + AppArmor profiles in `docker-compose.yml`; pinned base image with security updates. |
| E-02 | Exploit dependency vulnerability (CVE in NumPy etc.). | Medium | High | Dependabot, Trivy image scan, weekly `pip-audit` in CI. |
| E-03 | Pickled-object deserialisation attack. | None | n/a | We do not deserialise pickle or untrusted YAML. JSON only at the API boundary. |

## 5. Attack tree — "Bypass liveness with a non-live presenter"

The root goal of a real attacker is to convince the system that a deepfake,
photo, or video is a live person. This tree enumerates the paths.

```
GOAL: produce LIVE verdict without a live subject

├── A. Defeat PRNU hardware-forensics layer
│   ├── A.1 Use a high-end physical camera fed deepfake on a display
│   │       └─ caught by: rPPG (no pulse), texture (moire from second screen)
│   ├── A.2 Inject frames via virtual camera, simulate sensor noise
│   │       └─ caught by: kurtosis bound + rPPG, hard to fake both
│   └── A.3 Synthesise PRNU residual that matches the expected statistics
│           └─ research-grade attack; requires per-device fingerprint to defeat
│
├── B. Defeat rPPG layer
│   ├── B.1 Inject synthetic 1.0–1.5 Hz red-channel modulation
│   │       └─ partial; CHROM is robust to luminance-only artefacts; texture
│   │          and PRNU layers still flag the synthetic source
│   ├── B.2 Use a live accomplice whose face is then deepfake-swapped
│   │       └─ caught by: geometry (pose drift inconsistent with swap mask),
│   │          texture (blending artefacts)
│   └── B.3 High-fidelity 3-D mask with thermally-modulated regions
│           └─ targeted physical attack; outside our v0.1 threat scope
│
├── C. Defeat motion layer
│   ├── C.1 Loop a short genuine video
│   │       └─ caught by: PRNU mismatch (video and live calibration differ),
│   │          rPPG SNR collapses across the loop boundary
│   └── C.2 Generate per-frame micro-motion synthetically
│           └─ partial; flow becomes too uniform → motion_variability low
│
├── D. Defeat geometry layer
│   └─ generally cooperative; geometry exists to catch off-axis paper masks,
│      not as a primary signal
│
└── E. Defeat the API boundary entirely
    ├── E.1 Forge JWT
    │       └─ mitigation: S-02 above
    ├── E.2 MITM the response and rewrite the verdict
    │       └─ mitigation: TLS at the proxy, signed responses if needed
    └── E.3 Lower the decision threshold via config injection
            └─ mitigation: T-04 above (frozen settings, read-only FS)
```

The reason the system requires **defeating multiple paths simultaneously** is
the linear-weighted-sum fusion: no single layer's score can push the verdict
across the threshold by itself when the weights are bounded as configured.

## 6. Assumptions

We rely on the following operator-side assumptions. If any is broken, the
mitigations above degrade accordingly.

1. Operators terminate TLS at the proxy and do not weaken cipher suites.
2. JWT signing keys are stored in a secret manager, not in env files.
3. The audit log destination is append-only and access-controlled.
4. Model artefacts are fetched only via `scripts/download_models.sh`, which
   enforces SHA-256 checksums.
5. The container runs as the non-root user defined in the Dockerfile.

## 7. Residual risk

After all the mitigations above, the **most likely** residual risks are:

- **High-budget targeted attacks** using bespoke 3-D masks with thermal
  modulation, or research-grade synthetic-PRNU implants. These require
  significant attacker resources and are outside the v0.1 design scope.
- **Configuration drift** where an operator lowers the decision threshold or
  rebalances fusion weights to reduce friction. We log the active settings
  hash on startup so this is detectable in audit but not preventable.
- **Adversarial inputs targeting the fusion weights**, once those weights
  become public knowledge. The sustained-frames check is our main defence;
  a formal red-team is the appropriate next mitigation.

## 8. Update cadence

This document is reviewed:

- On every major version bump.
- Whenever a new detection layer is added or removed.
- Whenever an audited customer reports a new attack class.
- At least every 6 months.
