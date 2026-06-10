# Deployment

> Audience: SRE / platform engineers running this in production.

## 1. Prerequisites

- A container runtime: Docker 24+, Kubernetes 1.27+, ECS, or similar.
- A TLS-terminating proxy in front (Nginx, Envoy, ALB, CloudFront).
- A secret manager for the JWT public key and the audit-hash salt.
- A SIEM or log destination that supports JSON ingest.

## 2. GitHub repository hardening (one-time)

Before the first production deploy, configure the repository:

- [ ] Enable **branch protection** on `main`:
  - Require PR before merge.
  - Require status checks: `lint`, `type`, `security-static`, `secrets`,
    `test`, `deps`, `container-scan`.
  - Require linear history.
  - Restrict who can push (use `CODEOWNERS`).
- [ ] Enable **Dependabot** alerts and security updates.
- [ ] Enable **secret scanning** + **push protection**.
- [ ] Enable **code scanning** with the workflow shipped in
      `.github/workflows/security.yml`.
- [ ] Set the **GHCR** package to private (or the registry of your choice).
- [ ] Add `CODEOWNERS` reviewers for `docs/SECURITY.md`,
      `docs/THREAT_MODEL.md`, and `src/chromatic/security/`.

## 3. Build

CI publishes a container image to your configured registry (see
`.github/workflows/release.yml`). To build locally for a development
environment:

```bash
docker build -f docker/Dockerfile -t chromatic:dev .
```

The image:

- runs as user `chromatic` (UID 10001),
- has the MediaPipe model baked in (verified via SHA-256),
- exposes port 8080,
- declares a `HEALTHCHECK` against `/health`.

## 4. Deploy

### 4.1 Pre-deploy checklist

```markdown
## Deploy Checklist: chromatic vX.Y.Z
**Date:** ________  **Deployer:** ________  **Image digest:** sha256:____

### Pre-Deploy
- [ ] All CI gates green on the release tag
- [ ] Image scan reports zero HIGH/CRITICAL
- [ ] Changelog entry merged
- [ ] On-call rota notified in #ops-deploys
- [ ] Secrets in target environment present:
  - [ ] CHROMATIC_JWT_PUBLIC_KEY_PATH (or CHROMATIC_JWT_PUBLIC_KEY)
  - [ ] CHROMATIC_AUDIT_HASH_SALT
- [ ] Rollback target identified (previous digest pinned)
- [ ] No customer change-freeze in effect

### Deploy
- [ ] Deploy to staging
- [ ] Run smoke test:
       curl -fsS https://staging/.../health
       curl -fsS https://staging/.../ready
- [ ] Run synthetic detection probe (image of fixture/test_face.jpg)
- [ ] Promote to canary (5% traffic)
- [ ] Monitor for 15 minutes: error rate, P50/P95 latency, audit events
- [ ] Promote to 100%
- [ ] Monitor for 30 minutes

### Post-Deploy
- [ ] Metrics nominal vs baseline
- [ ] Tag release in monitoring tool
- [ ] Update internal status page
- [ ] Close release issue
```

### 4.2 Recommended container runtime configuration

The shipping `docker-compose.yml` reflects these. Mirror them in
Kubernetes via `securityContext` and `resources`.

| Setting | Value | Reason |
| --- | --- | --- |
| `readOnlyRootFilesystem` | true | Defeat T-04 (config tampering) |
| `runAsNonRoot` | true | Defeat E-01 (container escape) |
| `runAsUser` | 10001 | Matches Dockerfile user |
| `allowPrivilegeEscalation` | false | Defeat E-01 |
| `capabilities.drop` | `ALL` | No capability surface area |
| `tmpfs` mount | `/tmp` ≤ 64 MiB | Allow PRNU calibration scratch |
| `resources.limits.cpu` | 2 | Avoid noisy-neighbour pinning |
| `resources.limits.memory` | 512 MiB | Frame size cap × concurrency × headroom |
| `livenessProbe` | `GET /health` | Restart if the model graph dies |
| `readinessProbe` | `GET /ready` | Pull from LB during model load |

### 4.3 Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `CHROMATIC_LOG_LEVEL` | no | `INFO` | Logging verbosity |
| `CHROMATIC_FACE_MODEL_PATH` | no | `/app/models/face_landmarker.task` | Override model path |
| `CHROMATIC_MAX_FRAME_BYTES` | no | `16777216` | Reject larger frames |
| `CHROMATIC_DECISION_THRESHOLD` | no | `0.65` | Fusion threshold |
| `CHROMATIC_SUSTAINED_FRAMES` | no | `30` | Sustained-window length |
| `CHROMATIC_RATE_LIMIT_PER_MIN` | no | `60` | Token-bucket cap per principal |
| `CHROMATIC_AUDIT_LOG_PATH` | recommended | unset → stdout | Path to ship audit JSON |
| `CHROMATIC_JWT_PUBLIC_KEY_PATH` | yes (API) | — | Path to JWT verification key |
| `CHROMATIC_AUDIT_HASH_SALT` | yes | — | Per-deployment salt for `principal_id` hashing |
| `CHROMATIC_AUTH_MODE` | no | `jwt` | `jwt` \| `trust_proxy` |

Any change to a security-relevant value (threshold, rate limit, fusion
weights) is logged as a `CONFIG_CHANGED` event at boot.

## 5. Rollback

### 5.1 Rollback triggers

Trigger a rollback if **any** of the following holds for ≥ 5 consecutive
minutes after deploy:

- Error rate > 1.0 % (vs baseline 0.1 %).
- P95 detection latency > 100 ms (vs baseline ~30 ms).
- `DETECTION_FAILED` audit events > 0.5 % of traffic.
- Authentication-failure rate doubles vs the previous 24 h.

### 5.2 Rollback procedure

```bash
# Identify the previous digest from the registry
PREVIOUS=sha256:abcdef...

# In your deploy tool (kubectl/ecs/Helm), pin to the previous digest
kubectl set image deploy/chromatic chromatic=ghcr.io/owner/chromatic@$PREVIOUS

# Verify
kubectl rollout status deploy/chromatic
curl -fsS https://prod/.../health
```

Rollback target: under 5 minutes from decision to traffic stable on the
previous version.

## 6. Observability

### 6.1 What to monitor

| Metric | Source | Alert threshold |
| --- | --- | --- |
| HTTP error rate | proxy access logs | > 1 % over 5 min |
| Detection latency P95 | application metrics | > 100 ms over 5 min |
| Rate-limit hits / minute | audit log `RATE_LIMIT_HIT` count | > 10× baseline |
| Auth failures / minute | audit log `AUTH_FAILED` count | > 5× baseline |
| Model load time | startup log | > 10 s — investigate disk |

### 6.2 Where logs go

- **Operational stdout** → container log driver → log aggregator (CloudWatch,
  Stackdriver, Loki).
- **Audit JSON** → dedicated handler in `chromatic.audit` logger → SIEM
  (Splunk, Sentinel, Datadog Cloud SIEM). Configure via
  `CHROMATIC_AUDIT_LOG_PATH` and the logger's handler.
- **Trace IDs**: the API accepts `Traceparent` and propagates it through the
  audit `request_id`. Wire this into your tracer at the proxy.

## 7. Capacity

A single 2-core worker handles **~25 concurrent streams** at 30 fps before
queueing. To scale:

- Horizontal pods (HPA on CPU at 60 %) are recommended over vertical
  scaling.
- The detector is per-session and **not** thread-safe — use one process per
  worker, not threads.
- The MediaPipe graph cold-start is ~1.5 s; keep a small steady-state
  minimum replica count to absorb traffic spikes.

## 8. Disaster recovery

There is no persistent state to back up. Recovery is *redeploy the latest
known-good image*. The relevant artefacts to back up are:

- The container image registry (already redundant if using a managed
  service).
- The JWT public key and audit-hash salt (in your secret manager).
- The audit log destination (your SIEM's retention is the source of truth).

RTO: 15 minutes (image pull + container start). RPO: 0 (no user-data
storage).
