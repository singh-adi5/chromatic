# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report them privately by emailing
**security@REPLACE_DOMAIN.example** with:

1. A description of the vulnerability and its potential impact.
2. Steps to reproduce (ideally with a minimal proof-of-concept).
3. The version (commit SHA) you tested against.
4. Your name and any disclosure preferences (we offer optional credit).

You should receive an acknowledgement within **2 business days**. We aim to
provide an initial assessment within **5 business days** and a fix or
mitigation within **30 days** for high-severity issues.

We follow **coordinated disclosure**: please do not publicly disclose the
vulnerability until a fix has been released, or until 90 days have elapsed
(whichever comes first).

## Scope

In scope:

- Source code under `src/chromatic/`.
- Configuration defaults shipped in `src/chromatic/config/settings.py`.
- Container images published from this repository.
- Default workflows in `.github/workflows/`.

Out of scope:

- Vulnerabilities in third-party dependencies — please report those upstream.
- Misconfigurations introduced by downstream operators.
- Denial-of-service that requires sustained access already at rate-limit caps.
- Issues in `demo/` requiring local code execution.

## Security Properties (Threat Model Summary)

Read [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the full STRIDE analysis.

We make the following claims by default:

| Claim | How it is enforced |
| --- | --- |
| Input is validated against an allow-list. | `chromatic.security.validators.FrameValidator`, applied before any detection logic runs. |
| No biometric data is persisted. | The pipeline operates in-memory; frames are discarded after fusion. |
| No PII is logged. | The `chromatic.audit` logger emits structured JSON with hashed principal IDs only. |
| Resource consumption is bounded. | Frame byte-size limit (default 16 MiB), token-bucket rate limiter per principal. |
| Model artefacts are integrity-checked. | `scripts/download_models.sh` verifies SHA-256 against `models/CHECKSUMS.txt`. |

If you find a deviation from these claims, that is a vulnerability worth
reporting.

## Hardening Checklist for Operators

If you deploy this in front of real traffic, see
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — but at minimum:

- Run behind a TLS-terminating reverse proxy.
- Set `CHROMATIC_AUDIT_LOG_PATH` to ship audit events to your SIEM.
- Tighten `CHROMATIC_RATE_LIMIT_PER_MIN` for your traffic profile.
- Pin the container image to a digest, not a tag.
- Run as the non-root user provided in the image.
- Set CPU/memory limits at the container runtime.
