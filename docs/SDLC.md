# SDLC

> Audience: engineers, tech leads, and security partners reviewing how
> changes flow from idea to production.

## 1. Lifecycle stages

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  PLAN    │ → │  BUILD   │ → │   TEST   │ → │  DEPLOY  │ → │ OPERATE  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
  Issue +       PR with        CI gates       Release          Audit log,
  threat-       small,         (lint,         tag + signed     monitoring,
  model        focused        type,          image push,      incident
  delta if     commits        bandit,        deploy           response
  needed                      pytest,        checklist
                              SCA, SBOM)
```

Each stage has explicit security and quality gates documented below.

## 2. PLAN

Every change begins with a GitHub issue. For non-trivial changes the issue
must answer:

1. **What's the change?** A user-visible description.
2. **Why now?** Business or risk reason.
3. **Threat-model impact.** Does this add, remove, or modify any trust
   boundary, control, or assumption in [THREAT_MODEL.md](THREAT_MODEL.md)?
   If yes, the PR must update the threat model in the same commit.
4. **Roll-back plan.** How do we revert if it goes wrong?

The issue templates in `.github/ISSUE_TEMPLATE/` ask these questions
directly.

## 3. BUILD

### Branching

- Cut feature branches from `main`.
- Use short branch names: `feat/rppg-pos`, `fix/validator-edge`, etc.
- Squash-merge to `main`.

### Commits

Conventional-commit prefixes:

| Prefix | Use for |
| --- | --- |
| `feat:` | new functionality |
| `fix:` | bug fix |
| `perf:` | performance improvement |
| `refactor:` | code change without behaviour change |
| `docs:` | documentation only |
| `test:` | tests only |
| `chore:` | tooling, deps |
| `security:` | direct security improvement |

`security:` prefixes are surfaced in release notes and the changelog under
their own heading.

### Pull requests

The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) requires:

- Linked issue.
- Description of the change.
- Test plan.
- A box ticking "I updated the threat model if a control or boundary
  changed" — or marking that it does not apply.

PRs that touch detection layers must include a short note on how the change
was verified against `demo/attack_scenarios.py`.

## 4. TEST

CI runs on every push and every PR. Configuration lives under
`.github/workflows/`. The pipeline is fast (target ≤ 5 minutes) and breaks
into parallel jobs:

| Job | What it does | Tooling | Gating |
| --- | --- | --- | --- |
| `lint` | Style + simple bug patterns | `ruff check`, `ruff format --check` | required |
| `type` | Static type-check | `mypy` | required |
| `security-static` | Code-level security scan | `bandit -r src` | required |
| `secrets` | Detect committed secrets | `gitleaks` | required |
| `test` | Unit + integration + security tests, coverage ≥ 70 % | `pytest --cov` | required |
| `deps` | SCA, SBOM | `pip-audit`, `syft` | required, no HIGH/CRITICAL |
| `container-scan` | Image vulnerability scan | `trivy fs`, `trivy image` | required, no HIGH/CRITICAL |

`main` has branch protection so every gate must pass before merge.

### Test layers

- **Unit tests** (`tests/unit/`) — one file per module. Property-based tests
  via `hypothesis` for the validator and the fusion engine.
- **Integration tests** (`tests/integration/`) — end-to-end on fixed image
  fixtures. Asserts that valid inputs produce well-formed diagnostics, and
  that the LIVE scenario passes given enough warm-up frames.
- **Security tests** (`tests/security/`) — adversarial inputs (NaN/Inf,
  oversized frames, malformed dtypes, rate-limit exhaustion). These are
  the regression tests for OWASP-mapped controls.

### Manual review

Reviewers focus on:

- Does the change preserve the threat-model claims?
- Is the new code covered by a test that would catch a regression?
- Are public APIs backwards-compatible? If not, is `CHANGELOG.md` updated?
- Are dependencies pinned and minimal?

## 5. DEPLOY

Releases are cut by a maintainer. The process:

1. Bump version in `pyproject.toml` and `src/chromatic/__init__.py`.
2. Update `CHANGELOG.md` under a new `[X.Y.Z]` heading.
3. Open a release PR, get review, merge.
4. Tag the merge commit: `git tag -a v0.X.Y -m "release X.Y"` and push.
5. The `release.yml` workflow builds and signs the container image, pushes it
   to the registry by digest, and creates a GitHub Release with the
   changelog body.
6. Run through [DEPLOYMENT.md](DEPLOYMENT.md) for the deploy itself.

Container images are pushed by **digest**; downstream deployments must
reference the digest, not a tag.

## 6. OPERATE

After deploy:

- Monitor error rate and latency P50/P95 for at least 15 minutes (see
  [DEPLOYMENT.md](DEPLOYMENT.md) §4).
- Confirm audit events are reaching the SIEM.
- Confirm rate-limit metrics look normal.

Rollback triggers are documented in [DEPLOYMENT.md](DEPLOYMENT.md) §5.

### Incident response

If a deploy goes wrong:

1. Roll back per [DEPLOYMENT.md](DEPLOYMENT.md) §5.
2. Open an incident issue using the `.github/ISSUE_TEMPLATE/incident.md`
   template.
3. Within 5 business days, post a blameless postmortem describing root
   cause, contributing factors, and corrective actions.

### Patch cadence

| Severity | Response time |
| --- | --- |
| Critical (RCE, auth bypass) | patch within 24 h, deploy within 48 h |
| High (DoS, info disclosure) | patch within 7 days |
| Medium | next minor release |
| Low | next major release |

Dependency CVEs follow the same matrix — see Dependabot config in
`.github/dependabot.yml`.

## 7. Decommissioning

If a customer or environment is being decommissioned:

1. Stop accepting new traffic at the proxy.
2. Drain active sessions (wait for in-flight requests to complete).
3. Stop the container.
4. Audit logs are retained per the operator's policy — this service does
   not write to user-data storage, so there is nothing else to purge.
