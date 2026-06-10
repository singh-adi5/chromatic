# Changelog

All notable changes to this project will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This repository is a proof of concept. Versioned releases will begin once
the items in [docs/ROADMAP.md](docs/ROADMAP.md) are complete.

### Implemented

- Multi-layer liveness detection pipeline combining:
  - rPPG biological-signal analysis
  - PRNU hardware fingerprinting
  - texture-frequency analysis
  - geometry consistency checks
  - motion/blink analysis

- Explainable weighted-fusion scoring with sustained-confidence validation windows

- FastAPI-based inspection API with structured audit logging and authenticated request handling

- Input validation pipeline with OWASP ASVS-aligned controls

- Token-bucket rate limiting with non-blocking enforcement and audit events

- Hardened container deployment:
  - non-root runtime
  - read-only filesystem
  - dropped Linux capabilities
  - healthcheck support

- CI security pipeline:
  - SAST
  - dependency CVE scanning
  - secret scanning
  - container scanning
  - typed/linted test gates

- Interactive demos:
  - single-image diagnostics
  - attack-scenario comparisons
  - live dashboard mode

### Not yet implemented
See [docs/ROADMAP.md](docs/ROADMAP.md). 
Headline gaps: trained ML classifier,
public-dataset benchmarks, real-camera validation.

- Learned anomaly-scoring layer for high-fidelity synthetic media attacks
