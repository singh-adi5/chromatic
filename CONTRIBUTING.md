# Contributing

Thanks for considering a contribution. This document covers the practical
side; for the *why* behind the architecture, read
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick links

- Bug? File a GitHub issue with reproduction steps.
- Security issue? **Do not** open a public issue — see [SECURITY.md](SECURITY.md).
- Question? Use GitHub Discussions.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
pre-commit install
bash scripts/download_models.sh
```

Confirm everything is wired up:

```bash
pytest                              # full suite
ruff check src tests demo           # lint
ruff format --check src tests demo  # formatting
mypy                                # type check
bandit -r src                       # security lint
```

## Branching and commits

- Cut feature branches from `main`.
- Use **conventional-commit-style** prefixes: `feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, `chore:`, `perf:`, `security:`.
- Keep commits small and focused. A clean history helps reviewers.

Example:

```
feat(rppg): switch CHROM to POS algorithm for low-light robustness
```

## Pull requests

A good PR:

1. Has tests for new behaviour and for regressions you are fixing.
2. Updates documentation if it changes a contract or threat-model claim.
3. Keeps the diff focused — no unrelated formatting churn.
4. Passes CI (lint, type, security, tests, coverage ≥ 70 %).
5. Has a description that explains *why*, not just *what*.

If the change touches a detection layer or the fusion logic, include a brief
note on how you verified the change does not regress the attack-scenarios
output. Re-running `python demo/attack_scenarios.py --image fixtures/face.jpg`
and attaching the resulting PNG is usually enough.

## Code style

- Type hints on public APIs.
- Docstrings on public functions and classes.
- No `print()` outside `demo/`; use the `logging` module.
- No global mutable state.
- Use `pathlib.Path` rather than `os.path` for new code.

## Adding a new detection layer

The architecture is built to make this easy. To add a layer:

1. Create `src/chromatic/core/<layer>.py` with a `<Layer>Metrics`
   dataclass and a `<Layer>Analyzer` class.
2. Add it to `LivenessDetector.__init__` and call it from `process_frame`.
3. Add a `<layer>` key to `fusion_weights` (rebalance the existing weights
   so they sum to 1.0).
4. Add a scoring branch in `_compute_scores` that maps your raw metric to
   `[0, 1]` via `_saturating_score` or a custom transform.
5. Add a sub-panel to `demo/tech_demo.py` so the layer's output is visible.
6. Add unit tests in `tests/unit/test_<layer>.py`.
7. Update [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — particularly the
   layer table — and [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) if the
   new layer changes the attack surface.

## Reviewing detection-related PRs

Reviewers should specifically check:

- Does the new score saturate correctly at the boundaries?
- Does it degrade gracefully (return 0.5 or similar) before the layer has
  enough data?
- Is the metric calculation **deterministic** given the same input?
- Are any new external models pinned by checksum?
- Does the THREAT_MODEL claim about which attack class it catches survive
  contact with `attack_scenarios.py`?

## Releasing

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Releases are cut by a
maintainer; contributors should not bump versions in their PRs.
