#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  setup.sh — local development bootstrap
#
#  Creates a Python venv, installs the package in editable mode with dev extras,
#  fetches the MediaPipe model, and runs the test suite as a smoke check.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "✗ ${PYTHON} not found on PATH" >&2
    exit 1
fi

py_version="$(${PYTHON} -c 'import sys; print("{0[0]}.{0[1]}".format(sys.version_info))')"
echo "→ Using Python ${py_version}"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "→ Creating venv at ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "→ Upgrading pip and installing dev dependencies"
pip install --upgrade pip wheel
pip install -e ".[api,dev]"

echo "→ Downloading MediaPipe model"
bash scripts/download_models.sh

echo "→ Running tests"
pytest -q --no-cov

cat <<'EOF'

────────────────────────────────────────────────
✓ chromatic is ready.

Common commands:
  source .venv/bin/activate

  # Single-image diagnostic (writes diagnostic.png next to the input):
  python demo/tech_demo.py --image path/to/face.jpg --output diagnostic.png

  # Multi-scenario panel (live / static / replay):
  python demo/attack_scenarios.py

  # Live webcam dashboard (requires a working camera):
  python demo/live_dashboard.py

  # Start the API:
  uvicorn chromatic.api.server:app --host 0.0.0.0 --port 8080
────────────────────────────────────────────────
EOF
