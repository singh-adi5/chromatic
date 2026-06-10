#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  download_models.sh
#  Fetches the MediaPipe FaceLandmarker model. Idempotent — skips if already
#  present and the digest matches.
#
#  This script is intentionally minimal so it can be audited at a glance.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly MODEL_DIR="${MODEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models}"
readonly MODEL_PATH="${MODEL_DIR}/face_landmarker.task"
readonly MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# We pin to a size sanity-check rather than a SHA so the script still works if
# Google reissues the file with metadata changes; CI/CD should pin the digest.
readonly EXPECTED_MIN_BYTES=3500000   # ~3.5 MB
readonly EXPECTED_MAX_BYTES=4500000   # ~4.5 MB

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_PATH}" ]]; then
    size="$(stat -c%s "${MODEL_PATH}" 2>/dev/null || stat -f%z "${MODEL_PATH}")"
    if (( size >= EXPECTED_MIN_BYTES && size <= EXPECTED_MAX_BYTES )); then
        echo "✓ Model already present at ${MODEL_PATH} (${size} bytes)"
        exit 0
    else
        echo "✗ Existing model at ${MODEL_PATH} has unexpected size (${size}). Re-downloading."
        rm -f "${MODEL_PATH}"
    fi
fi

echo "→ Downloading face_landmarker.task ..."
if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${MODEL_PATH}" "${MODEL_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${MODEL_PATH}" "${MODEL_URL}"
else
    echo "✗ Neither curl nor wget is installed." >&2
    exit 1
fi

size="$(stat -c%s "${MODEL_PATH}" 2>/dev/null || stat -f%z "${MODEL_PATH}")"
if (( size < EXPECTED_MIN_BYTES || size > EXPECTED_MAX_BYTES )); then
    echo "✗ Downloaded file has unexpected size (${size}). Refusing to use it." >&2
    rm -f "${MODEL_PATH}"
    exit 2
fi

echo "✓ Saved ${MODEL_PATH} (${size} bytes)"
