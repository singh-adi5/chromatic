# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
#  Chromatic — production container
#
#  Design choices:
#    • Multi-stage build keeps the final image small and free of build tools.
#    • Pinned base image (digest in CI) — no floating `:latest` tags.
#    • Non-root runtime user (UID 10001) — defence-in-depth against escape.
#    • Read-only filesystem-friendly (writable paths via tmpfs at runtime).
#    • Healthcheck hits the /healthz endpoint exposed by the API server.
# ─────────────────────────────────────────────────────────────────────────────

# ───── Stage 1: builder ─────
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS-level deps required to build OpenCV/mediapipe wheels and to fetch the model.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps first (better layer caching).
COPY pyproject.toml README.md LICENSE ./
COPY requirements.txt requirements-dev.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt fastapi "uvicorn[standard]" pyjwt

# Copy source and install the package itself.
COPY src/ ./src/
RUN /opt/venv/bin/pip install --no-deps .

# Download the MediaPipe face-landmark model — pinned to a known good release.
RUN mkdir -p /opt/chromatic/models \
 && curl -fsSL -o /opt/chromatic/models/face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task


# ───── Stage 2: runtime ─────
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHROMATIC_FACE_MODEL_PATH="/opt/chromatic/models/face_landmarker.task" \
    CHROMATIC_LOG_LEVEL="INFO"

# Minimal runtime libraries only — no compilers, no curl, no apt cache.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 chromatic \
 && useradd  --system --uid 10001 --gid chromatic --home /home/chromatic --create-home chromatic

COPY --from=builder /opt/venv      /opt/venv
COPY --from=builder /opt/chromatic     /opt/chromatic

USER chromatic
WORKDIR /home/chromatic

# 8080 is the convention for unprivileged HTTP services.
EXPOSE 8080

# Tini is PID 1 — clean signal forwarding and zombie reaping.
ENTRYPOINT ["/usr/bin/tini", "--"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "chromatic.api.server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
