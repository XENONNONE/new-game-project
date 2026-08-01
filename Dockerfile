FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies for torch CPU and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY GestureLSM/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    -r /tmp/requirements.txt \
    omegaconf \
    pyyaml \
    kittentts \
    soundfile \
    "misaki[en]" \
    && rm /tmp/requirements.txt

# Copy application code (Python model implementations in /app/models/)
COPY GestureLSM/ /app/

# Copy model weights to /models/ (separate from Python code at /app/models/)
# The code expects weights at PROJECT / "models" where PROJECT = / (parent of /app)
COPY models/ /models/

# Copy checkpoints and test WAV to paths matching the code's expectations
COPY ckpt/ /app/ckpt/
COPY test.wav /test.wav

# Create non-root user for security
RUN useradd --system --no-create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /models /test.wav
USER appuser

# Expose the avatar server port
EXPOSE 8765

# Health check: verify the server is responding and ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/ready', timeout=5)" || exit 1

# Run the server
CMD ["python", "-m", "inference_runtime.server", "--host", "0.0.0.0", "--port", "8765"]
