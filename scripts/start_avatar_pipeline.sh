#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${AVATAR_PYTHON:-/root/avatar_py312/bin/python}
LLAMA_SERVER=${QWEN_LLAMA_SERVER:-/usr/bin/llama-server}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-"$ROOT/models/llm/Qwen3-0.6B-Q4_0.gguf"}
QWEN_HOST=${QWEN_HOST:-127.0.0.1}
QWEN_PORT=${QWEN_PORT:-8080}
QWEN_THREADS=${QWEN_THREADS:-4}
QWEN_CONTEXT=${QWEN_CONTEXT:-2048}
AVATAR_HOST=${AVATAR_HOST:-0.0.0.0}
AVATAR_PORT=${AVATAR_PORT:-8765}
AVATAR_THREADS=${AVATAR_THREADS:-6}

if [ ! -x "$PYTHON" ]; then
  PYTHON=/usr/bin/python3
fi

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "Missing llama-server at $LLAMA_SERVER" >&2
  exit 1
fi

if [ ! -f "$QWEN_MODEL_PATH" ]; then
  echo "Missing Qwen model at $QWEN_MODEL_PATH" >&2
  exit 1
fi

http_ok() {
  "$PYTHON" - "$1" <<'PY'
import sys
from urllib.request import urlopen
try:
    with urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_http() {
  url=$1
  name=$2
  tries=${3:-90}
  count=0
  while [ "$count" -lt "$tries" ]; do
    if http_ok "$url"; then
      return 0
    fi
    count=$((count + 1))
    sleep 1
  done
  echo "$name did not become ready at $url" >&2
  return 1
}

qwen_pid=
cleanup() {
  if [ -n "${qwen_pid:-}" ] && kill -0 "$qwen_pid" 2>/dev/null; then
    kill "$qwen_pid" 2>/dev/null || true
    wait "$qwen_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if http_ok "http://$QWEN_HOST:$QWEN_PORT/health"; then
  echo "Using existing Qwen server at http://$QWEN_HOST:$QWEN_PORT"
else
  echo "Starting Qwen server at http://$QWEN_HOST:$QWEN_PORT"
  "$LLAMA_SERVER" \
    -m "$QWEN_MODEL_PATH" \
    --host "$QWEN_HOST" \
    --port "$QWEN_PORT" \
    --reasoning off \
    --reasoning-budget 0 \
    -c "$QWEN_CONTEXT" \
    -t "$QWEN_THREADS" &
  qwen_pid=$!
  wait_for_http "http://$QWEN_HOST:$QWEN_PORT/health" "Qwen server" 120
fi

export QWEN_CHAT_URL="http://$QWEN_HOST:$QWEN_PORT/v1/chat/completions"
export QWEN_MODEL="${QWEN_MODEL:-local-qwen}"
export AVATAR_HOST AVATAR_PORT AVATAR_THREADS

if http_ok "http://127.0.0.1:$AVATAR_PORT/health"; then
  echo "Using existing avatar server at http://127.0.0.1:$AVATAR_PORT"
  exit 0
fi

echo "Starting avatar server at http://$AVATAR_HOST:$AVATAR_PORT"
cd "$ROOT/GestureLSM"
exec "$PYTHON" -m inference_runtime.server \
  --host "$AVATAR_HOST" \
  --port "$AVATAR_PORT" \
  --threads "$AVATAR_THREADS"
