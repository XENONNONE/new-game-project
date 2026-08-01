#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${AVATAR_PYTHON:-/root/avatar_py312/bin/python}

if [ ! -x "$PYTHON" ]; then
  PYTHON=/usr/bin/python3
fi

case "$($PYTHON -c 'import sys; print(sys.platform)')" in
  linux) ;;
  *)
    echo "Refusing non-Ubuntu Python: $PYTHON" >&2
    exit 1
    ;;
esac

cd "$ROOT/GestureLSM"
exec "$PYTHON" -m inference_runtime.server \
  --host "${AVATAR_HOST:-0.0.0.0}" \
  --port "${AVATAR_PORT:-8765}" \
  --threads "${AVATAR_THREADS:-6}"
