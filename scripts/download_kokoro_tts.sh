#!/data/data/com.termux/files/usr/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DEST="$ROOT/models/tts/kokoro"
BASE="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main"

mkdir -p "$DEST/voices"

curl -L --fail --retry 3 -o "$DEST/config.json" "$BASE/config.json"
curl -L --fail --retry 3 -o "$DEST/kokoro-v1_0.pth" "$BASE/kokoro-v1_0.pth"

for voice in af_heart af_bella af_nicole af_sarah am_fenrir; do
  curl -L --fail --retry 3 -o "$DEST/voices/$voice.pt" "$BASE/voices/$voice.pt"
done

cat <<EOF
Downloaded Kokoro v1.0 files to:
  $DEST

Install runtime packages in the same Python environment that runs GestureLSM:
  pip install "kokoro>=0.9.4" soundfile misaki[en]

The service uses Kokoro through its Python package. The files above are pinned
local assets for offline deployment and auditability.
EOF
