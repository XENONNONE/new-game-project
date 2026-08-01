# Offline Android runtime

This directory is the inference-only bridge between GestureLSM and Godot. It does
not import trainers, datasets, SMPL-X mesh generation, Gradio, WandB, Whisper, or
Montreal Forced Aligner.

## Components

- `inference_runtime.pipeline`: persistent MeanFlow and three RVQ-VAE decoders.
- `inference_runtime.server`: localhost WAV-to-humanoid HTTP service.
- `inference_runtime.conversation`: strict Qwen JSON contract for reply text,
  emotion, personality, speaking style, speed, mood, and gesture intensity.
- `inference_runtime.tts`: local TTS adapter that emits mono 16-bit 16 kHz WAV.
- `../godot/main.gd`: audio-synchronized Skeleton3D player.
- `../avatar.glb`: the current UniVRM 0.x avatar (despite the requested VRM 1.0).
- `../models/llm/Qwen3-0.6B-Q4_0.gguf`: offline Qwen chat model.

## Start Runtime

From the project directory:

```sh
sh scripts/start_avatar_server.sh
```

The launcher uses `/root/avatar_py312/bin/python` by default and falls back to
`/usr/bin/python3`. Do not start this service with bare `python` in the current
workspace: it resolves to the Termux Python, which is missing the GestureLSM and
Kokoro runtime packages.

The process must print `GestureLSM ready`. In another shell:

```sh
curl http://127.0.0.1:8765/health
curl -X POST --data-binary @../test.wav -H 'Content-Type: audio/wav' \
  http://127.0.0.1:8765/infer -o /tmp/motion.json
```

Put a 16-bit PCM WAV at `res://test.wav`, open the Godot project, and press the
test button. The Godot Android export must enable the INTERNET permission; this
is required even for localhost.

## Local LLM

Install llama.cpp as the normal Termux user (not root), then run:

```sh
pkg install llama-cpp
llama-server -m models/llm/Qwen3-0.6B-Q4_0.gguf \
  --host 127.0.0.1 --port 8080 -c 2048 -t 4
```

The 409 MiB Q4 model is intentionally small so it can coexist with PyTorch and
Godot. Disable Qwen thinking mode for short, low-latency spoken answers.

Use this system prompt for avatar turns:

```text
Return only compact JSON for an avatar speech turn. Schema: {"reply_text":string,"emotion":"neutral|happy|excited|curious|thinking|confused|sad|angry|surprised|listening|calm","personality":string,"speaking_style":"natural|warm|bright|serious|soft|energetic|whisper","speed":number,"gesture_intensity":number,"eye_contact":number,"mood":string}. Keep reply_text short for low latency. Use emotion and style to guide TTS and gestures.
```

Then send Qwen's JSON directly to:

```sh
curl -X POST -H 'Content-Type: application/json' \
  --data '{"reply_text":"Hey, I can explain that.","emotion":"happy","speaking_style":"warm","speed":1.03,"gesture_intensity":1.2,"eye_contact":0.85}' \
  http://127.0.0.1:8765/speak -o /tmp/speech_motion.json
```

Send plain user text through the complete Qwen, TTS, and gesture chain:

```sh
curl -X POST -H 'Content-Type: application/json' \
  --data '{"message":"Why is the sky blue?"}' \
  http://127.0.0.1:8765/chat -o /tmp/avatar_turn.json
```

The response includes:

- `audio_wav_base64`: generated speech as normalized 16 kHz PCM WAV.
- `audio_pcm16_base64`: header-free mono PCM for Godot `AudioStreamWAV`.
- `wav`: channel count, source rate, resample flag, duration, and expected
  motion frame count.
- `windows`: exact conditioning window timestamps in samples, seconds, and
  motion-frame coordinates.
- `speech_plan`: normalized emotion/style values used by TTS and Godot.

## TTS choice

For this project, the practical small local TTS target is Kokoro-82M. Its model
card says it is an Apache-2.0 open-weight 82M-parameter model, with v1.0
published on 2025-01-27, 8 language groups, 54 voices, and a 327 MB checkpoint.
Its own voice table also warns that voice quality varies and that very short
utterances under roughly 10-20 tokens can be weaker. This matters for avatar
sync: do not over-trust one-word replies.

Hard truth: Kokoro is expressive through voice choice and style embeddings; it
is not a native text emotion recognizer. In this runtime, Qwen predicts emotion
from text/context, and Kokoro maps that emotion to voice, speed, and style. The
same emotion also drives Godot gesture intensity, head motion, eye contact,
shoulder posture, and mouth energy. Qwen3-TTS is a better research direction for
instruction-level emotional speech, but its 2026 technical report describes a
larger streaming family, not a tiny drop-in Android companion for the current
Qwen 0.6B + PyTorch GestureLSM stack.

Download Kokoro assets for offline audit/deployment:

```sh
sh scripts/download_kokoro_tts.sh
```

Install the runtime package in the same Python environment that runs the server.
The current known-good runtime is `/root/avatar_py312`:

```sh
/root/avatar_py312/bin/python -m pip install "kokoro==0.9.4" soundfile "misaki[en]"
```

Kokoro 0.9.4 supports Python 3.10 through 3.12, so it cannot run in the
workspace's `/usr/bin/python3.14`. If Kokoro cannot run in your Python, set
`AVATAR_TTS_BACKEND=command` and
`AVATAR_TTS_COMMAND` to any local TTS command that writes `{wav}` from `{text}`.
The server still normalizes that WAV to the 16 kHz PCM format GestureLSM expects.

## Important runtime facts

The model consumes 68,224 sample conditioning windows (about 4.264 seconds) and
produces 128 frames at 30 FPS. Later windows overlap by 16 frames; the last four
latent frames seed the next window. This preserves the published tensor shapes.

Adjacent windows use a 16-frame shortest-path quaternion crossfade. Godot uses
the audio playback clock as the master clock and applies GestureLSM first, then
additive idle, head, gaze, finger, face, and lip layers.

The original audio loader accidentally stores librosa frame indices directly in
a sample-rate onset array. This runtime fixes their positions while preserving
the amplitude channel. Text tokens currently default to unknown/padding, which
matches the proven audio-only smoke-test path. TTS text can later be aligned
directly without ASR.

The service response includes `load_s`, `wav_decode_s`, `feature_s`,
`meanflow_s`, `rvq_s`, `retarget_s`, and `total_s`. Newer responses also include
WAV diagnostics and per-window timestamps, which are the first place to look if
gesture timing drifts from speech.

## Version and cache notes

The active Termux Python reports neither `numpy` nor `torch`. The system
`/usr/bin/python3.14` can run GestureLSM, but Kokoro 0.9.4 does not support
Python 3.14. The restored runtime is `/root/avatar_py312/bin/python` with Python
3.12, Torch 2.6 CPU, NumPy 1.26, Kokoro 0.9.4, Misaki 0.9.4, spaCy 3.8, and the
`en_core_web_sm` model.

Use `UV_LINK_MODE=copy` when rebuilding that environment. A previous uv install
left incomplete hardlinked packages in the cache, so the cache was cleaned and
the venv was rebuilt with copied wheels.

The installed ffmpeg still resolves to the Termux binary and has had a
shared-library mismatch. PCM conversion therefore uses the Python standard
library.

This first test endpoint is batch-oriented: it generates a window before audio
playback begins. True microphone latency below the 4.264-second model window
requires a causal approximation or speculative partial-window updates; the
checkpoint was trained for fixed 128-frame windows, so claiming 20-30 FPS
generation without measuring that approximation would be incorrect.

## Primary sources

- Kokoro-82M: <https://huggingface.co/hexgrad/Kokoro-82M>
- Kokoro implementation: <https://github.com/hexgrad/kokoro>
- Qwen3-TTS official releases: <https://github.com/QwenLM/Qwen3-TTS>
- GestureLSM ICCV 2025 paper: <https://openaccess.thecvf.com/content/ICCV2025/html/Liu_GestureLSM_Latent_Shortcut_based_Co-Speech_Gesture_Generation_with_Spatial-Temporal_Modeling_ICCV_2025_paper.html>
