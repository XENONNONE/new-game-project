#!/usr/bin/env python3
"""Smoke-test the local avatar HTTP server without dumping full motion JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    url: str,
    payload: dict | None = None,
    timeout: float = 300.0,
    method: str | None = None,
) -> tuple[dict, float]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    started = time.perf_counter()
    request_method = method or ("POST" if payload is not None else "GET")
    request = Request(url, data=body, headers=headers, method=request_method)
    with urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return data, time.perf_counter() - started


def request_sse(
    url: str,
    payload: dict | None = None,
    raw_data: bytes | None = None,
    timeout: float = 300.0,
) -> tuple[int, float, float]:
    body = None
    headers: dict[str, str] = {}
    if raw_data is not None:
        body = raw_data
        headers["Content-Type"] = "application/octet-stream"
    elif payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    event_count = 0
    first_event_time = None
    with urlopen(request, timeout=timeout) as response:
        if response.headers.get("Content-Type") != "text/event-stream":
            ct = response.headers.get("Content-Type")
            raise RuntimeError(f"streaming endpoint did not return text/event-stream (got {ct})")
        for line in response:
            line = line.decode("utf-8").strip()
            if line.startswith("data:") and line != "data:":
                event_count += 1
                if first_event_time is None:
                    first_event_time = time.perf_counter()
    elapsed = time.perf_counter() - started
    time_to_first = (first_event_time - started) if first_event_time else elapsed
    return event_count, time_to_first, elapsed


def summarize_motion(name: str, payload: dict, elapsed: float) -> None:
    if "error" in payload:
        raise RuntimeError(f"{name} failed: {payload['error']}")
    frames = payload.get("frames") or []
    wav = payload.get("wav") or {}
    timings = payload.get("timings") or {}
    print(
        f"{name}: ok, {len(frames)} frames, "
        f"{wav.get('duration_s', '?')}s audio, "
        f"{elapsed:.2f}s HTTP, {timings.get('total_s', '?')}s model"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--speak", action="store_true", help="also test Kokoro/TTS-backed /speak")
    parser.add_argument(
        "--chat", action="store_true", help="also test Qwen -> TTS -> gesture /chat"
    )
    parser.add_argument("--message", default="Say hello in one short sentence.")
    args = parser.parse_args()

    try:
        test_wav = Path(__file__).resolve().parent.parent / "test.wav"
        if not test_wav.is_file():
            test_wav = Path("test.wav")
        test_wav_bytes = test_wav.read_bytes()

        health, elapsed = request_json(f"{args.base_url}/health", timeout=10.0)
        print(f"health: ok={health.get('ok')} fps={health.get('fps')} ({elapsed:.2f}s)")

        infer, elapsed = request_json(f"{args.base_url}/infer_test", method="POST")
        summarize_motion("infer_test", infer, elapsed)

        stream_url = f"{args.base_url}/infer_stream"
        events, ttfb, elapsed = request_sse(stream_url, raw_data=test_wav_bytes)
        print(f"infer_stream: {events} events, TTFB={ttfb:.2f}s ({elapsed:.2f}s total)")

        if args.speak:
            speak_payload = {
                "reply_text": "Hey, I can explain that.",
                "emotion": "happy",
                "speaking_style": "warm",
                "speed": 1.03,
                "gesture_intensity": 1.2,
                "eye_contact": 0.85,
            }
            speak, elapsed = request_json(f"{args.base_url}/speak", speak_payload)
            summarize_motion("speak", speak, elapsed)
            tts = speak.get("tts") or {}
            print(f"speak: tts={tts.get('backend')} {tts.get('tts_s', '?')}s")
        if args.chat:
            chat, elapsed = request_json(f"{args.base_url}/chat", {"message": args.message})
            summarize_motion("chat", chat, elapsed)
            plan = chat.get("speech_plan") or {}
            tts = chat.get("tts") or {}
            print(f"chat: reply={plan.get('reply_text')!r}")
            emotion = plan.get("emotion")
            backend = tts.get("backend")
            tts_s = tts.get("tts_s", "?")
            print(f"chat: emotion={emotion} tts={backend} {tts_s}s")
    except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
        raise SystemExit(f"smoke test failed: {error}") from error


if __name__ == "__main__":
    main()
