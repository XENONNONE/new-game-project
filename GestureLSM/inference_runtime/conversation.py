"""Structured conversation-to-speech contract for local Qwen output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

EMOTIONS = (
    "neutral",
    "happy",
    "excited",
    "curious",
    "thinking",
    "confused",
    "sad",
    "angry",
    "surprised",
    "listening",
    "calm",
)
STYLES = ("natural", "warm", "bright", "serious", "soft", "energetic", "whisper")


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AvatarSpeechPlan:
    reply_text: str
    emotion: str = "neutral"
    personality: str = "helpful"
    speaking_style: str = "natural"
    speed: float = 1.0
    gesture_intensity: float = 1.0
    eye_contact: float = 0.7
    mood: str = "neutral"

    def normalized(self) -> AvatarSpeechPlan:
        emotion = self.emotion.lower().strip()
        style = self.speaking_style.lower().strip()
        return AvatarSpeechPlan(
            reply_text=self.reply_text.strip(),
            emotion=emotion if emotion in EMOTIONS else "neutral",
            personality=(self.personality or "helpful").strip()[:64],
            speaking_style=style if style in STYLES else "natural",
            speed=min(1.35, max(0.72, float(self.speed or 1.0))),
            gesture_intensity=min(1.8, max(0.25, float(self.gesture_intensity or 1.0))),
            eye_contact=min(1.0, max(0.0, float(self.eye_contact or 0.7))),
            mood=(self.mood or emotion or "neutral").strip()[:48],
        )

    def to_dict(self) -> dict:
        return asdict(self.normalized())


def qwen_system_prompt() -> str:
    return (
        "Return only compact JSON for an avatar speech turn. Schema: "
        '{"reply_text":string,"emotion":"neutral|happy|excited|curious|thinking|confused|sad|angry|surprised|listening|calm",'
        '"personality":string,"speaking_style":"natural|warm|bright|serious|soft|energetic|whisper",'
        '"speed":number,"gesture_intensity":number,"eye_contact":number,"mood":string}. '
        "Keep reply_text short for low latency. Use emotion and style to guide TTS and gestures."
    )


def parse_speech_plan(raw: str | dict) -> AvatarSpeechPlan:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {"reply_text": raw}
    if not isinstance(raw, dict):
        raw = {"reply_text": str(raw)}
    return AvatarSpeechPlan(
        reply_text=str(raw.get("reply_text") or raw.get("text") or ""),
        emotion=str(raw.get("emotion") or "neutral"),
        personality=str(raw.get("personality") or "helpful"),
        speaking_style=str(raw.get("speaking_style") or raw.get("style") or "natural"),
        speed=_number(raw.get("speed"), 1.0),
        gesture_intensity=_number(raw.get("gesture_intensity"), 1.0),
        eye_contact=_number(raw.get("eye_contact"), 0.7),
        mood=str(raw.get("mood") or raw.get("emotion") or "neutral"),
    ).normalized()
