"""Small no-model tests for the avatar speech contract."""

from __future__ import annotations

import json

from inference_runtime.conversation import parse_speech_plan, qwen_system_prompt


def test_parse_plain_text() -> None:
    plan = parse_speech_plan("Hello there")
    assert plan.reply_text == "Hello there"
    assert plan.emotion == "neutral"


def test_parse_json_bounds() -> None:
    plan = parse_speech_plan(
        json.dumps(
            {
                "reply_text": "Nice.",
                "emotion": "HAPPY",
                "speaking_style": "energetic",
                "speed": 9,
                "gesture_intensity": -1,
                "eye_contact": 5,
            }
        )
    )
    assert plan.emotion == "happy"
    assert plan.speed == 1.35
    assert plan.gesture_intensity == 0.25
    assert plan.eye_contact == 1.0


def test_prompt_mentions_json() -> None:
    assert "Return only compact JSON" in qwen_system_prompt()


def test_bad_numbers_use_defaults() -> None:
    plan = parse_speech_plan({"reply_text": "Okay", "speed": "fast", "eye_contact": None})
    assert plan.speed == 1.0
    assert plan.eye_contact == 0.7
