"""Tests for the conversation/speech-plan contract."""

import json

import pytest

from inference_runtime.conversation import (
    EMOTIONS,
    STYLES,
    AvatarSpeechPlan,
    parse_speech_plan,
    qwen_system_prompt,
)


class TestParsePlainText:
    def test_plain_text_becomes_reply(self):
        plan = parse_speech_plan("Hello there")
        assert plan.reply_text == "Hello there"
        assert plan.emotion == "neutral"

    def test_empty_string(self):
        plan = parse_speech_plan("")
        assert plan.reply_text == ""
        assert plan.emotion == "neutral"

    def test_whitespace_stripped(self):
        plan = parse_speech_plan("  hello  ")
        assert plan.reply_text == "hello"


class TestParseJson:
    def test_full_json(self):
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

    def test_unknown_emotion_defaults(self):
        plan = parse_speech_plan({"reply_text": "test", "emotion": "unknown_emotion"})
        assert plan.emotion == "neutral"

    def test_unknown_style_defaults(self):
        plan = parse_speech_plan({"reply_text": "test", "speaking_style": "unknown_style"})
        assert plan.speaking_style == "natural"

    def test_text_key_fallback(self):
        plan = parse_speech_plan({"text": "fallback text"})
        assert plan.reply_text == "fallback text"

    def test_style_key_fallback(self):
        plan = parse_speech_plan({"reply_text": "test", "style": "warm"})
        assert plan.speaking_style == "warm"

    def test_mood_fallback_to_emotion(self):
        plan = parse_speech_plan({"reply_text": "test", "emotion": "happy"})
        assert plan.mood == "happy"


class TestBoundsChecking:
    def test_speed_clamped(self):
        plan = parse_speech_plan({"reply_text": "x", "speed": 0.1})
        assert plan.speed == 0.72
        plan = parse_speech_plan({"reply_text": "x", "speed": 100})
        assert plan.speed == 1.35

    def test_gesture_intensity_clamped(self):
        plan = parse_speech_plan({"reply_text": "x", "gesture_intensity": 0.01})
        assert plan.gesture_intensity == 0.25
        plan = parse_speech_plan({"reply_text": "x", "gesture_intensity": 99})
        assert plan.gesture_intensity == 1.8

    def test_eye_contact_clamped(self):
        plan = parse_speech_plan({"reply_text": "x", "eye_contact": -5})
        assert plan.eye_contact == 0.0
        plan = parse_speech_plan({"reply_text": "x", "eye_contact": 5})
        assert plan.eye_contact == 1.0


class TestBadInput:
    def test_bad_numbers_use_defaults(self):
        plan = parse_speech_plan({"reply_text": "Okay", "speed": "fast", "eye_contact": None})
        assert plan.speed == 1.0
        assert plan.eye_contact == 0.7

    def test_non_dict_input(self):
        plan = parse_speech_plan(12345)
        assert plan.reply_text == "12345"

    def test_invalid_json_string(self):
        plan = parse_speech_plan("{not valid json")
        assert plan.reply_text == "{not valid json"

    def test_personality_truncated(self):
        long_personality = "a" * 200
        plan = parse_speech_plan({"reply_text": "x", "personality": long_personality})
        assert len(plan.personality) <= 64


class TestToDict:
    def test_to_dict_roundtrip(self):
        plan = parse_speech_plan({"reply_text": "hello", "emotion": "happy"})
        d = plan.to_dict()
        assert d["reply_text"] == "hello"
        assert d["emotion"] == "happy"
        assert "speed" in d
        assert "gesture_intensity" in d

    def test_to_dict_is_normalized(self):
        plan = parse_speech_plan({"reply_text": "x", "speed": 100})
        d = plan.to_dict()
        assert d["speed"] == 1.35


class TestSystemPrompt:
    def test_prompt_mentions_json(self):
        assert "Return only compact JSON" in qwen_system_prompt()

    def test_prompt_lists_emotions(self):
        for emotion in EMOTIONS:
            assert emotion in qwen_system_prompt()

    def test_prompt_lists_styles(self):
        for style in STYLES:
            assert style in qwen_system_prompt()

    def test_prompt_has_schema(self):
        prompt = qwen_system_prompt()
        assert "reply_text" in prompt
        assert "emotion" in prompt
        assert "speed" in prompt
        assert "gesture_intensity" in prompt
        assert "eye_contact" in prompt
        assert "mood" in prompt


class TestAvatarSpeechPlan:
    def test_frozen_dataclass(self):
        plan = AvatarSpeechPlan(reply_text="test")
        with pytest.raises(AttributeError):
            plan.reply_text = "changed"

    def test_default_values(self):
        plan = AvatarSpeechPlan(reply_text="test")
        assert plan.emotion == "neutral"
        assert plan.personality == "helpful"
        assert plan.speaking_style == "natural"
        assert plan.speed == 1.0
        assert plan.gesture_intensity == 1.0
        assert plan.eye_contact == 0.7
        assert plan.mood == "neutral"

    def test_normalized_returns_new_instance(self):
        plan = AvatarSpeechPlan(reply_text="test", emotion="HAPPY")
        normalized = plan.normalized()
        assert normalized is not plan
        assert normalized.emotion == "happy"
