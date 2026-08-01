"""Client for a local llama.cpp/Qwen OpenAI-compatible chat server."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .config import CONFIG
from .conversation import AvatarSpeechPlan, parse_speech_plan, qwen_system_prompt
from .logging_config import get_logger

logger = get_logger("llm")


def _compact_json(content: str) -> str:
    content = content.replace("```json", "").replace("```", "").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return content[start : end + 1]
    return content


def qwen_chat(message: str) -> AvatarSpeechPlan:
    """Send *message* to the local Qwen server and return a speech plan.

    Raises ``RuntimeError`` if the server returns an error or the response
    cannot be parsed.
    """
    message = message.strip()
    if not message:
        raise ValueError("message cannot be empty")
    endpoint = CONFIG.llm.chat_url
    payload = {
        "model": CONFIG.llm.model,
        "messages": [
            {"role": "system", "content": qwen_system_prompt()},
            {"role": "user", "content": message + "\n/no_think"},
        ],
        "temperature": 0.35,
        "max_tokens": 128,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    logger.debug("POST %s (model=%s, msg_len=%d)", endpoint, CONFIG.llm.model, len(message))
    request = Request(endpoint, body, {"Content-Type": "application/json"})
    with urlopen(request, timeout=CONFIG.llm.timeout) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    logger.debug("Qwen response: %s", content[:120])
    return parse_speech_plan(_compact_json(content))
