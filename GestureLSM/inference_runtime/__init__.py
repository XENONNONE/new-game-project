"""Small, training-free runtime for GestureLSM on Android/Termux."""

from __future__ import annotations

from typing import Any

__all__ = ["GesturePipeline"]


def __getattr__(name: str) -> Any:
    if name == "GesturePipeline":
        from .pipeline import GesturePipeline

        return GesturePipeline
    raise AttributeError(name)
