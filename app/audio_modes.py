"""Explicit user audio-mode contract shared by planning, TTS, and rendering."""

from __future__ import annotations

from typing import Any


AUDIO_MODES = frozenset({"original", "original_enhanced", "voiceover", "replace_voice", "mixed"})
TTS_AUDIO_MODES = frozenset({"voiceover", "replace_voice", "mixed"})


def allows_tts(audio_mode: str) -> bool:
    return audio_mode in TTS_AUDIO_MODES


def tts_eligibility(plan: Any) -> tuple[bool, str]:
    """Return a deterministic provider-call decision for a ProductionPlan."""

    mode = str(getattr(plan, "audio_mode", "original"))
    if not allows_tts(mode):
        return False, "source_audio_mode"
    if not bool(getattr(plan, "tts_eligible", False)):
        return False, "plan_has_no_eligible_narration"
    narration = [item for item in getattr(plan, "segments", []) if getattr(item, "segment_type", "") == "narration"]
    if not narration:
        return False, "plan_has_no_narration"
    if not any(str(getattr(item, "text", "")).strip() for item in narration):
        return False, "empty_narration"
    return True, "explicit_voiceover_intent"
