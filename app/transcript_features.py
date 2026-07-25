from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.config import TranscriptFeatureConfig


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)
SENTENCE_END = re.compile(r"[.!?…][\"'»”)]*\s*$")


def analyse_transcript(
    transcript: dict[str, Any], config: TranscriptFeatureConfig
) -> dict[str, Any]:
    """Build deterministic lightweight features for each Whisper segment."""

    language = str(transcript.get("language") or "unknown")
    raw_segments = transcript.get("segments", [])
    segments: list[dict[str, Any]] = []
    previous_end: float | None = None
    for index, raw in enumerate(raw_segments):
        start = float(raw.get("start", 0))
        end = float(raw.get("end", start))
        text = str(raw.get("text", "")).strip()
        words = _words(text)
        duration = max(0.01, end - start)
        pause_before = max(0.0, start - previous_end) if previous_end is not None else 0.0
        words_per_second = len(words) / duration
        sentence_start = _is_sentence_start(text, previous_end is None or pause_before >= 0.45)
        sentence_end = bool(SENTENCE_END.search(text))
        confidence_values = [
            float(word["probability"])
            for word in raw.get("words", [])
            if isinstance(word, dict) and word.get("probability") is not None
        ]
        segments.append({
            "id": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "word_count": len(words),
            "words_per_second": round(words_per_second, 3),
            "speech_density": round(min(1.0, words_per_second / 3.5), 3),
            "pause_before_seconds": round(pause_before, 3),
            "pause_after_seconds": 0.0,
            "sentence_start": sentence_start,
            "sentence_end": sentence_end,
            "starts_mid_sentence": not sentence_start,
            "ends_mid_sentence": not sentence_end,
            "question_count": text.count("?"),
            "exclamation_count": text.count("!"),
            "hook_phrase_score": _hook_score(text, config.hook_patterns),
            "context_dependency_score": _context_dependency(text),
            "completeness_score": _segment_completeness(text, sentence_start, sentence_end),
            "repetition_score": 0.0,
            "filler_word_ratio": _filler_ratio(words, config.filler_words),
            "language": language,
            "transcript_confidence": (
                round(sum(confidence_values) / len(confidence_values), 3)
                if confidence_values else None
            ),
        })
        previous_end = end
    for index, segment in enumerate(segments):
        next_start = segments[index + 1]["start"] if index + 1 < len(segments) else segment["end"]
        segment["pause_after_seconds"] = round(max(0.0, next_start - segment["end"]), 3)
        segment["repetition_score"] = round(_repetition_score(segment["text"], segments, index), 3)
    return {
        "language": language,
        "segment_count": len(segments),
        "segments": segments,
        "processing_duration_seconds": 0.0,
    }


def candidate_transcript_features(
    candidate_start: float,
    candidate_end: float,
    transcript_features: dict[str, Any],
) -> dict[str, Any]:
    segments = [
        segment for segment in transcript_features.get("segments", [])
        if float(segment["end"]) > candidate_start and float(segment["start"]) < candidate_end
    ]
    if not segments:
        return {"segment_ids": [], "word_count": 0, "speech_density": 0.0}
    duration = max(0.01, candidate_end - candidate_start)
    word_count = sum(int(segment["word_count"]) for segment in segments)
    def average(key: str) -> float:
        values = [float(segment[key]) for segment in segments if segment.get(key) is not None]
        return sum(values) / len(values) if values else 0.0
    return {
        "segment_ids": [int(segment["id"]) for segment in segments],
        "word_count": word_count,
        "words_per_second": round(word_count / duration, 3),
        "speech_density": round(average("speech_density"), 3),
        "pause_before_seconds": float(segments[0]["pause_before_seconds"]),
        "pause_after_seconds": float(segments[-1]["pause_after_seconds"]),
        "sentence_start": bool(segments[0]["sentence_start"]),
        "sentence_end": bool(segments[-1]["sentence_end"]),
        "starts_mid_sentence": bool(segments[0]["starts_mid_sentence"]),
        "ends_mid_sentence": bool(segments[-1]["ends_mid_sentence"]),
        "question_count": sum(int(segment["question_count"]) for segment in segments),
        "exclamation_count": sum(int(segment["exclamation_count"]) for segment in segments),
        "hook_phrase_score": round(max(float(segment["hook_phrase_score"]) for segment in segments), 3),
        "context_dependency_score": round(average("context_dependency_score"), 3),
        "completeness_score": round(average("completeness_score"), 3),
        "repetition_score": round(average("repetition_score"), 3),
        "filler_word_ratio": round(average("filler_word_ratio"), 3),
        "language": transcript_features.get("language"),
        "transcript_confidence": round(average("transcript_confidence"), 3),
    }


def _words(text: str) -> list[str]:
    return [value.lower() for value in WORD_RE.findall(text)]


def _is_sentence_start(text: str, after_pause: bool) -> bool:
    stripped = text.lstrip()
    if stripped.lower().startswith(("и ", "а ", "но ", "and ", "but ", "so ")):
        return False
    return after_pause or bool(stripped and stripped[0].isupper())


def _hook_score(text: str, patterns: list[str]) -> float:
    lowered = text.lower()
    matches = sum(pattern.lower() in lowered for pattern in patterns)
    score = 20.0 * matches
    if "?" in text:
        score += 20
    if "!" in text:
        score += 12
    if re.search(r"\b\d+[,.]?\d*%?\b", text):
        score += 12
    return min(100.0, score)


def _context_dependency(text: str) -> float:
    lowered = text.strip().lower()
    dependency = ("и ", "а ", "но ", "это ", "they ", "it ", "and ", "but ")
    score = 65.0 if lowered.startswith(dependency) else 25.0
    if re.search(r"\b(он|она|они|this|that|these|those)\b", lowered):
        score += 15
    return min(100.0, score)


def _segment_completeness(text: str, sentence_start: bool, sentence_end: bool) -> float:
    return float(35 + (30 if sentence_start else 0) + (35 if sentence_end else 0))


def _filler_ratio(words: list[str], fillers: list[str]) -> float:
    if not words:
        return 0.0
    joined = " ".join(words)
    count = sum(joined.count(pattern.lower()) for pattern in fillers)
    return min(1.0, count / len(words))


def _repetition_score(text: str, segments: list[dict[str, Any]], index: int) -> float:
    tokens = _words(text)
    if not tokens:
        return 0.0
    prior = _words(" ".join(item["text"] for item in segments[max(0, index - 2):index]))
    if not prior:
        return 0.0
    repeated = sum(count for token, count in Counter(tokens).items() if token in set(prior))
    return min(1.0, repeated / len(tokens))
