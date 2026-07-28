from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(slots=True)
class Candidate:
    id: str
    start: float
    end: float
    text: str
    reason: str = ""
    transcript_segment_ids: list[int] = field(default_factory=list)
    start_boundary_reason: str = ""
    end_boundary_reason: str = ""
    feature_vector: dict[str, Any] = field(default_factory=dict)
    local_scores: dict[str, Any] = field(default_factory=dict)
    local_quality_score: float = 0.0
    ai_score: float | None = None
    explanations: list[str] = field(default_factory=list)
    chapter_id: str | None = None
    story_unit_id: str | None = None
    core_idea: str = ""
    content_signature: dict[str, Any] = field(default_factory=dict)
    boundary_diagnostics: dict[str, Any] = field(default_factory=dict)
    incremental_coverage_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "text": self.text,
            "reason": self.reason,
            "transcript_segment_ids": self.transcript_segment_ids,
            "start_boundary_reason": self.start_boundary_reason,
            "end_boundary_reason": self.end_boundary_reason,
            "feature_vector": self.feature_vector,
            "local_scores": self.local_scores,
            "local_quality_score": round(self.local_quality_score, 3),
            "ai_score": round(self.ai_score, 3) if self.ai_score is not None else None,
            "explanations": self.explanations,
            "chapter_id": self.chapter_id,
            "story_unit_id": self.story_unit_id,
            "core_idea": self.core_idea,
            "content_signature": self.content_signature,
            "boundary_diagnostics": self.boundary_diagnostics,
            "incremental_coverage_score": round(self.incremental_coverage_score, 3),
        }


AI_FIELDS = (
    "start", "end", "title", "hook", "summary", "score", "hook_score",
    "completeness_score", "emotional_score", "clarity_score",
    "context_dependency_score", "rejection_reason", "selected",
)


@dataclass(slots=True)
class ScoredCandidate:
    candidate: Candidate
    title: str
    hook: str
    summary: str
    score: int
    hook_score: int
    completeness_score: int
    emotional_score: int
    clarity_score: int
    context_dependency_score: int
    rejection_reason: str | None
    selected: bool
    selection_reason: str | None = None
    selection_diagnostics: dict[str, Any] = field(default_factory=dict)
    virality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "title": self.title,
            "hook": self.hook,
            "summary": self.summary,
            "score": self.score,
            "hook_score": self.hook_score,
            "completeness_score": self.completeness_score,
            "emotional_score": self.emotional_score,
            "clarity_score": self.clarity_score,
            "context_dependency_score": self.context_dependency_score,
            "rejection_reason": self.rejection_reason,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "selection_diagnostics": self.selection_diagnostics,
            "virality": self.virality,
        }


def candidate_from_dict(data: dict[str, Any]) -> Candidate:
    return Candidate(
        id=str(data["id"]),
        start=float(data["start"]),
        end=float(data["end"]),
        text=str(data.get("text", "")),
        reason=str(data.get("reason", "")),
        transcript_segment_ids=[int(value) for value in data.get("transcript_segment_ids", [])],
        start_boundary_reason=str(data.get("start_boundary_reason", "")),
        end_boundary_reason=str(data.get("end_boundary_reason", "")),
        feature_vector=dict(data.get("feature_vector", {})),
        local_scores=dict(data.get("local_scores", {})),
        local_quality_score=float(data.get("local_quality_score", 0)),
        ai_score=float(data["ai_score"]) if data.get("ai_score") is not None else None,
        explanations=[str(value) for value in data.get("explanations", [])],
        chapter_id=str(data["chapter_id"]) if data.get("chapter_id") is not None else None,
        story_unit_id=str(data["story_unit_id"]) if data.get("story_unit_id") is not None else None,
        core_idea=str(data.get("core_idea", "")),
        content_signature=dict(data.get("content_signature", {})),
        boundary_diagnostics=dict(data.get("boundary_diagnostics", {})),
        incremental_coverage_score=float(data.get("incremental_coverage_score", 0)),
    )


def scored_from_dict(data: dict[str, Any]) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=candidate_from_dict(data),
        title=str(data.get("title", "")),
        hook=str(data.get("hook", "")),
        summary=str(data.get("summary", "")),
        score=int(data.get("score", 0)),
        hook_score=int(data.get("hook_score", 0)),
        completeness_score=int(data.get("completeness_score", 0)),
        emotional_score=int(data.get("emotional_score", 0)),
        clarity_score=int(data.get("clarity_score", 0)),
        context_dependency_score=int(data.get("context_dependency_score", 100)),
        rejection_reason=data.get("rejection_reason"),
        selected=bool(data.get("selected", False)),
        selection_reason=data.get("selection_reason"),
        selection_diagnostics=dict(data.get("selection_diagnostics", {})),
        virality=dict(data.get("virality", {})),
    )
