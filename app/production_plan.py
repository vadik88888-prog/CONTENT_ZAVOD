from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.errors import ProductionPlanError
from app.production_models import (
    AudioLayer,
    BoundaryDecision,
    DialogueSegment,
    NarrationSegment,
    ProductionPlanEnvelope,
    ProductionPlanIdentity,
    ProductionPlanInputFingerprints,
    ProductionPlanPreset,
    ProductionPlanTarget,
    PauseSegment,
    ProductionMetadata,
    ProductionPlan,
    SubtitleCue,
    SubtitleTrack,
    SourceSegmentRange,
    TimelineEntry,
    TimelineEstimate,
    VoiceProfile,
)
from app.transformation_models import validate_final_script
from app.utils import stable_text_hash, utc_now


PRODUCTION_PLAN_VERSION = "5F.1"
LEGACY_PRODUCTION_PLAN_VERSION = "3A.2"
TIMELINE_VERSION = "3A.0"


@dataclass(frozen=True, slots=True)
class ProductionPlanEnvelopeContext:
    """Pipeline-owned immutable inputs needed to create a native 5F plan."""

    project_id: str
    run_id: str
    analysis_id: str
    analysis_fingerprint: str
    source_sha256: str
    transcript_sha256: str
    preset_id: str
    preset_version: str
    platform: str
    target_width: int
    target_height: int
    target_fps: float
    created_at: str | None = None

    def build(
        self, *, candidate_id: str, source_id: str, final_script_hash: str,
        boundary_decision: BoundaryDecision | None,
    ) -> ProductionPlanEnvelope:
        if boundary_decision is None:
            raise ProductionPlanError("EDIT_PLAN_SCHEMA_INVALID: native ProductionPlan requires a BoundaryDecision.")
        return ProductionPlanEnvelope(
            identity=ProductionPlanIdentity(
                project_id=self.project_id,
                run_id=self.run_id,
                analysis_id=self.analysis_id,
                candidate_id=candidate_id,
                source_id=source_id,
            ),
            boundary_decision_ref=boundary_decision.decision_id,
            preset=ProductionPlanPreset(
                preset_id=self.preset_id,
                preset_version=self.preset_version,
                platform=self.platform,  # type: ignore[arg-type]
            ),
            target=ProductionPlanTarget(
                width=self.target_width,
                height=self.target_height,
                fps=self.target_fps,
            ),
            input_fingerprints=ProductionPlanInputFingerprints(
                source_sha256=self.source_sha256,
                transcript_sha256=self.transcript_sha256,
                analysis_sha256=self.analysis_fingerprint,
                final_script_sha256=final_script_hash,
                boundary_decision_sha256=stable_text_hash(boundary_decision.model_dump_json()),
            ),
            created_at=self.created_at or utc_now(),
        )

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)


def build_production_plan(
    transformation: dict[str, Any], production_config: Any,
    *, envelope_context: ProductionPlanEnvelopeContext | None = None,
) -> ProductionPlan:
    """Build the Goal 3A source of truth without generating any media."""

    final = _dict(transformation.get("final_script"), "FinalScript")
    context = _dict(transformation.get("source_context"), "SourceContext")
    semantic = _dict(transformation.get("semantic_representation"), "SemanticRepresentation")
    expected_candidate_id = str(transformation.get("candidate_id") or context.get("candidate_id") or "")
    validation = validate_final_script(final, context, semantic, expected_candidate_id)
    if not validation.passed:
        raise ProductionPlanError(_final_script_diagnostic(transformation, final, expected_candidate_id, validation.errors))
    candidate_id = str(final.get("candidate_id") or "")
    language = str(final.get("language") or context.get("language") or "unknown")
    sentences = [item for item in final.get("sentences", []) if isinstance(item, dict) and str(item.get("text", "")).strip()]
    if not candidate_id or not sentences:
        raise ProductionPlanError("ProductionPlan требует FinalScript с candidate_id и хотя бы одним предложением.")
    boundary_decision = _boundary_decision_from_context(context, candidate_id)
    facts = {
        str(item.get("fact_id")): item
        for item in semantic.get("supporting_facts", [])
        if isinstance(item, dict) and item.get("fact_id")
    }
    evidence = {
        int(item.get("segment_id")): item
        for item in [*context.get("primary_evidence", []), *context.get("supporting_context", [])]
        if isinstance(item, dict) and item.get("segment_id") is not None
    }
    voice = VoiceProfile(
        profile_id=str(production_config.voice_profile_id),
        gender=str(production_config.voice_gender),
        style=str(production_config.voice_style),
        language=language,
    )
    segments: list[Any] = []
    dialogue_mappings: list[DialogueSegment] = []
    narration_to_dialogue: dict[str, list[str]] = {}
    used_dialogue_fact_ids: set[str] = set()
    audio_mode = str(getattr(production_config, "audio_mode", "original"))
    source_audio_mode = audio_mode in {"original", "original_enhanced"}
    # A transformed FinalScript is evidence for selection, never implicit consent
    # to replace the speaker. Original modes always remain dialogue-only.
    dialogue_only = source_audio_mode or final.get("production_ready_for_tts") is False
    order = 1
    for index, sentence in enumerate(sentences):
        sentence_id = str(sentence.get("sentence_id") or f"sentence-{index + 1:03d}")
        text = str(sentence["text"]).strip()
        fact_ids = [str(item) for item in sentence.get("supported_by_fact_ids", []) if str(item) in facts]
        source_ids = [int(item) for item in sentence.get("source_segment_ids", []) if int(item) in evidence]
        if not fact_ids or not source_ids:
            raise ProductionPlanError(f"FinalScript sentence {sentence_id} не имеет подтверждённого fact/transcript mapping.")
        if dialogue_only:
            for fact_id in fact_ids:
                if fact_id in used_dialogue_fact_ids:
                    continue
                fact = facts[fact_id]
                source_id = _first_known_segment_id(fact, evidence, source_ids)
                source = evidence[source_id]
                dialogue = DialogueSegment(
                    segment_id=f"dialogue-{len(dialogue_mappings) + 1:03d}",
                    order=order,
                    estimated_duration_seconds=max(
                        0.0,
                        float(fact.get("evidence_end", source.get("end", 0)))
                        - float(fact.get("evidence_start", source.get("start", 0))),
                    ),
                    fact_id=fact_id,
                    transcript_segment_id=source_id,
                    source_start_seconds=float(fact.get("evidence_start", source.get("start", 0))),
                    source_end_seconds=float(fact.get("evidence_end", source.get("end", 0))),
                    source_text=str(source.get("text") or fact.get("evidence_quote") or fact.get("statement")),
                    speaker=str(production_config.original_dialogue_speaker),
                    confidence=float(fact.get("confidence", 0.0)),
                    boundary_decision_id=boundary_decision.decision_id if boundary_decision else None,
                    linked_segment_ids=[],
                )
                segments.append(dialogue)
                dialogue_mappings.append(dialogue)
                used_dialogue_fact_ids.add(fact_id)
                order += 1
            continue
        narration_id = f"narration-{index + 1:03d}"
        narration = NarrationSegment(
            segment_id=narration_id,
            order=order,
            estimated_duration_seconds=_duration(text, production_config.narration_words_per_second),
            text=text,
            narration_role=_narration_role(index, len(sentences), str(sentence.get("role", "claim"))),
            source_sentence_id=sentence_id,
            fact_ids=fact_ids,
            source_segment_ids=source_ids,
            source_ranges=_source_ranges_for_facts(fact_ids, source_ids, facts, evidence),
            word_count=_word_count(text),
            words_per_second=float(production_config.narration_words_per_second),
            voice_profile_id=voice.profile_id,
            boundary_decision_id=boundary_decision.decision_id if boundary_decision else None,
        )
        segments.append(narration)
        narration_to_dialogue[narration_id] = []
        order += 1
        for fact_id in fact_ids:
            if fact_id in used_dialogue_fact_ids:
                continue
            fact = facts[fact_id]
            segment_id = _first_known_segment_id(fact, evidence, source_ids)
            source = evidence[segment_id]
            dialogue_id = f"dialogue-{len(dialogue_mappings) + 1:03d}"
            dialogue = DialogueSegment(
                segment_id=dialogue_id,
                order=order,
                estimated_duration_seconds=max(0.0, float(fact.get("evidence_end", source.get("end", 0))) - float(fact.get("evidence_start", source.get("start", 0)))),
                fact_id=fact_id,
                transcript_segment_id=segment_id,
                source_start_seconds=float(fact.get("evidence_start", source.get("start", 0))),
                source_end_seconds=float(fact.get("evidence_end", source.get("end", 0))),
                source_text=str(source.get("text") or fact.get("evidence_quote") or fact.get("statement")),
                speaker=str(production_config.original_dialogue_speaker),
                confidence=float(fact.get("confidence", 0.0)),
                boundary_decision_id=boundary_decision.decision_id if boundary_decision else None,
                linked_segment_ids=[narration_id],
            )
            segments.append(dialogue)
            dialogue_mappings.append(dialogue)
            narration_to_dialogue[narration_id].append(dialogue_id)
            used_dialogue_fact_ids.add(fact_id)
            order += 1
        if index < len(sentences) - 1:
            reason = "intro_breath" if index == 0 else "outro_breath" if index == len(sentences) - 2 else "narration_transition"
            segments.append(PauseSegment(
                segment_id=f"pause-{index + 1:03d}",
                order=order,
                estimated_duration_seconds=float(production_config.pause_after_narration_seconds),
                reason=reason,
            ))
            order += 1
    for segment in segments:
        if isinstance(segment, NarrationSegment):
            segment.linked_segment_ids = narration_to_dialogue[segment.segment_id]
    timeline = _build_timeline(segments)
    subtitle_track = _build_subtitle_track(segments, timeline, language)
    source = _dict(context.get("source"), "SourceContext.source")
    final_script_hash = stable_text_hash(json.dumps(final, ensure_ascii=False, sort_keys=True))
    native_envelope = envelope_context.build(
        candidate_id=candidate_id,
        source_id=str(source.get("id", "")),
        final_script_hash=final_script_hash,
        boundary_decision=boundary_decision,
    ) if envelope_context is not None else None
    try:
        return ProductionPlan(
            plan_id=f"production-{candidate_id}-{final_script_hash[:12]}",
            schema_version=PRODUCTION_PLAN_VERSION if native_envelope is not None else LEGACY_PRODUCTION_PLAN_VERSION,
            envelope=native_envelope,
            segments=segments,
            dialogue_mappings=dialogue_mappings,
            timeline=timeline,
            voice_profile=voice,
            audio_layers=[
                AudioLayer(layer_id="layer-narration", layer_type="narration"),
                AudioLayer(layer_id="layer-original-dialogue", layer_type="original_dialogue"),
                AudioLayer(layer_id="layer-music", layer_type="music"),
                AudioLayer(layer_id="layer-effects", layer_type="effects"),
            ],
            subtitle_track=subtitle_track,
            metadata=ProductionMetadata(
                plan_version=PRODUCTION_PLAN_VERSION if native_envelope is not None else LEGACY_PRODUCTION_PLAN_VERSION,
                candidate_id=candidate_id,
                source_id=str(source.get("id", "")),
                final_script_hash=final_script_hash,
            ),
            audio_mode=audio_mode,
            tts_eligible=not dialogue_only and audio_mode in {"voiceover", "replace_voice", "mixed"},
            audio_mode_reason="source_audio_mode" if source_audio_mode else "explicit_voiceover_intent",
            boundary_decision=boundary_decision,
        )
    except ValueError as error:
        raise ProductionPlanError(str(error)) from error


def production_summary(plan: ProductionPlan) -> str:
    timeline = plan.timeline
    return "\n".join([
        f"Production Plan: {plan.plan_id}",
        f"Candidate: {plan.metadata.candidate_id}",
        f"Estimated timeline: {timeline.estimated_duration_seconds:.2f} s",
        f"Narration segments: {timeline.narration_count}",
        f"Original dialogue placeholders: {timeline.dialogue_count}",
        f"Pause segments: {timeline.pause_count}",
        "Audio/TTS/render: placeholders only; no media was generated.",
    ]) + "\n"


def _build_timeline(segments: list[Any]) -> TimelineEstimate:
    cursor = 0.0
    entries: list[TimelineEntry] = []
    active_ranges: dict[str, tuple[float, float]] = {}
    for segment in segments:
        if segment.timeline_included:
            start = cursor
            end = round(start + segment.estimated_duration_seconds, 3)
            cursor = end
            active_ranges[segment.segment_id] = (start, end)
        else:
            linked = segment.linked_segment_ids[0] if segment.linked_segment_ids else ""
            start, end = active_ranges.get(linked, (cursor, cursor))
        entries.append(TimelineEntry(
            segment_id=segment.segment_id,
            order=segment.order,
            estimated_start_seconds=start,
            estimated_end_seconds=end,
            included_in_master_timeline=segment.timeline_included,
            linked_segment_ids=segment.linked_segment_ids,
        ))
    return TimelineEstimate(
        timeline_version=TIMELINE_VERSION,
        estimated_duration_seconds=cursor,
        narration_count=sum(isinstance(item, NarrationSegment) for item in segments),
        dialogue_count=sum(isinstance(item, DialogueSegment) for item in segments),
        pause_count=sum(isinstance(item, PauseSegment) for item in segments),
        entries=entries,
    )


def _build_subtitle_track(segments: list[Any], timeline: TimelineEstimate, language: str) -> SubtitleTrack:
    by_segment = {entry.segment_id: entry for entry in timeline.entries}
    cues = [
        SubtitleCue(
            cue_id=f"subtitle-{index + 1:03d}", text=segment.text,
            estimated_start_seconds=by_segment[segment.segment_id].estimated_start_seconds,
            estimated_end_seconds=by_segment[segment.segment_id].estimated_end_seconds,
            speaker="narrator", segment_id=segment.segment_id,
        )
        for index, segment in enumerate(segments) if isinstance(segment, NarrationSegment)
    ]
    return SubtitleTrack(track_id="subtitle-track-placeholder", language=language, cues=cues)


def _boundary_decision_from_context(context: dict[str, Any], candidate_id: str) -> BoundaryDecision | None:
    """Read a 5C decision without making legacy draft artifacts unusable."""

    raw = context.get("boundary_decision")
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ProductionPlanError("BOUNDARY_DECISION_INVALID: SourceContext boundary_decision must be an object.")
    try:
        decision = BoundaryDecision.model_validate(raw)
    except Exception as error:
        raise ProductionPlanError(f"BOUNDARY_DECISION_INVALID: {error}") from error
    if decision.candidate_id != candidate_id:
        raise ProductionPlanError(
            "BOUNDARY_CANDIDATE_MISMATCH: SourceContext boundary decision belongs to another candidate."
        )
    return decision


def _source_ranges_for_facts(
    fact_ids: list[str], source_ids: list[int], facts: dict[str, dict[str, Any]], evidence: dict[int, dict[str, Any]],
) -> list[SourceSegmentRange]:
    ranges: list[SourceSegmentRange] = []
    seen: set[tuple[int, float, float]] = set()
    for fact_id in fact_ids:
        fact = facts[fact_id]
        source_id = _first_known_segment_id(fact, evidence, source_ids)
        source = evidence[source_id]
        start = float(fact.get("evidence_start", source.get("start", 0)))
        end = float(fact.get("evidence_end", source.get("end", 0)))
        key = (source_id, start, end)
        if end > start and key not in seen:
            ranges.append(SourceSegmentRange(
                transcript_segment_id=source_id,
                source_start_seconds=start,
                source_end_seconds=end,
            ))
            seen.add(key)
    return ranges


def _first_known_segment_id(fact: dict[str, Any], evidence: dict[int, dict[str, Any]], source_ids: list[int]) -> int:
    ids = [int(item) for item in fact.get("evidence_segment_ids", [])]
    for identifier in [*ids, *source_ids]:
        if identifier in evidence:
            return identifier
    raise ProductionPlanError(f"Fact {fact.get('fact_id')} не связан с известным transcript segment.")


def _narration_role(index: int, total: int, source_role: str) -> str:
    if source_role == "cta":
        return "cta"
    if index == 0:
        return "intro"
    if index == total - 1:
        return "outro"
    return "body"


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _duration(text: str, words_per_second: float) -> float:
    return round(_word_count(text) / max(0.1, words_per_second), 3)


def _dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ProductionPlanError(f"{name} отсутствует или не имеет ожидаемой структуры.")
    return value


def _final_script_diagnostic(
    transformation: dict[str, Any], final: dict[str, Any], expected_candidate_id: str, errors: list[str],
) -> str:
    actual_candidate_id = str(final.get("candidate_id") or "")
    raw_sentences = final.get("sentences", [])
    sentences_count = len(raw_sentences) if isinstance(raw_sentences, list) else 0
    source = str(transformation.get("final_script_source") or "unknown")
    compact_errors = "; ".join(str(item) for item in errors[:5]) or "unknown validation error"
    return (
        "ProductionPlan rejected FinalScript contract: "
        f"expected_candidate_id={expected_candidate_id or '<missing>'}; "
        f"actual_candidate_id={actual_candidate_id or '<missing>'}; "
        f"sentences_count={sentences_count}; source={source}; validation_errors={compact_errors}"
    )
