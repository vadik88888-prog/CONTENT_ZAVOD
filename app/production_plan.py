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
    # One transcript range may ground more than one fact, but original-audio
    # delivery must not replay it as sequential dialogue.  Keep the first
    # deterministic mapping and preserve every suppression as plan evidence.
    used_dialogue_source_ranges: set[tuple[float, float]] = set()
    suppressed_duplicate_dialogue_ranges: list[tuple[str, float, float]] = []
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
                source_start = float(fact.get("evidence_start", source.get("start", 0)))
                source_end = float(fact.get("evidence_end", source.get("end", 0)))
                source_range = (source_start, source_end)
                if source_range in used_dialogue_source_ranges:
                    used_dialogue_fact_ids.add(fact_id)
                    suppressed_duplicate_dialogue_ranges.append((fact_id, source_start, source_end))
                    continue
                dialogue = DialogueSegment(
                    segment_id=f"dialogue-{len(dialogue_mappings) + 1:03d}",
                    order=order,
                    estimated_duration_seconds=max(
                        0.0,
                        source_end - source_start,
                    ),
                    fact_id=fact_id,
                    transcript_segment_id=source_id,
                    source_start_seconds=source_start,
                    source_end_seconds=source_end,
                    source_text=str(source.get("text") or fact.get("evidence_quote") or fact.get("statement")),
                    speaker=str(production_config.original_dialogue_speaker),
                    confidence=float(fact.get("confidence", 0.0)),
                    boundary_decision_id=boundary_decision.decision_id if boundary_decision else None,
                    linked_segment_ids=[],
                )
                segments.append(dialogue)
                dialogue_mappings.append(dialogue)
                used_dialogue_fact_ids.add(fact_id)
                used_dialogue_source_ranges.add(source_range)
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
            source_start = float(fact.get("evidence_start", source.get("start", 0)))
            source_end = float(fact.get("evidence_end", source.get("end", 0)))
            source_range = (source_start, source_end)
            if source_range in used_dialogue_source_ranges:
                used_dialogue_fact_ids.add(fact_id)
                suppressed_duplicate_dialogue_ranges.append((fact_id, source_start, source_end))
                continue
            dialogue_id = f"dialogue-{len(dialogue_mappings) + 1:03d}"
            dialogue = DialogueSegment(
                segment_id=dialogue_id,
                order=order,
                estimated_duration_seconds=max(0.0, source_end - source_start),
                fact_id=fact_id,
                transcript_segment_id=segment_id,
                source_start_seconds=source_start,
                source_end_seconds=source_end,
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
            used_dialogue_source_ranges.add(source_range)
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
    _register_grounded_dialogue_boundaries(
        dialogue_mappings, boundary_decision=boundary_decision, evidence=evidence,
    )
    story_continuity_warnings = _preserve_story_continuity(
        dialogue_mappings,
        source_audio_mode=source_audio_mode,
        content_type=str(semantic.get("content_type") or ""),
        context=context,
        boundary_decision=boundary_decision,
    )
    boundary_padding_warnings = _apply_boundary_padding(dialogue_mappings, boundary_decision)
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
    if native_envelope is not None and suppressed_duplicate_dialogue_ranges:
        native_envelope = native_envelope.model_copy(update={
            "warnings": [
                *native_envelope.warnings,
                *[
                    "DUPLICATE_EXACT_SOURCE_RANGE_SUPPRESSED:"
                    f"{fact_id}:{source_start:.3f}-{source_end:.3f}"
                    for fact_id, source_start, source_end in suppressed_duplicate_dialogue_ranges
                ],
            ],
        })
    if native_envelope is not None and boundary_padding_warnings:
        native_envelope = native_envelope.model_copy(update={
            "warnings": [*native_envelope.warnings, *boundary_padding_warnings],
        })
    if native_envelope is not None and story_continuity_warnings:
        native_envelope = native_envelope.model_copy(update={
            "warnings": [*native_envelope.warnings, *story_continuity_warnings],
        })
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
            composition_intent=dict(context.get("composition_intent") or {}),
        )
    except ValueError as error:
        raise ProductionPlanError(str(error)) from error


def _register_grounded_dialogue_boundaries(
    dialogue_mappings: list[DialogueSegment],
    *,
    boundary_decision: BoundaryDecision | None,
    evidence: dict[int, dict[str, Any]],
) -> None:
    """Promote only exact transcript evidence edges to safe edit points."""

    if boundary_decision is None:
        return
    safe_starts = set(boundary_decision.safe_start_points)
    safe_ends = set(boundary_decision.safe_end_points)
    for dialogue in dialogue_mappings:
        source = evidence.get(dialogue.transcript_segment_id)
        if not isinstance(source, dict):
            continue
        try:
            evidence_start = float(source["start"])
            evidence_end = float(source["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            abs(dialogue.source_start_seconds - evidence_start) <= 0.001
            and abs(dialogue.source_end_seconds - evidence_end) <= 0.001
        ):
            safe_starts.add(dialogue.source_start_seconds)
            safe_ends.add(dialogue.source_end_seconds)
    boundary_decision.safe_start_points = sorted(safe_starts)
    boundary_decision.safe_end_points = sorted(safe_ends)


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


def _apply_boundary_padding(
    dialogue_mappings: list[DialogueSegment], decision: BoundaryDecision | None,
) -> list[str]:
    """Keep approved word-safe pre/post-roll in the source-audio hand-off.

    Facts generally start and end on words; a selected BoundaryDecision can
    additionally retain a safe silence before the hook and after the payoff.
    Without this hand-off the subtitle fitter has no legitimate tail time to
    finish a fast final phrase, despite the approved source boundary providing
    it.  The first/last *source-time* mappings are used rather than script
    order, so an editorially reordered FinalScript remains source-safe.
    """

    if decision is None or not dialogue_mappings:
        return []
    # Padding is presentation-only.  It must never turn a plan that omitted a
    # required hook/completion/payoff into an apparently valid one.
    if not _required_boundary_evidence_is_covered(dialogue_mappings, decision):
        return []
    refined = decision.refined_range
    first = min(dialogue_mappings, key=lambda item: (item.source_start_seconds, item.order))
    last = max(dialogue_mappings, key=lambda item: (item.source_end_seconds, -item.order))
    warnings: list[str] = []
    if refined.start_seconds < first.source_start_seconds:
        previous = first.source_start_seconds
        first.source_start_seconds = refined.start_seconds
        first.estimated_duration_seconds = max(0.0, first.source_end_seconds - first.source_start_seconds)
        warnings.append(
            "BOUNDARY_PRE_ROLL_APPLIED:"
            f"{first.segment_id}:{previous:.3f}->{first.source_start_seconds:.3f}"
        )
    preserve_until = float(decision.multimodal_context.get("preserve_until_seconds", refined.end_seconds))
    if not decision.multimodal_context.get("multimodal_payoff_grounded"):
        preserve_until = refined.end_seconds
    target_end = min(decision.allowed_source_range.end_seconds, max(refined.end_seconds, preserve_until))
    if target_end > last.source_end_seconds:
        previous = last.source_end_seconds
        last.source_end_seconds = target_end
        last.estimated_duration_seconds = max(0.0, last.source_end_seconds - last.source_start_seconds)
        warnings.append(
            ("MULTIMODAL_PAYOFF_POST_ROLL_APPLIED:" if target_end > refined.end_seconds else "BOUNDARY_POST_ROLL_APPLIED:") +
            f"{last.segment_id}:{previous:.3f}->{last.source_end_seconds:.3f}"
        )
    return warnings


def _preserve_story_continuity(
    dialogue_mappings: list[DialogueSegment],
    *,
    source_audio_mode: bool,
    content_type: str,
    context: dict[str, Any],
    boundary_decision: BoundaryDecision | None,
) -> list[str]:
    """Keep causal source bridges instead of concatenating story islands.

    Semantic facts prove what may be used, but their evidence spans are not an
    edit decision to remove everything between setup and payoff.  In original
    audio modes a story therefore keeps the source interval between consecutive
    grounded facts.  This is bounded by the already-approved candidate and
    BoundaryDecision; no new source material is introduced outside that range.
    """

    if not source_audio_mode or len(dialogue_mappings) < 2:
        return []
    gaps = [
        max(0.0, right.source_start_seconds - left.source_end_seconds)
        for left, right in zip(dialogue_mappings, dialogue_mappings[1:])
    ]
    candidate_duration = max(
        0.0, float(context.get("end_time", 0)) - float(context.get("start_time", 0)),
    )
    multimodal = context.get("multimodal_context")
    scene_boundaries = context.get("scene_boundaries")
    visual_story_bridge = (
        max(gaps, default=0.0) >= max(3.0, candidate_duration * 0.35)
        and (
            isinstance(multimodal, dict)
            and multimodal.get("multimodal_payoff_grounded") is True
            or isinstance(scene_boundaries, list) and len(scene_boundaries) >= 2
        )
    )
    if content_type != "story" and not visual_story_bridge:
        return []
    warnings: list[str] = []
    for left, right in zip(dialogue_mappings, dialogue_mappings[1:]):
        if right.source_end_seconds < left.source_start_seconds - 0.001:
            raise ProductionPlanError(
                "STORY_SOURCE_ORDER_INVALID: source-audio story facts must remain chronological."
            )
        if right.source_start_seconds <= left.source_end_seconds + 0.001:
            continue
        previous = left.source_end_seconds
        left.source_end_seconds = right.source_start_seconds
        left.estimated_duration_seconds = max(
            0.0, left.source_end_seconds - left.source_start_seconds,
        )
        # The adjacent mappings now meet at the already-approved start of the
        # next complete word.  Since no source time is skipped at that join,
        # the same point is also a valid end boundary for the preceding bridge.
        if (
            boundary_decision is not None
            and any(
                abs(right.source_start_seconds - point) <= 0.001
                for point in boundary_decision.safe_start_points
            )
        ):
            boundary_decision.safe_end_points = sorted({
                *boundary_decision.safe_end_points,
                right.source_start_seconds,
            })
        warnings.append(
            "STORY_CAUSAL_BRIDGE_PRESERVED:"
            f"{left.segment_id}:{previous:.3f}->{left.source_end_seconds:.3f}"
        )
    return warnings


def _required_boundary_evidence_is_covered(
    dialogue_mappings: list[DialogueSegment], decision: BoundaryDecision,
) -> bool:
    ranges = sorted(
        ((item.source_start_seconds, item.source_end_seconds) for item in dialogue_mappings),
        key=lambda item: item[0],
    )
    for requirement in decision.required_evidence:
        if not requirement.required:
            continue
        cursor = requirement.source_range.start_seconds
        for start, end in ranges:
            if end < cursor - 0.001:
                continue
            if start > cursor + 0.001:
                break
            cursor = max(cursor, end)
            if cursor >= requirement.source_range.end_seconds - 0.001:
                break
        if cursor < requirement.source_range.end_seconds - 0.001:
            return False
    return True


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
