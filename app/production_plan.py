from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.errors import ProductionPlanError
from app.continuity import build_continuity_decision
from app.production_models import (
    AudioLayer,
    BoundaryDecision,
    ContinuityDecision,
    DialogueEvidenceMapping,
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
        continuity_decision: ContinuityDecision | None,
    ) -> ProductionPlanEnvelope:
        if boundary_decision is None:
            raise ProductionPlanError("EDIT_PLAN_SCHEMA_INVALID: native ProductionPlan requires a BoundaryDecision.")
        if continuity_decision is None:
            raise ProductionPlanError("EDIT_PLAN_SCHEMA_INVALID: native ProductionPlan requires a ContinuityDecision.")
        return ProductionPlanEnvelope(
            identity=ProductionPlanIdentity(
                project_id=self.project_id,
                run_id=self.run_id,
                analysis_id=self.analysis_id,
                candidate_id=candidate_id,
                source_id=source_id,
            ),
            boundary_decision_ref=boundary_decision.decision_id,
            continuity_decision_ref=continuity_decision.decision_id,
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
                continuity_decision_sha256=continuity_decision.fingerprint(),
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
                    evidence_mappings=[DialogueEvidenceMapping(
                        fact_id=fact_id,
                        transcript_segment_id=source_id,
                        source_start_seconds=source_start,
                        source_end_seconds=source_end,
                        source_text=str(source.get("text") or fact.get("evidence_quote") or fact.get("statement")),
                        confidence=float(fact.get("confidence", 0.0)),
                    )],
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
                evidence_mappings=[DialogueEvidenceMapping(
                    fact_id=fact_id,
                    transcript_segment_id=segment_id,
                    source_start_seconds=source_start,
                    source_end_seconds=source_end,
                    source_text=str(source.get("text") or fact.get("evidence_quote") or fact.get("statement")),
                    confidence=float(fact.get("confidence", 0.0)),
                )],
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
    continuity_decision = _continuity_decision_from_context(
        context, candidate_id, boundary_decision,
    )
    _validate_grounded_dialogue_evidence(
        dialogue_mappings, boundary_decision, continuity_decision,
    )
    if source_audio_mode:
        segments, dialogue_mappings = _build_continuous_source_dialogue(
            dialogue_mappings,
            boundary_decision=boundary_decision,
            continuity_decision=continuity_decision,
        )
        continuity_warnings = []
    else:
        continuity_warnings = _apply_continuity_required_spans(
            dialogue_mappings,
            continuity_decision=continuity_decision,
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
        continuity_decision=continuity_decision,
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
    if native_envelope is not None and continuity_warnings:
        native_envelope = native_envelope.model_copy(update={
            "warnings": [*native_envelope.warnings, *continuity_warnings],
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
            continuity_decision=continuity_decision,
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


def _validate_grounded_dialogue_evidence(
    dialogue_mappings: list[DialogueSegment],
    decision: BoundaryDecision | None,
    continuity_decision: ContinuityDecision | None,
) -> None:
    """Validate exact ASR/fact provenance before source media is consolidated."""

    if decision is None:
        return
    allowed = decision.allowed_source_range
    ranges = []
    errors: list[str] = []
    for dialogue in dialogue_mappings:
        start = dialogue.source_start_seconds
        end = dialogue.source_end_seconds
        ranges.append((start, end))
        if start < allowed.start_seconds - 0.001 or end > allowed.end_seconds + 0.001:
            errors.append(f"BOUNDARY_SOURCE_RANGE_OUTSIDE:{dialogue.segment_id}")
    if continuity_decision is not None:
        ranges.extend(
            (span.source_range.start_seconds, span.source_range.end_seconds)
            for span in continuity_decision.required_spans
        )
    for requirement in decision.required_evidence:
        if requirement.required and not _tuple_range_is_covered(
            requirement.source_range.start_seconds,
            requirement.source_range.end_seconds,
            ranges,
        ):
            errors.append(f"BOUNDARY_{requirement.requirement_type.upper()}_LOST")
    if errors:
        raise ProductionPlanError("; ".join(dict.fromkeys(errors)))


def _tuple_range_is_covered(
    required_start: float, required_end: float, ranges: list[tuple[float, float]],
) -> bool:
    cursor = required_start
    for start, end in sorted(ranges):
        if end < cursor - 0.001:
            continue
        if start > cursor + 0.001:
            break
        cursor = max(cursor, end)
        if cursor >= required_end - 0.001:
            return True
    return False


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


def _apply_continuity_required_spans(
    dialogue_mappings: list[DialogueSegment],
    *,
    continuity_decision: ContinuityDecision | None,
) -> list[str]:
    """Retain only explicitly evidence-backed spans, never a whole boundary.

    The existing source-audio pipeline renders ``DialogueSegment`` intervals.
    Extending the immediately preceding/next dialogue mapping is therefore the
    narrowest compatible way to retain a required visual action, semantic
    bridge, reaction, or payoff.  The distinct ContinuityDecision authorizes
    its endpoints; BoundaryDecision remains unchanged.
    """

    if continuity_decision is None or not continuity_decision.required_spans:
        return []
    if not dialogue_mappings:
        raise ProductionPlanError("CONTINUITY_REQUIRED_SPAN_UNMAPPABLE: no source dialogue mappings.")
    warnings: list[str] = []
    for requirement in continuity_decision.required_spans:
        source_range = requirement.source_range
        if _source_range_covered(dialogue_mappings, source_range.start_seconds, source_range.end_seconds):
            continue
        ordered = sorted(dialogue_mappings, key=lambda item: (item.source_start_seconds, item.source_end_seconds))
        before = [item for item in ordered if item.source_end_seconds <= source_range.start_seconds + 0.001]
        after = [item for item in ordered if item.source_start_seconds >= source_range.end_seconds - 0.001]
        if before:
            target = before[-1]
            previous = target.source_end_seconds
            target.source_end_seconds = max(target.source_end_seconds, source_range.end_seconds)
            target.estimated_duration_seconds = max(0.0, target.source_end_seconds - target.source_start_seconds)
            warnings.append(
                "CONTINUITY_REQUIRED_SPAN_PRESERVED:"
                f"{requirement.requirement_type}:{target.segment_id}:{previous:.3f}->{target.source_end_seconds:.3f}"
            )
        elif after:
            target = after[0]
            previous = target.source_start_seconds
            target.source_start_seconds = min(target.source_start_seconds, source_range.start_seconds)
            target.estimated_duration_seconds = max(0.0, target.source_end_seconds - target.source_start_seconds)
            warnings.append(
                "CONTINUITY_REQUIRED_SPAN_PRESERVED:"
                f"{requirement.requirement_type}:{target.segment_id}:{previous:.3f}->{target.source_start_seconds:.3f}"
            )
        else:
            raise ProductionPlanError(
                "CONTINUITY_REQUIRED_SPAN_UNMAPPABLE: "
                f"{requirement.requirement_type}:{source_range.start_seconds:.3f}-{source_range.end_seconds:.3f}"
            )
    return warnings


def _build_continuous_source_dialogue(
    dialogue_mappings: list[DialogueSegment],
    *,
    boundary_decision: BoundaryDecision | None,
    continuity_decision: ContinuityDecision | None,
) -> tuple[list[Any], list[DialogueSegment]]:
    """Separate exact evidence geometry from source-audio edit geometry.

    Original-audio delivery retains the approved source interval as one
    continuous media segment by default.  Only a persisted, typed omission
    rationale authorizes an internal physical cut; an ``unexplained`` ASR gap
    remains in the media and is left for A-2 to verify against the render map.
    """

    if boundary_decision is None or not dialogue_mappings:
        return list(dialogue_mappings), dialogue_mappings
    approved = (
        continuity_decision.approved_source_range
        if continuity_decision is not None else boundary_decision.refined_range
    )
    explained_omissions = [
        span for span in (continuity_decision.omitted_spans if continuity_decision else [])
        if span.authorizes_physical_cut()
    ]
    for omission in explained_omissions:
        if continuity_decision and any(
            _ranges_overlap(omission.source_range, required.source_range)
            for required in continuity_decision.required_spans
        ):
            raise ProductionPlanError(
                "CONTINUITY_OMISSION_CONFLICT_REQUIRED_SPAN: "
                f"{omission.source_range.start_seconds:.3f}-{omission.source_range.end_seconds:.3f}"
            )
    retained = [(approved.start_seconds, approved.end_seconds)]
    for omission in sorted(
        explained_omissions,
        key=lambda item: (item.source_range.start_seconds, item.source_range.end_seconds),
    ):
        retained = _subtract_interval(
            retained,
            omission.source_range.start_seconds,
            omission.source_range.end_seconds,
        )
    if not retained:
        raise ProductionPlanError("CONTINUITY_SOURCE_BOUNDARY_FULLY_OMITTED")

    exact_evidence = [
        evidence
        for dialogue in dialogue_mappings
        for evidence in (
            dialogue.evidence_mappings
            or [DialogueEvidenceMapping(
                fact_id=dialogue.fact_id,
                transcript_segment_id=dialogue.transcript_segment_id,
                source_start_seconds=dialogue.source_start_seconds,
                source_end_seconds=dialogue.source_end_seconds,
                source_text=dialogue.source_text,
                confidence=dialogue.confidence,
            )]
        )
    ]
    assignments: list[list[DialogueEvidenceMapping]] = [[] for _ in retained]
    for evidence in exact_evidence:
        midpoint = (evidence.source_start_seconds + evidence.source_end_seconds) / 2
        target_index = min(
            range(len(retained)),
            key=lambda index: _distance_to_interval(midpoint, retained[index]),
        )
        assignments[target_index].append(evidence)

    continuous: list[DialogueSegment] = []
    for index, ((start, end), evidence_items) in enumerate(zip(retained, assignments), start=1):
        representative = min(
            evidence_items or exact_evidence,
            key=lambda item: (item.source_start_seconds, item.source_end_seconds, item.fact_id),
        )
        source_text = " ".join(dict.fromkeys(item.source_text for item in evidence_items))
        continuous.append(DialogueSegment(
            segment_id=f"dialogue-{index:03d}",
            order=index,
            estimated_duration_seconds=max(0.0, end - start),
            fact_id=representative.fact_id,
            transcript_segment_id=representative.transcript_segment_id,
            source_start_seconds=start,
            source_end_seconds=end,
            source_text=source_text or representative.source_text,
            speaker=dialogue_mappings[0].speaker,
            confidence=min(
                (item.confidence for item in evidence_items),
                default=representative.confidence,
            ),
            evidence_mappings=evidence_items,
            boundary_decision_id=boundary_decision.decision_id,
            linked_segment_ids=[],
        ))
    return list(continuous), continuous


def _subtract_interval(
    retained: list[tuple[float, float]], omission_start: float, omission_end: float,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in retained:
        if omission_end <= start + 0.001 or omission_start >= end - 0.001:
            result.append((start, end))
            continue
        if omission_start > start + 0.001:
            result.append((start, min(end, omission_start)))
        if omission_end < end - 0.001:
            result.append((max(start, omission_end), end))
    return result


def _ranges_overlap(left: Any, right: Any) -> bool:
    return (
        left.start_seconds < right.end_seconds - 0.001
        and right.start_seconds < left.end_seconds - 0.001
    )


def _distance_to_interval(value: float, interval: tuple[float, float]) -> float:
    start, end = interval
    if start <= value <= end:
        return 0.0
    return min(abs(value - start), abs(value - end))


def _source_range_covered(dialogue_mappings: list[DialogueSegment], start: float, end: float) -> bool:
    cursor = start
    for item in sorted(dialogue_mappings, key=lambda value: (value.source_start_seconds, value.source_end_seconds)):
        if item.source_end_seconds < cursor - 0.001:
            continue
        if item.source_start_seconds > cursor + 0.001:
            break
        cursor = max(cursor, item.source_end_seconds)
        if cursor >= end - 0.001:
            return True
    return False


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


def _continuity_decision_from_context(
    context: dict[str, Any], candidate_id: str, boundary_decision: BoundaryDecision | None,
) -> ContinuityDecision | None:
    """Read A-2 provenance or deterministically derive it from cached evidence."""

    raw = context.get("continuity_decision")
    derived: ContinuityDecision | None = None
    if boundary_decision is not None:
        try:
            derived = build_continuity_decision(
                candidate_id=candidate_id,
                boundary_decision=boundary_decision,
                primary_evidence=[
                    item for item in context.get("primary_evidence", []) if isinstance(item, dict)
                ],
                multimodal_context=(
                    context.get("multimodal_context")
                    if isinstance(context.get("multimodal_context"), dict) else {}
                ),
            )
        except Exception as error:
            raise ProductionPlanError(f"CONTINUITY_DECISION_INVALID: {error}") from error
    if raw not in (None, {}):
        if not isinstance(raw, dict):
            raise ProductionPlanError("CONTINUITY_DECISION_INVALID: SourceContext continuity_decision must be an object.")
        try:
            decision = ContinuityDecision.model_validate(raw)
        except Exception as error:
            raise ProductionPlanError(f"CONTINUITY_DECISION_INVALID: {error}") from error
        if derived is None or decision.model_dump(mode="json") != derived.model_dump(mode="json"):
            raise ProductionPlanError(
                "CONTINUITY_DECISION_EVIDENCE_MISMATCH: persisted decision does not match cached source evidence."
            )
    elif derived is not None:
        decision = derived
    else:
        return None
    if decision is None:
        return None
    if decision.candidate_id != candidate_id:
        raise ProductionPlanError(
            "CONTINUITY_CANDIDATE_MISMATCH: SourceContext continuity decision belongs to another candidate."
        )
    if boundary_decision is None:
        raise ProductionPlanError("CONTINUITY_BOUNDARY_MISSING: continuity decision requires BoundaryDecision.")
    if decision.boundary_decision_id != boundary_decision.decision_id:
        raise ProductionPlanError("CONTINUITY_BOUNDARY_REFERENCE_MISMATCH")
    if decision.boundary_decision_sha256 != stable_text_hash(boundary_decision.model_dump_json()):
        raise ProductionPlanError("CONTINUITY_BOUNDARY_FINGERPRINT_MISMATCH")
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
