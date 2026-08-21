from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.audio_service import AudioCompositionService
from app.creative_evidence import build_native_evidence_handoff
from app.creative_execution import compile_native_creative_plan, default_native_creative_intent
from app.creative_policy import CREATIVE_POLICY_VERSION
from app.creative_lifecycle import (
    CreativeArtifactError,
    build_creative_execution,
    build_creative_handoff,
    creative_policy_changed,
    load_candidate_creative_identity,
    persist_candidate_creative_identity,
    revise_creative_intent,
)
from app.pipeline import Pipeline, StageTracker
from app.sources import Source
from app.tts_providers import MockTTSProvider
from app.tts_service import TTSService
from app.utils import read_json, write_json
from tests.test_audio_composition import _audio_config
from tests.test_native_evidence_handoff import (
    _native_plan_for_media,
    _phase6_artifacts,
)
from tests.test_video_composition import _source_video
from app.creative_contracts import EditMapSegment, OutputInterval, SourceInterval, SourceOutputTimeMap


def _parent_candidate(tmp_path: Path):
    config = _audio_config()
    config.production_render.enabled = True
    config.production_render.output_width = 180
    config.production_render.output_height = 320
    config.production_render.output_fps = 30
    config.production_render.video_bitrate = "500k"
    config.production_render.encoder = "cpu"
    config.production_render.same_source_broll_allowed = True
    config.validate()
    source = Source("source-audio", _source_video(tmp_path / "source.mp4"), "source.mp4", "test")
    transcript = {
        "source_id": source.id,
        "segments": [{"id": 0, "start": 1.0, "end": 2.0, "text": "Source dialogue."}],
        "words": [
            {"start": 1.0, "end": 1.45, "text": "Source"},
            {"start": 1.45, "end": 2.0, "text": "dialogue."},
        ],
    }
    plan = _native_plan_for_media(config, source, transcript)
    upstream = tmp_path / "parent-run"
    candidate_root = upstream / "selected-candidate"
    tts = TTSService(tmp_path, config).generate(
        plan, tmp_path / "work", candidate_root, provider=MockTTSProvider(),
    )
    audio = AudioCompositionService(tmp_path, config).compose(
        plan, source, transcript, tts.model_dump(mode="json"), tmp_path / "work", candidate_root,
    )
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="candidate-map",
        source=SourceInterval.from_seconds(1.0, 2.0),
        output=OutputInterval.from_seconds(0.0, 1.0),
    ),))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)
    evidence = build_native_evidence_handoff(
        plan, mapping, config, candidate=candidate,
        multimodal_timeline=timeline, story_units=stories,
    )
    compiled = compile_native_creative_plan(
        evidence.intent, transcript, config, source_width=320, source_height=180,
        target_observations=evidence.target_observations, source_scenes=evidence.source_scenes,
    )
    handoff = build_creative_handoff(
        evidence.intent,
        target_observations=evidence.target_observations,
        source_scenes=evidence.source_scenes,
    )
    execution = build_creative_execution(
        evidence.intent, compiled, execution_status=evidence.execution_status,
        reason_codes=evidence.reason_codes, diagnostics=evidence.diagnostics,
    )
    persist_candidate_creative_identity(
        candidate_root / "production-render", intent=evidence.intent,
        compiled_plan=compiled, handoff=handoff, execution=execution,
    )
    ghost_id = "candidate-first-must-not-leak"
    write_json(upstream / "report.json", {
        "production_plan": {"items": [
            {"candidate_id": ghost_id, "requested_index": 1, "plan": plan.model_dump(mode="json")},
            {"candidate_id": plan.metadata.candidate_id, "requested_index": 2, "plan": plan.model_dump(mode="json")},
        ]},
        "production_render": {"items": [
            {"candidate_id": ghost_id, "output_directory": str(upstream / "ghost-output")},
            {"candidate_id": plan.metadata.candidate_id, "output_directory": str(candidate_root)},
        ]},
    })
    write_json(tmp_path / "work" / "transcript.json", transcript)
    return config, source, transcript, plan, audio, upstream, candidate_root, evidence, compiled


def test_candidate_creative_identity_is_stable_revisionable_and_corruption_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_family, policy_revision = CREATIVE_POLICY_VERSION.rsplit(".", 1)
    next_policy_version = f"{policy_family}.{int(policy_revision) + 1}"
    current_policy_marker = f"creative_policy:{CREATIVE_POLICY_VERSION}"
    next_policy_marker = f"creative_policy:{next_policy_version}"
    config, _source, transcript, plan, _audio, _upstream, root, evidence, compiled = _parent_candidate(tmp_path)

    loaded_intent, loaded_compiled, loaded_handoff, loaded_execution = load_candidate_creative_identity(
        root / "production-render", plan,
    )
    assert loaded_intent.canonical_hash() == evidence.intent.canonical_hash()
    assert loaded_compiled.plan_hash == compiled.plan_hash
    assert loaded_handoff.candidate_id == loaded_execution.candidate_id == plan.metadata.candidate_id

    assert not creative_policy_changed(loaded_intent, config)
    assert current_policy_marker in loaded_intent.provenance
    parent_hash = loaded_intent.canonical_hash()
    monkeypatch.setattr("app.creative_lifecycle.CREATIVE_POLICY_VERSION", next_policy_version)
    unchanged = revise_creative_intent(loaded_intent, config)
    assert unchanged is loaded_intent
    assert unchanged.canonical_hash() == parent_hash
    assert current_policy_marker in unchanged.provenance
    assert next_policy_marker not in unchanged.provenance

    legacy_intent = loaded_intent.model_copy(update={
        "provenance": tuple(
            item for item in loaded_intent.provenance
            if not item.startswith(("creative_policy:", "preset_"))
        ),
    })
    config.product_flow.preset_selection_mode = "explicit"
    config.product_flow.preset_provenance = "legacy_pinned"
    config.product_flow.subtitle_preset = legacy_intent.policy.preset_id
    assert revise_creative_intent(legacy_intent, config) is legacy_intent

    # A new recommendation must not rewrite an already approved auto draft.
    config.product_flow.preset_selection_mode = "auto"
    config.product_flow.preset_provenance = "content_recommendation"
    config.product_flow.subtitle_preset = "dynamic"
    config.product_flow.recommended_subtitle_preset = "dynamic"
    pinned = revise_creative_intent(loaded_intent, config)
    assert pinned is loaded_intent
    assert pinned.canonical_hash() == loaded_intent.canonical_hash()

    # A manual selection is a real revision, even after an auto-selected draft.
    config.product_flow.preset_selection_mode = "explicit"
    config.product_flow.preset_provenance = "explicit_selection"
    config.product_flow.configured_subtitle_preset = "dynamic"
    revised = revise_creative_intent(loaded_intent, config)
    assert revised.revision == loaded_intent.revision + 1
    assert revised.evidence_fingerprint == loaded_intent.evidence_fingerprint
    assert revised.source_output_mapping.fingerprint == loaded_intent.source_output_mapping.fingerprint
    assert revised.source_broll == loaded_intent.source_broll
    assert next_policy_marker in revised.provenance
    assert current_policy_marker not in revised.provenance
    assert revised.canonical_hash() != parent_hash
    revised_compiled = compile_native_creative_plan(
        revised, transcript, config, source_width=320, source_height=180,
        target_observations=evidence.target_observations,
        source_scenes=evidence.source_scenes,
    )
    assert revised_compiled.intent_hash != loaded_compiled.intent_hash
    assert {
        item.node_id: item.cache_key for item in revised_compiled.render_graph_nodes
    } != {
        item.node_id: item.cache_key for item in loaded_compiled.render_graph_nodes
    }

    # Policy version is persisted in creative provenance and feeds both the
    # intent hash and every downstream cache key for new drafts.
    config.product_flow.preset_selection_mode = "auto"
    config.product_flow.preset_provenance = "content_recommendation"
    baseline_intent = default_native_creative_intent(
        plan, evidence.intent.source_output_mapping, config,
    )
    baseline_compiled = compile_native_creative_plan(
        baseline_intent, transcript, config, source_width=320, source_height=180,
    )
    monkeypatch.setattr("app.creative_execution.CREATIVE_POLICY_VERSION", next_policy_version)
    updated_intent = default_native_creative_intent(
        plan, evidence.intent.source_output_mapping, config,
    )
    updated_compiled = compile_native_creative_plan(
        updated_intent, transcript, config, source_width=320, source_height=180,
    )
    assert current_policy_marker in baseline_intent.provenance
    assert next_policy_marker in updated_intent.provenance
    assert updated_intent.canonical_hash() != baseline_intent.canonical_hash()
    assert updated_compiled.intent_hash != baseline_compiled.intent_hash
    assert {
        item.node_id: item.cache_key for item in updated_compiled.render_graph_nodes
    } != {
        item.node_id: item.cache_key for item in baseline_compiled.render_graph_nodes
    }

    compiled_path = root / "production-render" / "compiled-render-plan.json"
    corrupted = json.loads(compiled_path.read_text(encoding="utf-8"))
    corrupted["plan_hash"] = "0" * 64
    write_json(compiled_path, corrupted)
    with pytest.raises(CreativeArtifactError, match="CREATIVE_PARENT_ARTIFACT_INVALID"):
        load_candidate_creative_identity(root / "production-render", plan)


def test_multi_output_rerender_loads_only_selected_candidate_and_blocks_corrupt_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source, _transcript, plan, _audio, upstream, root, evidence, compiled = _parent_candidate(tmp_path)
    selected_id = plan.metadata.candidate_id
    pipeline = Pipeline(
        tmp_path, config, run_id="rerender-selected", production_render_only=True,
        upstream_run_directory=upstream, selected_candidate_ids=[selected_id],
    )
    calls: list[dict] = []

    def fake_compose(_tracker, loaded_plan, _audio_project, _source, _transcript, _work, output, **kwargs):
        calls.append({"candidate_id": loaded_plan.metadata.candidate_id, **kwargs})
        artifact = output / "production-render" / "final-short.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"candidate-owned-render")
        return {
            "enabled": True, "status": "completed", "output_file": str(artifact),
            "compatibility_mode": "native", "execution_status": "native_rich",
        }

    monkeypatch.setattr(pipeline, "_compose_production_render", fake_compose)
    monkeypatch.setattr(
        pipeline, "_persist_quality_reports",
        lambda **kwargs: (kwargs["registry"], []),
    )
    output = tmp_path / "rerender-output"
    result = pipeline._run_candidate_production_rerender(
        StageTracker(tmp_path / "rerender-state.json"), source, tmp_path / "work", output,
    )

    assert [item["candidate_id"] for item in calls] == [selected_id]
    assert calls[0]["compiled_plan"].plan_hash == compiled.plan_hash
    assert calls[0]["creative_intent"].canonical_hash() == evidence.intent.canonical_hash()
    assert result.output_files == [output / "results" / "final-short-02.mp4"]
    assert read_json(result.report_path, {})["run"]["selected_candidate_ids"] == [selected_id]

    compiled_path = root / "production-render" / "compiled-render-plan.json"
    raw = read_json(compiled_path, {})
    raw["intent_hash"] = "0" * 64
    write_json(compiled_path, raw)
    blocked = Pipeline(
        tmp_path, config, run_id="rerender-corrupt", production_render_only=True,
        upstream_run_directory=upstream, selected_candidate_ids=[selected_id],
    )
    blocked_calls: list[object] = []
    monkeypatch.setattr(blocked, "_compose_production_render", lambda *args, **kwargs: blocked_calls.append(args))
    monkeypatch.setattr(
        blocked, "_persist_quality_reports",
        lambda **kwargs: (kwargs["registry"], []),
    )
    blocked_result = blocked._run_candidate_production_rerender(
        StageTracker(tmp_path / "blocked-state.json"), source, tmp_path / "work", tmp_path / "blocked-output",
    )
    blocked_report = read_json(blocked_result.report_path, {})
    assert not blocked_calls
    assert blocked_result.terminal_status == "failed"
    assert "CREATIVE_PARENT_ARTIFACT_INVALID" in blocked_report["production_render"]["items"][0]["errors"][0]
