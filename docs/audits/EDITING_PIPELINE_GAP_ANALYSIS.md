# Editing Pipeline Gap Analysis

**Date:** 2026-07-31
**Scope:** read-only audit of the existing Content Factory editing pipeline and the persisted artifacts already in the workspace. No analysis or render was started for this audit.

## 1. Executive Summary

Content Factory already has a substantial, reusable editing pipeline. In particular, Goal 5A is not a blank slate: it introduced source-content understanding, `StoryUnit` candidates, word/sentence-safe semantic boundaries, coverage-aware selection, analysis/draft hand-offs, and a run-scoped final-result manifest. The renderer side also already has typed `ProductionPlan`, `AudioProject`, `VideoProject`, `CompositionSegment`, subtitle-layout, render-validation, recovery, and canonical-result contracts.

The target system must therefore evolve these contracts; it must not create a second candidate store, a second render pipeline, a new path-derived output resolver, or a parallel "edit-plan" renderer.

The largest gaps are not basic media generation. They are quality ownership and the missing contract that joins existing decisions:

- Candidate selection has useful inputs, but no persisted `EligibilityDecision`, no versioned V2 score/penalty evidence, and no hook/payoff or context-debt decision contract.
- Coverage-aware selection is a deterministic diversity heuristic, not the requested evidence-bearing MMR policy.
- `ProductionPlan` is the seed of the deterministic edit plan, but it is a per-candidate source/dialogue plan rather than the V2 renderer-only envelope. It lacks project/run/analysis identity, a boundary-decision reference, target/preset/platform data, and a validated edit-plan schema.
- `output_quality.py` is useful pre-render validation, but it is not a final Quality Gate. `Pipeline.build_terminal_state()` declares completion from output count and render failures; the desktop facade then checks the canonical MP4/manifest. Neither derives `PASS`, `PASS_WITH_WARNINGS`, or `BLOCKED` from a persisted `QualityReport`.
- The latest completed production run has a concrete correctness defect: every approved `ProductionPlan` repeats one exact source range as two sequential dialogue clips. This is a real source of repeated phrases, not merely a ranking hypothesis.
- Visual analysis failed schema validation in that run. All three rendered videos fell back to zero-confidence `center_crop` with no subject/active-speaker tracking. The stored contracts cannot prove that the face-crop or empty-frame defects are absent.

**Recommended first implementation Goal:** Goal 5B, narrowly scoped as **persistent Candidate Eligibility + Candidate Score V2**, reusing `Candidate`, `ScoredCandidate`, `SemanticBoundaryEngine`, and `select_with_coverage`. Include one non-negotiable existing-plan integrity check: reject a candidate plan that schedules the same exact source range twice until the model can represent a merged evidence mapping. This is the smallest safe change that blocks the observed repeat defect without inventing an EditPlan V2 or changing rendering/composition behavior.

## 2. Current Pipeline Map

```text
source / Source
  -> metadata + ffprobe, transcript + word timings
  -> audio features, scene boundaries, optional visual analysis
  -> VideoContentProfile + GlobalContentMap + StoryUnit
  -> SemanticBoundaryEngine -> Candidate.boundary_diagnostics
  -> local scoring + optional AI rank -> ScoredCandidate
  -> select_with_coverage -> selected candidate IDs
  -> analysis.json -> draft.json / approved draft
  -> Content Transformation + FinalScript validation
  -> ProductionPlan (per candidate)
  -> AudioProject / mixed_audio.wav
  -> VideoProject / ReframePlan / SubtitleProject / render-result.json
  -> canonical ClipResult + manifest.json + report.json
  -> desktop completion and project/run snapshots
```

| Concern | Existing implementation and persisted output | Audit finding |
|---|---|---|
| Transcript and audio analysis | `app/transcription.py`, `app/transcript_features.py`, `app/audio_features.py`; `transcript.json`, `transcript_features.json`, `audio_features.json` | Reusable. Word timings feed semantic boundaries and dialogue subtitles. |
| Scene and visual analysis | `detect_scene_boundaries()` in `app/scene_detection.py`; `analyse_video_subjects()` in `app/visual_analysis.py`; `scene_boundaries.json`, `visual_analysis.json` | Scene boundaries work as deterministic evidence. Optional subject analysis has a safe fallback, but the most recent completed real run fell into it because its strict OpenAI schema was invalid. |
| Candidate generation and selection | `generate_semantic_candidates()`, `SemanticBoundaryEngine`, `select_with_coverage()` in `app/content_understanding.py`; `score_candidates()` in `app/local_scoring.py`; `merge_ai_ranking()` in `app/intelligence.py` | Goal 5A already supplies candidate identity, rough/refined-safe range, basic context signal, and coverage selection. It is not yet V2 eligibility/scoring/MMR. |
| Draft and approval lifecycle | `AnalysisArtifact`, `DraftArtifact`, `Pipeline._run_draft_preview()`, `Pipeline._run_production_from_draft()` | Reusable and important: analysis, draft preview, approval, and production render are intentionally separate. |
| Production edit/render | `ProductionPlan`, `AudioProject`, `VideoProject`; `build_production_plan()`, `AudioCompositionService.compose()`, `VideoCompositionService.compose()` | A typed downstream pipeline exists. The production plan is the correct contract to evolve into the V2 edit-plan envelope. |
| Composition | `CompositionSegment`, `ReframePlan`, `build_composition_segments()`, `_validate_tracking_decisions()` in `app/video_composition.py` | Scene-aware composition, targets, tracking decisions, smoothing, and fallback paths already exist. The real run had no usable subject observations, so it used only centered crops. |
| Subtitles | `build_subtitle_project()`, `_semantic_groups()`, `_fit_tokens()` in `app/production_subtitles.py`; `SubtitleProject` | Two-line fitting and a safe fallback are implemented; language/CPS/post-render geometry quality requirements are incomplete. |
| Media and artifact validation | `validate_final_video()`, `probe_media()`; `ClipResult`, `write_run_manifest()`, `PipelineFacade.completion()` | H.264/AAC/size/resolution/FPS/AV-duration checks plus run-scoped canonical-result checks are present. They are not a QualityReport. |
| Fallback and recovery | `StageTracker`, `run_artifacts.py`, `run_manifest.py`, desktop recovery services | Explicit cache, run metadata, draft recovery, CPU encoder fallback, and canonical run results are already in place and must be retained. |

## 3. Existing Contracts

| Contract | Current owner and shape | What is reusable | Gap relevant to V2 |
|---|---|---|---|
| Project / run / source identity | `DesktopProject`, `ProjectRun`, `Source`, `Pipeline.run_id`; `work/run-metadata/<hash>.json` | Desktop project/run IDs and source IDs are stable enough to extend. Run metadata resolves actual paths without deriving them from a filename. | IDs are not carried all the way through `ProductionPlan` or `VideoProject` as the V2 graph requires. |
| Candidate identity | `Candidate.id`, `ScoredCandidate`, `story_unit_id`, `chapter_id`, `content_signature`, `boundary_diagnostics` in `candidates.scored.json` | Existing IDs are used by analysis, draft, plan, render, and manifest. | No separate eligibility/score decision ID/version/evidence object. |
| Selected moments | `final_selection.json`, `coverage_map.json`, `AnalysisArtifact.recommendation`, `DraftArtifact.candidates` | Selection IDs, selection reasons, coverage, and review previews exist. | Selection is not a first-class `DiversityDecision` with config version, all exclusions, and reproducible similarity inputs. |
| Boundary decision | `SemanticBoundaryEngine.resolve()` writes `Candidate.boundary_diagnostics` (`schema_version` `5A.1`) | Rough/resolved ranges, word integrity, sentence integrity, padding, confidence, payoff and fallback evidence already persist. | It is embedded in a candidate and not referenced by a downstream plan; the final dialogue ranges may be narrower/different. |
| Production plan | `ProductionPlan` / `ProductionMetadata` in `app/production_models.py`; `production-plan-<candidate>.json` in draft runs | Ordered dialogue/narration/pause segments, source ranges, timeline, audio layers, and plan ID are a strong base. | `plan_version` is 3A.1, not an EditPlan schema; it has no project/run/analysis ID, target/preset/platform, input fingerprints, boundary ref, complete composition/subtitle/audio plans, or renderer-only validation. |
| Targets and composition | `SubjectBounds`, `CompositionSegment`, `ReframePlan`, `VideoProject`; `reframe-plan.json`, `video-timeline.json` | Target, crop, tracking mode/reasons, scene interval, quality diagnostics, and fallbacks are persisted. | Metrics required by V2 are partial and mostly meaningful only when subject observations exist; no final frame-level face/empty-frame gate exists. |
| Subtitle metadata | `SubtitleProject`, `SubtitleCue`; `subtitle-project.json`, `production-subtitles.ass` | Cue timing, fitted lines/font size, split reason, fallback marker and ASS output are persisted. | No explicit language profile/version, CPS measurements, rendered bounding boxes, mask-overlap checks, or post-render subtitle QualityReport checks. |
| Audio metadata | `AudioProject`, `AudioMix`, `AudioValidation`; `audio-project.json`, `audio-manifest.json` | Source range, timeline, checksum, mix duration, ducking and configured narration loudness are typed and cacheable. | No measured final LUFS/true peak/noise/intelligibility/licensing quality result; exact duplicate dialogue ranges are currently accepted. |
| Render metadata | `RenderMetadata`, `RenderResult`, `RenderValidation`, `VideoProject`; `render-result.json`, `video-project.json` | Render config/cache keys, source/mix checksums, actual duration, encoder/fallback, technical validation and artifact checksums exist. | No edit-plan identity/fingerprint check against a QualityReport, and no `artifact_id`. |
| Report / state / draft progress | `make_report()`, `StageTracker`, `DraftArtifact`; `report.json`, `state.json`, `draft-progress.json` | Atomic persistence, stage cache facts, review state and run-scoped reports work. | `report.json` is an aggregate operational report; it is not the normalized V2 QualityReport nor a source of readiness truth. |
| Final artifact path | `ClipResult`, `write_run_manifest()`, `Pipeline._publish_run_result()`; `runs/<run_id>/results/final-short-*.mp4` | Canonical results are copied into the current run and manifest values are ID-checked and path-scoped by the desktop facade. | The manifest has no explicit artifact/quality-report IDs or final-file checksum field; the content fingerprint is not the full V2 parent-identity graph. |

## 4. Mapping to Editing Pipeline V2

The V2 order can be adopted as an evolution of the current path:

```text
existing analysis + StoryUnit + boundary diagnostics
  -> extend to EligibilityDecision and CandidateScore V2
  -> extend select_with_coverage to a recorded DiversityDecision/MMR policy
  -> retain SemanticBoundaryEngine as BoundaryDecision producer
  -> evolve ProductionPlan into the versioned edit-plan envelope
  -> retain AudioProject, ReframePlan, SubtitleProject and VideoProject as plan sections/artifacts
  -> retain current renderer and canonical manifest
  -> add a persisted QualityReport that alone maps checks to PASS/PASS_WITH_WARNINGS/BLOCKED
```

The proposed `edit_plan.v1` example must be treated as an illustrative target, not a parallel implementation. There is no `EditPlan`, `edit_plan_id`, `EligibilityDecision`, `CandidateScore`, `DiversityDecision`, `BoundaryDecision`, or `QualityReport` implementation in `app/` today. The correct migration is a backward-compatible versioned extension of `ProductionPlan` and its adjacent typed artifacts, with adapters for draft-era 3A.1 plans.

## 5. Real Run Findings

### Run chosen

The most recent workspace activity (`Только_не_со_мной…`) has analysis/draft artifacts only and no completed final MP4 triplet. The latest completed production run is therefore the evidence run below:

```text
source: Большинство_бизнесов_провалит_ИИ-трансформацию-c0dcb7e71e9f
analysis run: 8169d9024d43407ca372e19c43912a95
draft run:    fcb960a9c73249068875a707c48f6857
final run:    66ed9e7014c54fb9a4325db7bc5b578f
project_id:   816cbdfd882c4091a1f91f6023a2a162
analysis_id:  analysis-ca453ec35a8d9ccd
```

The final manifest records three primary, distinct candidate/plan pairs. Each canonical `results/final-short-0N.mp4` has the same SHA-256 as the candidate renderer's intermediate `production-render/final-short.mp4`; the draft and final share the same candidate and `production_plan_id`. This validates the current draft-to-final and intermediate-to-canonical linkage.

| Candidate | Candidate safe range -> plan source range | Final MP4 / ffprobe | Current quality/composition evidence |
|---|---|---|---|
| `candidate-chapter-062-story-002` | 5345.60–5371.03 -> 5345.85–5370.43 | `final-short-01.mp4`, 22.933 s; H.264/AAC, 1080x1920, yuv420p, 30 fps | Render `warning`; technical validation `valid`; 16 subtitle cues / 6 fitted-fallback cues; 16 `center_crop`, no tracking, subject confidence 0. |
| `candidate-chapter-089-story-002` | 7409.60–7428.27 -> 7409.85–7427.71 | `final-short-02.mp4`, 20.140 s; H.264/AAC, 1080x1920, yuv420p, 30 fps | Render `warning`; technical validation `valid`; 15 subtitle cues / 7 fitted-fallback cues; 12 `center_crop`, no tracking, subject confidence 0. |
| `candidate-chapter-011-story-002` | 908.88–966.77 -> 909.13–939.93 | `final-short-03.mp4`, 30.220 s; H.264/AAC, 1080x1920, yuv420p, 30 fps | Render `warning`; technical validation `valid`; 21 subtitle cues / 10 fitted-fallback cues; 22 `center_crop`, no tracking, subject confidence 0. |

The analysis artifact selected three different chapters. The existing heuristic recorded selected-to-selected semantic similarities of 0.061 and 0.097 where applicable, with no duplicate cluster. That is evidence that the current lexical/content-signature heuristic did not flag cross-output duplicates; it is not proof of semantic diversity under the requested embedding/MMR standard.

### Confirmed quality defects in persisted facts

1. **Repeated material within every final output.** The source-range duplication exists in both the draft `ProductionPlan` and final `AudioProject`, sequentially in the audio timeline:

   - candidate 062: 5347.650–5348.910 appears as `dialogue-002` at 1.760–3.020 and `dialogue-003` at 3.020–4.280;
   - candidate 089: 7424.870–7427.710 appears as `dialogue-010` at 14.460–17.300 and `dialogue-011` at 17.300–20.140;
   - candidate 011: 934.490–936.150 appears as `dialogue-014` at 23.240–24.900 and `dialogue-015` at 24.900–26.560.

   The cause is not a duplicate `candidate_id`: it is multiple facts resolving to the same source transcript range, while `build_production_plan()` deduplicates by `fact_id` only. The audio service then composes every `DialogueSegment` in order.

2. **Visual target evidence is unavailable.** `visual_analysis.json` says `status: fallback`; the provider rejected the strict response schema because `normalized_width` was optional in `properties` but absent from `required`. All selected analysis records explicitly warn that visual observations are unavailable. Their final `ReframePlan` records only zero-confidence centered crops. This makes the known cropped-face/empty-table risks unverified, not resolved.

3. **Subtitle fallback is systematic.** Each final render has a technically valid subtitle project, but 6/16, 7/15, and 10/21 cues respectively use `fallback_fitted`. `output_quality.py` reports warnings, not errors. The artifacts do not persist CPS, actual rendered coordinates, platform-mask overlap, or semantic-line-break quality metrics, so the requested Subtitle Quality V2 cannot certify them.

4. **Boundary evidence is candidate-level only.** All three selected candidates have the same healthy 5A boundary indicators: overall score 0.957, semantic completion 0.95, context independence 0.75, payoff preserved, and continuation risk 0.15. But the final plan's source range is derived from the fact/dialogue mapping, not a persisted reference to that boundary decision. The plan can therefore narrow or reorder material without a downstream boundary-quality assertion.

### Where "ready" is actually decided

The current decision is operational, not the V2 quality decision:

1. `VideoCompositionService._render()` rejects a pre-render `validate_output_quality()` failure and rejects an invalid `validate_final_video()` ffprobe result.
2. `Pipeline.build_terminal_state()` then returns `completed` when at least one required delivery exists and `candidate_flow.failed == 0`; it does not aggregate semantic, duplicate, composition, subtitle, audio, or identity checks into a quality status.
3. `PipelineFacade.completion()` verifies the current run manifest, run-scoped paths, `ClipResult` IDs/revisions, and final MP4 technical validity. `DesktopServices._finish_completion()` marks the run `completed_with_warnings` merely when its warning list is non-empty.

For this run, the manifest/terminal status is `completed`, while every primary `ClipResult` and its nested render report is `warning`. It can be delivered as a technically valid current-run artifact despite the repeated source material and unverified visual target quality. That directly conflicts with the V2 requirement that final readiness be owned by `QualityGate`.

## 6. Reuse / Extend / Missing Matrix

The status below is intentionally one of the requested values for every V2 component. Target files are for future Goals only; this audit changes none of them.

| Component and status | Existing files, symbols, current contract, and working behavior | Exact gap / extension / do not duplicate | Migration risks, regression tests, likely target files |
|---|---|---|---|
| **Candidate Eligibility Gate** — `EXISTS_BUT_NEEDS_EXTENSION` | `app/content_understanding.py`: `SemanticBoundaryEngine.resolve()`, `generate_semantic_candidates()`, `select_with_coverage()`; `Candidate.boundary_diagnostics`; `tests/test_semantic_boundaries.py`. It rejects unsafe sentence/word boundaries and selection enforces duration, score and boundary eligibility. | Add an evidence-bearing eligibility decision with reason codes, recoverability, visual/audio viability, context debt and hook/payoff inputs. Reuse `Candidate` and its scored artifact; do **not** add a second candidates JSON or a parallel gate after selection. | Old candidate cache records lack the decision; default them to legacy/unassessed rather than silently eligible. Test safe/unsafe boundary, invalid source range, context and no-payoff reasons. Likely: `app/models.py`, `app/content_understanding.py`, `app/selection.py`, `app/config.py`, `tests/test_semantic_boundaries.py`, `tests/test_coverage_selection.py`. |
| **Candidate Scoring V2** — `EXISTS_BUT_NEEDS_EXTENSION` | `app/local_scoring.py: score_candidates()`; `app/intelligence.py: merge_ai_ranking()`; `ScoredCandidate` and `candidates.scored.json`. Local score already includes hook, completeness, clarity, density, pacing, audio, scene, context and boundary signals. | Replace implicit weights/flat fields with a versioned component score, penalties and evidence. Current 55/45 AI merge has no V2 score provenance and a high total can coexist with an unrecorded critical defect. Do **not** create a competing ranker beside `local_scoring.py`. | Score changes invalidate selection/diversity only, not transcript/render caches. Test exact weight/version serialization, penalties and deterministic fallback. Likely: `app/local_scoring.py`, `app/intelligence.py`, `app/models.py`, `app/config.py`, `tests/test_candidates.py`, `tests/test_ai_selection.py`, `tests/test_coverage_selection.py`. |
| **Context debt** — `EXISTS_BUT_NEEDS_EXTENSION` | `_context_dependency()` / `candidate_transcript_features()` and `StoryUnit.context_dependency_score`; current review payload exposes a context-independence number. | Turn a scalar heuristic into explicit unresolved reference/entity/question/setup evidence with thresholds and reason codes. Do not recalculate it independently in the UI or Quality Gate. | Legacy scores need an `unknown`/legacy evidence state. Test pronouns, question-only answers, undefined entities and high-debt rejection. Likely: `app/transcript_features.py`, `app/content_understanding.py`, `app/models.py`, `tests/test_content_understanding.py`. |
| **Hook/payoff evaluation** — `EXISTS_BUT_NEEDS_EXTENSION` | `_hook_score()`; `StoryUnit` core idea/setup/payoff; candidate review `hook_summary` and `payoff_summary`; semantic transformation facts. | Current fields are summaries/heuristics, not a scored, grounded hook/payoff decision or false-hook blocker. Reuse StoryUnit and FinalScript evidence; do not add an ungrounded LLM-only timestamp path. | Preserve local fallback when no AI is available. Test weak opening, missing payoff, payoff outside chosen boundary and false hook. Likely: `app/transcript_features.py`, `app/content_understanding.py`, `app/semantic_extraction.py`, `tests/test_content_understanding.py`. |
| **Semantic Diversity / MMR reranking** — `EXISTS_BUT_NEEDS_EXTENSION` | `select_with_coverage()`, `_coverage_duplicate()`, `_signature_similarity()`; `app/diversity.py`; `coverage_map.json`. Existing greedy selection records coverage and blocks temporal/lexical/signature duplicates. | Add recorded pairwise similarity components and MMR selection (`lambda`, exclusions, config version). Keep coverage as a selection feature; do not replace it with `sort_by_score` or a second selected-moments store. | Similarity-version changes must not invalidate analysis. Test tie determinism, near semantic duplicates, topic diversity and fewer-than-requested outputs. Likely: `app/content_understanding.py`, `app/diversity.py`, `app/config.py`, `tests/test_coverage_selection.py`, `tests/test_candidates.py`. |
| **Boundary Refinement** — `EXISTS_BUT_NEEDS_EXTENSION` | `SemanticBoundaryEngine`, `BoundaryPoint`, `SemanticBoundaryResolution`, `Candidate.boundary_diagnostics`; `app/candidate_review.py` boundary override validation. | Promote the diagnostics to a referenced BoundaryDecision and ensure every downstream source segment is a validated transformation of it. Add speaker/scene evidence and explicit post-edit no-cut assertions. Do not replace the existing semantic-boundary engine with a duration-window cutter. | Existing 5A artifacts must remain readable. Test candidate-to-plan narrowing, pauses, terminal transcript fallback and no word split. Likely: `app/content_understanding.py`, `app/production_plan.py`, `app/production_models.py`, `tests/test_semantic_boundaries.py`, `tests/test_production_plan.py`. |
| **Deterministic Edit Plan** — `EXISTS_BUT_NEEDS_EXTENSION` | `ProductionPlan`/`ProductionMetadata`, `AudioProject`, `VideoProject`, `ReframePlan`, `SubtitleProject`; `build_production_plan()` and `VideoCompositionService.compose()`. | Evolve `ProductionPlan`; do not introduce an independent `EditPlan` renderer. Add explicit schema/version, project/run/analysis/candidate identities, boundary reference, target/preset/platform, input fingerprints, warning list and plan validation. Renderer currently consumes plan **plus** audio, source, transcript, paths and config; V2 must validate the envelope before rendering. | Pydantic `extra=forbid` and cached 3A.1 drafts require defaulted migration adapters. The observed duplicated dialogue ranges need a strict plan invariant before rendering. Test plan determinism, duplicate source ranges, identity mismatch and rerender cache keys. Likely: `app/production_models.py`, `app/production_plan.py`, `app/audio_models.py`, `app/video_models.py`, `app/pipeline.py`, `tests/test_production_plan.py`, `tests/test_content_understanding_cache.py`. |
| **Composition Quality checks** — `EXISTS_BUT_NEEDS_EXTENSION` | `CompositionSegment`, `ReframePlan`, `_validate_tracking_decisions()`, `_tracking_quality_metrics()`, `_apply_composition_quality_diagnostics()`; `validate_output_quality()`. | Keep scene-aware composition and its safe fallback. Add required V2 metrics, no-observation policy, active-speaker/semantic-target evidence, frame-level face/empty-frame checks and a blocking route. Do not rebuild composition or use largest-face-only logic. | Visual-provider schema/config changes must invalidate visual/composition only. Test provider fallback, face-at-edge, empty-table, crop movement and safe fallback. Likely: `app/visual_analysis.py`, `app/video_composition.py`, `app/video_models.py`, `app/output_quality.py`, `tests/test_visual_analysis.py`, `tests/test_video_composition.py`. |
| **Subtitle Quality V2** — `EXISTS_BUT_NEEDS_EXTENSION` | `build_subtitle_project()`, `_semantic_groups()`, `_fit_tokens()`; `SubtitleProject` layout state; `validate_output_quality()`; fitting tests. | Preserve fitting/ASS generation. Add versioned RU/EN profiles, CPS measurement, syntax violations, rendered-box and platform-mask overlap checks, timing confidence and a post-render result. Do not make an alternate subtitle writer or bypass `SubtitleProject`. | Existing fallback cue warning semantics must map to the new severity policy without breaking old display. Test Cyrillic long words, two lines, CPS, weak line break, overflow and face/UI overlap. Likely: `app/production_subtitles.py`, `app/video_models.py`, `app/output_quality.py`, `app/config.py`, `tests/test_subtitle_fitting.py`, `tests/test_video_composition.py`. |
| **Audio Plan** — `EXISTS_BUT_NEEDS_EXTENSION` | `AudioProject`, `AudioTimeline`, `AudioMix`, `AudioValidation`; `AudioCompositionService.compose()` and `audio_report_section()`. Source audio, mix duration, checksums, ducking and configured narration loudness are already persisted. | Extend this contract with a logical V2 audio plan plus measured loudness/peak/noise/intelligibility, licensing and duplicate-source-range validation. Do not create a separate FFmpeg audio path. | Preserve source-audio modes and current audio cache keys. Test duplicate dialogue range, silent/clip/desync, loudness policy and licensed/unlicensed asset decisions. Likely: `app/audio_models.py`, `app/audio_service.py`, `app/production_models.py`, `app/output_quality.py`, `tests/test_audio_composition.py`. |
| **Final Quality Gate** — `EXISTS_BUT_CONFLICTS` | `app/output_quality.py: validate_output_quality()` and `validate_final_video()` currently prevent selected pre-render/technical failures; render reports persist `quality` and `validation`. | Reuse these checks inside a new owner; do not duplicate their low-level probes. The existing terminal/facade status path conflicts with the required quality ownership because it can complete a warned output before a full quality aggregate exists. | Change readiness only after backward-compatible report/desktop mapping is available. Test a blocker cannot be masked by score/count, warning aggregation and interrupted recovery. Likely: `app/output_quality.py`, `app/pipeline.py`, `app/reporting.py`, `app/gui/services/pipeline_facade.py`, `app/gui/services/desktop_services.py`, new focused quality tests. |
| **QualityReport** — `MISSING` | `report.json` and render report nested `quality`/`validation` are operational aggregates, not a stable quality-report model. | Add one typed, persisted per-artifact report referenced by `render-result.json`, `manifest.json` and the aggregate report. It should aggregate existing checks rather than duplicate their execution. | Do not make the UI infer quality from English warning strings. Test schema versioning, check evidence, persistence and status aggregation. Likely: new `app/quality_models.py` or an extension of `app/output_quality.py`, `app/reporting.py`, `app/run_manifest.py`, `tests/test_output_quality.py` (new). |
| **Artifact identity validation** — `EXISTS_BUT_NEEDS_EXTENSION` | `ClipResult`, `write_run_manifest()`, `run_artifacts.py`, `PipelineFacade.completion()`; current manifest validates run ID, candidate/plan uniqueness, revision and run-scoped paths. | Add artifact ID, parent IDs, full output checksum, edit-plan fingerprint and QualityReport reference. Extend the manifest/ClipResult; do not add a filename/index resolver. | Maintain legacy manifest fallback while requiring full identity only for new plan versions. Test same-named files, wrong candidate/run, path escape and changed MP4. Likely: `app/clip_results.py`, `app/run_manifest.py`, `app/run_artifacts.py`, `app/gui/services/pipeline_facade.py`, `tests/test_clip_results.py`, `tests/test_run_isolation.py`, `tests/test_engine_artifact_paths.py`. |
| **Rerender without reanalysis** — `EXISTS_AND_REUSABLE` | Approved draft render path (`Pipeline._run_production_from_draft()`), production-render-only path, render cache keys and GUI rerender flow already exist; tests cover no upstream re-analysis. | Keep this lifecycle. Extend invalidation from existing plan/audio/render fingerprints to the future edit-plan/QC revisions; do not add a separate rerender service. | New quality auto-fix must rerender only affected candidates and retain prior report/revision. Test subtitle/crop-only rerender, analysis call count zero, stale result rejection. Likely: `app/pipeline.py`, `app/video_composition.py`, `app/gui/services/pipeline_facade.py`, `tests/test_content_understanding_cache.py`, `tests/test_video_composition.py`. |
| **Real-media regression fixtures** — `EXISTS_BUT_NEEDS_EXTENSION` | `validation/` has local scripts, synthetic format variants, Windows smoke tooling and a fixture policy; `validation/fixtures/` contains no committed licensed-real suite. | Keep source media out of Git and extend the manifest-based validation harness with permitted local fixtures, expected quality outcomes and human review evidence. Do not call synthetic proxies evidence of semantic/editorial quality. | Fixture licensing, hashes, retention and machine availability are risks. Test the observed repeated-range plan and categories in the quality standard. Likely: `validation/README.md`, `validation/collect_health.py`, `validation/windows_media_smoke.py`, a local ignored fixture manifest, plus focused unit tests. |

## 7. Architecture Conflicts

1. **Renderer input conflict.** V2 says the renderer consumes only a validated edit plan. The actual `VideoCompositionService.compose()` takes `ProductionPlan`, `AudioProject`, `Source`, transcript, work/output paths, config and optional visual-analysis data. The solution is to validate and version the existing plan/artifact envelope, not to add a second renderer.

2. **Identity graph is incomplete downstream.** The current plan has `candidate_id` and `source_id`; the run manifest has project/run/plan/result identity. Neither `ProductionPlan` nor `VideoProject` carries the full V2 project/source/analysis/run/candidate/plan chain. File paths are correctly not the primary desktop identity, but they still serve as artifact linkage without an `artifact_id`/QualityReport reference.

3. **Boundary contract is not bound to final segments.** Candidate `boundary_diagnostics` is strong Goal 5A evidence, while production segments are fact-derived. The completed run demonstrates a different final source range and repeated fact-derived ranges. A V2 plan must explicitly reference the boundary revision and validate all source segments against it.

4. **Quality status has two conflicting meanings.** Render quality is `valid`/`warning`/`invalid`; `output_quality.py` returns `passed`/`warning`/`failed`; pipeline terminal is `completed`/`completed_with_warnings`/`failed`; desktop status is derived from warnings. None is the required `PASS`/`PASS_WITH_WARNINGS`/`BLOCKED` owner.

5. **Composition safety cannot be inferred from fallback success.** The current composition subsystem has well-typed fallback paths, but the real run's unavailable subject observations led to `center_crop` records marked `passed`. That means "no target evidence" and "composition quality passed" can coexist. V2 must distinguish unavailable evidence from demonstrated target safety.

6. **The example EditPlan is incomplete against its own architecture document.** `EDIT_PLAN_SCHEMA_EXAMPLE.json` omits the required boundary-decision reference and created timestamp. It also does not show the existing audio/render/reframe identities that must survive migration.

## 8. Migration Risks

- `ProductionPlan`, `AudioProject`, `VideoProject`, subtitle and render models use Pydantic `extra=forbid`. Add versioned optional/defaulted fields and explicit migration readers before writing new records; old draft plans must remain renderable or fail visibly as unsupported.
- The approved-draft workflow intentionally reuses plans without re-analysis. Any eligibility/score/boundary version mismatch must be detected before reuse, not silently recomputed during a render-only run.
- Source caches are shared by source fingerprint while output manifests are run-scoped. Cache keys must include new score, boundary, plan, visual-analysis, preset and QC config versions at the correct layer so a style edit does not re-run semantic analysis.
- Do not auto-merge duplicate source ranges by deleting a segment without retaining all fact/evidence links. The first safe behavior is deterministic rejection with a machine-readable reason; a later schema can model a single segment with multiple fact IDs.
- New quality blockers can change a currently completed run into blocked. Preserve historical artifacts/reports and expose a migration status rather than rewriting their meaning.
- The current visual-analysis schema failure is a production dependency risk. Composition quality must have a conservative `evidence_unavailable` state and a clearly defined fallback/block policy.
- Avoid allowing `report.json` warning strings to become a data contract. Persist codes, severity, measured values, thresholds and config versions.

## 9. Recommended Goal Sequence

1. **Goal 5B, phase 1 — Eligibility and Candidate Score V2.** Persist decision/evidence/version fields on the existing scored candidates; include the exact-source-range duplication precondition at the existing plan hand-off.
2. **Goal 5B-2 — Diversity.** Extend existing coverage selection with recorded MMR/similarity/exclusion data; retain coverage as a positive signal.
3. **Goal 5C — Boundary decision evolution.** Bind existing 5A semantic-boundary data to final source segments and plan revisions.
4. **Goal 5F — ProductionPlan/EditPlan hardening.** Promote the existing plan and adjacent audio/composition/subtitle sections into a validated V2 envelope. This should not be a new large parallel pipeline because its foundation already exists.
5. **Goal 5D — Composition Quality.** First repair/verify the subject-analysis contract, then add target/face/empty-frame metrics and explicit safe/block behavior without rebuilding scene-aware composition.
6. **Goal 5E — Subtitle Quality V2.** Add profiles, CPS, semantic breaks, geometry and mask checks around the existing fitter.
7. **Goal 5G — Final Quality Gate and QualityReport.** Make the persisted report the only owner of `PASS`/`PASS_WITH_WARNINGS`/`BLOCKED`; migrate terminal and desktop mappings to it.
8. **Goal 5H — Licensed-real regression suite.** Add local manifest-backed fixtures and human review evidence for the defined defect classes.
9. **Goal 5I — Preset calibration.** Only tune effects, crop and subtitle/audio preset behavior after quality blockers are trustworthy.

## 10. Exact Scope of the First Implementation Goal

**Name:** Goal 5B — Candidate Eligibility and Candidate Score V2 foundation.

In scope:

- Extend the existing serialized candidate/scored-candidate record with a schema/config version, `eligible`, typed reason codes, recoverable issues, component scores, penalties and evidence references.
- Reuse Goal 5A `boundary_diagnostics`, transcript/audio/scene features, StoryUnit content signature and current fallback scoring. Do not call a new provider solely to obtain the initial decision.
- Make `select_with_coverage()` accept only explicitly eligible records while retaining its existing coverage behavior.
- Define deterministic context-debt, hook and payoff evidence sufficient for a first version; unavailable evidence must be explicit rather than treated as a pass.
- Add a plan-handoff invariant that detects duplicate exact source ranges in one `ProductionPlan` and returns a machine-readable failure before audio composition. This directly covers the three real-run defects above.
- Preserve analysis/draft approval/rerender path, source cache, `ClipResult`, manifest, UI selection and existing production renderer unchanged.

Out of scope:

- no new EditPlan class or renderer;
- no embeddings/provider-only MMR (Goal 5B-2);
- no crop/subtitle/audio algorithm rewrite;
- no final QualityGate/QualityReport owner change (Goal 5G);
- no preset/effects work;
- no migration of old final artifacts' quality status.

The first Goal is successful when an ineligible candidate cannot enter coverage selection, an exact duplicate source range cannot produce duplicated audio, and a score/penalty decision is persisted and reproducible from the existing analysis inputs.

## 11. Likely Target Files

Primary first-goal files:

- `app/models.py`
- `app/transcript_features.py`
- `app/local_scoring.py`
- `app/content_understanding.py`
- `app/intelligence.py`
- `app/selection.py`
- `app/production_models.py`
- `app/production_plan.py`
- `app/config.py`
- `tests/test_semantic_boundaries.py`
- `tests/test_coverage_selection.py`
- `tests/test_production_plan.py`
- `tests/test_content_understanding_cache.py`

Later goals should extend, rather than replace, `app/video_composition.py`, `app/video_models.py`, `app/production_subtitles.py`, `app/audio_service.py`, `app/audio_models.py`, `app/output_quality.py`, `app/reporting.py`, `app/run_manifest.py`, `app/clip_results.py`, `app/run_artifacts.py`, `app/gui/services/pipeline_facade.py`, and `app/gui/services/desktop_services.py`.

## 12. Regression Plan

Unit coverage for Goal 5B:

- eligibility reason codes for invalid range, word/sentence boundary, high context debt, no payoff, missing visual evidence and low intelligibility;
- deterministic V2 score components, penalties, config version and local fallback;
- selection rejects ineligible candidates even with a high score;
- existing coverage selection remains deterministic and keeps distinct candidates;
- duplicate exact source range in a single plan is rejected with code/evidence; adjacent but non-identical ranges remain allowed;
- legacy candidate/plan records without V2 fields load as explicit legacy data or fail with a clear migration error.

Integration coverage:

- analysis -> draft -> approved render preserves candidate/plan IDs and does not repeat analysis for a render-only change;
- source candidate range, boundary decision and final plan segments remain traceable;
- manifest rejects incorrect run/candidate/plan/path identity and final MP4 checksum mismatch once added;
- a quality warning cannot be hidden by terminal output count after Goal 5G.

Real-media suite after Goal 5H:

- the three observed duplicated dialogue ranges;
- semantic near-duplicates across selected candidates;
- answer without question/hidden context;
- face at frame edge, empty table while speaker is present, speaker switch and visual-analysis-unavailable fallback;
- Cyrillic long words, high CPS, weak line break, subtitle/face/platform overlap;
- silent/noisy/clipped audio and AV desync;
- interrupted render, same-named outputs, Cyrillic paths and rerender without reanalysis.

Use the existing `validation/` policy: fixtures are licensed/local and ignored by Git; the fixture manifest and expected machine/human outcomes are versioned, not the media itself.

## 13. Required Documentation Corrections

1. `docs/README_EDITING_SYSTEM.md` references `docs/research/SHORT_FORM_EDITING_RESEARCH.md`, but that file is absent from this workspace. Restore it or remove/correct the index entry before treating it as a source.
2. `docs/architecture/EDIT_PLAN_SCHEMA_EXAMPLE.json` needs the boundary-decision reference and created timestamp required by `EDITING_PIPELINE_V2.md`; it should also say explicitly that it is a future envelope around the existing `ProductionPlan`/audio/render contracts.
3. `docs/architecture/EDITING_PIPELINE_V2.md` should identify `ProductionPlan` as the migration base and state that the current renderer has additional typed inputs until the envelope hardening Goal completes.
4. `docs/quality/OUTPUT_QUALITY_STANDARD.md` should label `PASS`/`PASS_WITH_WARNINGS`/`BLOCKED` and `QualityReport` as target behavior until Goal 5G changes terminal and desktop status ownership.
5. `docs/product/CONTENT_FACTORY_EDITING_SYSTEM.md` should distinguish desired product status values from the current operational statuses (`completed`, `warning`, `completed_with_warnings`, `failed`) to prevent a false readiness claim.
6. `docs/roadmap/EDITING_QUALITY_ROADMAP.md` should add the real-run duplicate-source-range defect as a mandatory first-goal regression and describe the visual-analysis schema-fallback prerequisite for Goal 5D.
7. `validation/architecture_audit.md` predates the current run-manifest work and still lists an append-only run manifest as a future P1. Its historical result should be marked superseded for that point; its licensed-real-fixture warning remains valid.

## 14. Open Questions

1. Should a duplicate source range in a plan initially block only that candidate, or should it be safely merged into one segment with multiple fact IDs? The latter needs a deliberate `DialogueSegment` contract evolution.
2. What evidence threshold permits a `PASS_WITH_WARNINGS` composition result when visual observations are unavailable: a designed padded fallback only, or a center crop under constrained source formats?
3. Which checks are allowed to auto-fix, and which require a new approved draft/edit-plan revision before a rerender?
4. What is the authoritative platform/preset/safe-zone registry and version policy? Current `product_flow` settings and render config are useful inputs, but there is no V2 platform-mask artifact.
5. Who supplies policy/licensing provenance for source, music and effects, and how is an unknown license represented before export?
6. What accepted manual-review evidence is required for a licensed-real fixture to establish semantic, face/target and subtitle quality beyond machine checks?
7. Should existing historical `completed` manifests be displayed as legacy technical completions after QualityReport migration, rather than retroactively as `PASS`?
