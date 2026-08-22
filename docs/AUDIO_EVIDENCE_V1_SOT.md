# Content Factory — Audio Evidence v1 Source of Truth

**Status:** target behavior for Audio Intelligence v1. Current behavior remains the code and persisted runtime artifacts.

## Purpose and ownership

Audio Intelligence v1 extends the existing source-scoped Multimodal Evidence and Semantic Brain. It does not create another clip engine, candidate store, selection path, boundary owner, renderer, or lifecycle.

```text
prepared 16 kHz mono PCM
→ one cached local signal-analysis pass
→ bounded semantic-audio inference
→ existing Multimodal Timeline / candidate provenance
→ Candidates → Brain → Diversity → Boundaries → Preview/Final
```

Audio peaks may create bounded candidate seeds. Seeds are proposals only. Stable candidate identity, semantic/boundary validation, eligibility, ranking, diversity, and final selection remain code-owned by the existing pipeline.

## Evidence contract

The cached `audio_features.json` pass measures relative RMS loudness, robust source-relative spikes, onsets/burst density, active/quiet intervals and dead zones, plus a bounded zero-crossing-rate noisiness proxy. It streams the already prepared PCM once and never decodes the source video again.

Semantic events use a YAMNet-compatible AudioSet classifier only on:

- a bounded, deduplicated set of the strongest local audio peaks; and
- a bounded set of existing shortlisted candidate regions.

The expensive classifier must not scan the whole source. Persist its model/config/input fingerprints, selected regions, event timestamps, class IDs/labels, confidence, inference status, and cache diagnostics. Raw model output is never executable input.

## Runtime and deployment

The v1 runtime is CPU `onnxruntime` with a pinned Apache-2.0 YAMNet ONNX graph whose mel frontend consumes raw 16 kHz mono float waveform. The model and AudioSet class map are packaged local assets with verified SHA-256; production analysis performs no model download. TensorFlow and `tensorflow-hub` are forbidden production dependencies for v1.

Missing runtime/model, hash mismatch, inference failure, or unsupported graph is an explicit `unavailable`/`partial` evidence state and a warning, not a new candidate blocker. Signal evidence and the rest of the batch continue.

## Brain interpretation

The Semantic Brain receives a compact, allowlisted summary: activity and dead-zone ratios, strongest spikes/onsets, meaningful semantic events, background-music state, candidate-seed provenance, and evidence availability. Audio event confidence is evidence confidence, not a quality or payoff score.

All 15 profiles use the same evidence contract with profile-aware interpretation:

- speech-led (`podcast`, `interview`, `talking_head_expert`, `news_commentary`) values reactions/emphasis but still requires coherent meaning;
- action/result-led (`gameplay`, `stream`, `sports_fitness`) may use impacts, explosions, crowd/reaction and game/sports sounds as action evidence;
- process/reveal-led (`food`, `tutorial_education`, `review`) may use cooking, tools, mechanisms and result/reaction sounds when tied to the candidate;
- scene/experience-led (`vlog_lifestyle`, `travel`, `reaction`, `story_entertainment`, `movie_series`) may use grounded environmental, reaction and action events without requiring dense speech.

Background music alone never adds recommendation value. A semantic audio event alone is not payoff; payoff requires a meaningful result/resolution from the candidate's combined text, Vision and audio evidence.

## Sparse-content policy

Apply a strong code-owned soft downgrade when one candidate contains a long semantic/speech gap and has neither meaningful Vision action/payoff nor meaningful audio activity/event. Do not create a new `BLOCKED` state: the candidate remains `AVAILABLE` and selectable.

Low speech must not be penalized when strong visual or audio action evidence exists. Tight, self-contained candidates outrank duration-padded candidates; target duration never overrides semantic completeness or existing boundary invariants.

## Cache and invalidation

Signal cache key: prepared-audio fingerprint + signal config/version. Semantic-audio cache key: Audio Evidence/content-map/Vision fingerprints + model/class-map hashes + semantic config/version + deterministic bounded-region derivation inputs.

Audio signal/model/config changes invalidate Audio Evidence, candidate provenance, Brain scoring and downstream selection only. They do not invalidate transcription, source/scene analysis, boundary algorithms, Creative/Render, or Preview/Final lineage. A pre-v1 Vision artifact is rebound locally without a provider call only when its complete keyframe ID/timestamp plan and provider contract are unchanged; a real Vision-input change still invalidates it. Style, caption, crop, playback/mix and render changes never rerun Audio Intelligence.

## Acceptance

Acceptance requires deterministic synthetic coverage across podcast/interview, gameplay, food, sports/fitness, tutorial and movie/series; a real saved Gameplay before/after audit; visible audio-derived seed diagnostics; cache-hit proof; no extra full source decode; and proof that a failed candidate or unavailable classifier does not terminate the batch.
