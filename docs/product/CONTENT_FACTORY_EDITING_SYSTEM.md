# CONTENT FACTORY — EDITING SYSTEM

**Status:** Source of Truth
**Scope:** Quality-controlled automatic editing of vertical short-form videos
**Applies to:** candidate selection, diversity, boundaries, composition, subtitles, audio, presets and final quality evaluation
**Does not replace:** current project lifecycle, artifact recovery, desktop UX specifications or existing render contracts unless an implementation Goal explicitly changes them

**Implementation status:** This document defines the target editing system. The
current pipeline implements only parts of it; in particular, the Final Quality
Gate and its `QualityReport` readiness ownership are not implemented yet.

## 0. Current operational statuses versus target product quality statuses

Current pipeline statuses are operational execution/recovery states:

```text
completed
warning
completed_with_warnings
failed
```

They describe whether the current pipeline or render completed and whether it
recorded warnings. They are not proof of semantic, composition, subtitle, audio
or identity quality, and must not be mapped implicitly to a product-ready state.

Target product-quality statuses, owned by `QualityReport` only after Goal 5G,
are:

```text
PASS
PASS_WITH_WARNINGS
BLOCKED
```

Until Goal 5G implements the Final Quality Gate, these target statuses are a
specification, not current pipeline output.

## 1. Product objective

Content Factory turns long-form source video into several short vertical videos that are:

1. understandable without hidden context;
2. meaningfully different from one another;
3. naturally started and finished;
4. visually safe in 9:16;
5. readable with subtitles;
6. technically valid;
7. traceable to the correct source, candidate and output artifact;
8. reproducible without repeating expensive analysis.

The product must optimize for **meaning, clarity and watchability**, not for a fixed number of zooms, transitions, sound effects or animated words.

## 2. Non-negotiable principles

### 2.1 AI proposes; code decides

AI may identify topics, claims, hooks, payoff, emotional beats, rough candidate windows, content type and editing style.

AI must not create arbitrary FFmpeg commands, write final artifacts directly, bypass validators, mark an output ready, invent ungrounded timestamps or overwrite lifecycle state.

All critical state transitions are deterministic and code-owned.

### 2.2 Quality before effects

```text
Meaning
→ eligibility
→ ranking
→ diversity
→ boundaries
→ edit plan
→ composition
→ subtitles
→ audio
→ render
→ quality gate
→ user preview
```

Effects cannot compensate for incomplete thought, duplicate content, cut words, missing faces, empty crop, unreadable captions or wrong artifact linkage.

### 2.3 Target rule: a high score never cancels a critical defect

After the Final Quality Gate is implemented, a candidate or output with a hard
blocker is rejected or marked `BLOCKED`, regardless of total score.

### 2.4 Re-render must be cheaper than re-analysis

Visual or export changes should reuse transcript, word timestamps, detected scenes, candidate identity, semantic analysis, edit-plan inputs and validated source metadata.

## 3. Candidate Eligibility Gate

A candidate enters ranking only when:

- source interval is valid;
- start/end do not cut a word;
- the clip can become syntactically complete;
- the subject or claim is understandable without excessive context;
- there is a value unit: claim, answer, result, conflict, transformation, punchline or useful takeaway;
- audio is intelligible;
- viable vertical composition or safe fallback exists;
- no policy/licensing blocker exists;
- refined duration fits product constraints;
- identity is stable and traceable.

Reject reasons:

```text
SOURCE_INTERVAL_INVALID
WORD_BOUNDARY_UNRECOVERABLE
SEMANTIC_INCOMPLETE
CONTEXT_DEBT_CRITICAL
NO_PAYOFF
AUDIO_UNINTELLIGIBLE
VERTICAL_COMPOSITION_IMPOSSIBLE
POLICY_BLOCKED
LICENSE_BLOCKED
DURATION_OUT_OF_RANGE
CANDIDATE_IDENTITY_INVALID
```

## 4. Candidate Scoring V2

```text
quality_score =
    0.16 × self_containment
  + 0.14 × hook_strength
  + 0.12 × payoff_strength
  + 0.10 × narrative_arc
  + 0.10 × informational_value
  + 0.08 × emotional_intensity
  + 0.08 × novelty_or_conflict
  + 0.08 × speech_clarity
  + 0.06 × visual_viability
  + 0.04 × pacing_density
  + 0.04 × platform_fit
  - penalties
```

Weights are versioned product parameters.

Context debt measures unresolved pronouns, unnamed entities, references to previous events, answers without enough question context, undefined terms and omitted setup.

Initial penalties:

| Defect | Policy |
|---|---|
| Starts mid-thought | `−15…−30` or reject |
| Ends before completion | `−15…−30` or reject |
| High context debt | `−10…−25` |
| Internal repetition | `−5…−15` |
| Similar to selected output | `−10…−40` |
| Empty/irrelevant visual | `−10…−30` |
| Low intelligibility | `−15` or reject |
| Long meaningless opening pause | `−5…−20` |
| False hook without payoff | reject |
| Policy/licensing blocker | reject |

## 5. Semantic deduplication and diversity

Never use:

```text
sort_by_score(candidates)[:requested_count]
```

Use clustering plus MMR-like reranking:

```text
selection_score(candidate) =
    lambda × quality_score(candidate)
  - (1 - lambda) × max_similarity(candidate, selected)
```

Initial `lambda`: `0.72–0.80`.

Composite similarity:

```text
0.45 × semantic_embedding_similarity
+ 0.20 × key_claim_similarity
+ 0.15 × named_entity_overlap
+ 0.10 × source_time_overlap
+ 0.10 × visual_scene_similarity
```

For three outputs:

- no more than two with the same payoff;
- no more than two adjacent windows from one scene;
- no more than two with the same rhetorical pattern;
- prefer at least two topic clusters when quality is close;
- a slightly lower-scoring unique clip may replace a duplicate.

## 6. Boundary refinement

Goals:

- no cut words or phonemes;
- no start inside hidden sentence;
- no end before thought/reaction/payoff completion;
- preserve meaningful pauses;
- remove filler only when natural;
- align with scene/speaker transitions where useful.

Initial ranges:

- pre-roll `120–350 ms`;
- extended pre-roll up to `500 ms`;
- post-roll `180–600 ms`;
- jump-cut silence threshold approximately `500–800 ms`.

Every boundary stores rough/refined timestamps, selected boundary, pre/post-roll, scene/speaker evidence, reason code, confidence and fallback usage.

## 7. Editing rhythm

Visual resets are event-driven, not timer-driven.

Valid triggers:

- new semantic beat;
- speaker change;
- important number or term;
- problem → explanation → result;
- source scene cut;
- meaningful object appearance;
- significant reaction;
- long static interval.

Default policy:

- hard cut is primary;
- dissolve only for time/place/mood transition;
- whoosh only with directional movement;
- glitch only in compatible preset;
- transition never hides bad boundaries;
- no transition splits a continuous phrase.

Punch-in:

- only on important claim, reaction or static-shot relief;
- ordinary range `105–115%`;
- stronger `115–122%` only for compatible presets;
- fallback to text emphasis or no effect;
- block if face clipping or unstable movement appears.

## 8. Vertical composition

Short-form output uses an actual 9:16 crop/reframe for every source segment.
The crop is resolved from the existing scene/subject/target evidence and may
move only at evidence-backed scene or target changes. A candidate-wide
full-frame `fit_background` / blurred background is not a default composition
fallback; sparse evidence uses a calm 9:16 stable crop, and an unresolved
protected target is surfaced as a composition failure rather than hidden by
blur.

Priority:

```text
semantic target
→ active speaker
→ relevant object/screen region
→ stable group framing
→ designed padded fallback
```

Do not choose the largest face by default.

Required metrics:

```text
face_clipping_ratio
headroom_ratio
target_occupancy
speaker_target_match
crop_velocity
crop_acceleration
crop_switch_frequency
salient_object_visibility
empty_frame_ratio
subtitle_overlap_ratio
platform_ui_overlap_ratio
```

Critical failures:

- active face materially clipped;
- semantic target absent for sustained interval;
- empty background/table shown while confident target exists;
- excessive crop movement;
- repeated switches without real speaker change;
- subtitles cover face, UI or key action.

## 9. Platform safe zones

Safe zones are versioned data, not permanent constants.

Minimum masks:

```text
tiktok_organic_default
instagram_reels_default
youtube_shorts_default
```

Variants:

```text
with_long_caption
with_cta
with_product_anchor
rtl_language
large_accessibility_text
```

## 10. Subtitle system

Base rules:

- max two lines;
- syntax-aware line breaks;
- no break after preposition/conjunction;
- no early punchline reveal;
- avoid long ALL CAPS;
- no random font/color changes;
- no more than two simultaneous emphasis types;
- verify names and critical terms;
- validate actual rendered bounding boxes.

Initial Russian profile:

| Parameter | Value |
|---|---|
| Target reading speed | `13–17 CPS` |
| Short allowed maximum | `20 CPS` |
| Lines | max 2 |
| Simple phrase minimum | `0.8–1.0 s` |
| Typical phrase maximum | `4–6 s` |
| Dynamic words per screen | `2–7` |
| Emphasized words | `0–2` |

Allowed motion primitives:

- fade;
- controlled scale emphasis;
- simple slide;
- progressive reveal with reliable timestamps;
- karaoke only with strong timing confidence;
- bounce only in compatible presets;
- type-on only for short title/hook cards.

Weak timing degrades to static phrase captions.

## 11. Audio system

Priority:

```text
dialogue intelligibility
→ essential source sound
→ music bed
→ semantic sound accents
→ decorative effects
```

Initial targets:

- around `−16 LUFS`;
- preferred range `−18…−14 LUFS`;
- true peak not above `−1 dBTP`.

Ducking defaults:

- reduction `6–14 dB`;
- attack `50–150 ms`;
- release `250–700 ms`.

Unknown or incompatible licensing blocks export.

## 12. MVP presets

1. `Clean Podcast`
2. `Viral Expert`
3. `Educational`
4. `Minimal Premium`

A preset configures allowed effects, limits, crop behavior, subtitle profile, typography, audio policy, fallbacks and forbidden behavior.

A preset never overrides quality blockers.

## 13. Target user-facing quality status (after Goal 5G)

```text
PASS
PASS_WITH_WARNINGS
BLOCKED
```

- `PASS`: ready.
- `PASS_WITH_WARNINGS`: usable, concrete warnings shown.
- `BLOCKED`: export disabled until fixed or safe fallback succeeds.

## 14. Provenance and originality

Store source declaration and provenance. Warn for third-party film/series/repost use. Never promise that crop, captions or zoom make content original. Never remove watermarks to evade attribution. Do not declare fair use automatically. Preserve asset license manifests.

## 15. MVP scope

Include:

- eligibility;
- scoring V2;
- context debt;
- hook/payoff;
- semantic dedupe;
- diversity;
- boundary refinement;
- active-speaker smoothing;
- safe composition fallback;
- face/empty-frame checks;
- RU/EN subtitle profiles;
- syntax-aware segmentation;
- CPS validation;
- four MVP presets;
- dialogue normalization;
- optional licensed music/SFX;
- deterministic edit plan;
- artifact traceability;
- quality gate;
- real-media regression fixtures;
- Windows smoke tests.

Defer:

- B-roll retrieval;
- generative inserts;
- advanced kinetic typography;
- retention-trained personalization;
- automatic analytics ingestion;
- huge SFX/music library;
- direct publishing;
- complex timeline editor;
- AI-generated hook variants.

## 16. Product acceptance criteria

- requested outputs are meaningfully different;
- no output begins/ends mid-word;
- each output is independently understandable;
- critical face/target failures are blocked;
- subtitle geometry and CPS pass;
- every UI card points to correct MP4;
- rerender is reproducible;
- visual-only rerender does not repeat semantic analysis;
- interrupted work restores valid artifacts;
- warnings/blockers are machine-readable and explainable.
