# CONTENT FACTORY — EDITING PIPELINE V2

**Status:** Architecture Source of Truth

## 1. Integration constraint

Extend the existing pipeline. Do not create a parallel candidate, render, artifact or project lifecycle.

`ProductionPlan` is the migration base for the future versioned edit-plan envelope.
V2 must evolve that plan and the existing typed contracts around it; it must not
introduce a second plan store or a parallel EditPlan renderer.

```text
Existing source analysis
→ CandidateEligibilityGate
→ CandidateScorerV2
→ DiversityReranker
→ BoundaryRefiner
→ EditPlanBuilder
→ CompositionPlanner
→ SubtitlePlanner
→ AudioPlanner
→ Existing renderer
→ QualityGate
→ Existing artifact lifecycle and desktop UI
```

Before implementation, map these concepts to current modules and contracts.

## 2. Architectural invariants

1. One documented time base.
2. Immutable candidate/run/project/source/artifact IDs.
3. Stale async results cannot attach to another run/candidate.
4. Renderer consumes only validated edit plans.
5. AI output is schema-validated and evidence-grounded.
6. Every fallback is explicit and logged.
7. Expensive analysis is reusable.
8. Final readiness is owned by QualityGate.
9. Project state derives from persisted facts.
10. Recovery resumes missing work only.
11. UI thread never runs blocking media analysis.
12. Schema versions are explicit.

## 3. Identity graph

```text
project_id
  └── source_id
       └── analysis_run_id
            └── candidate_id
                 └── edit_plan_id
                      └── render_run_id
                           └── artifact_id
                                └── quality_report_id
```

Rules:

- file path is never identity;
- UI selection stores IDs, not indexes or filenames;
- boundary revision creates new edit plan, not new source;
- visual rerender creates new render run;
- artifact metadata stores all parent IDs.

## 4. Core contracts

### CandidateAnalysis

Must contain:

- schema version;
- project/source/run/candidate IDs;
- rough interval;
- transcript refs;
- topic, key claim, hook, payoff, rhetorical pattern;
- feature scores;
- context debt;
- evidence;
- model provenance.

### EligibilityDecision

Must contain:

- candidate ID;
- eligible boolean;
- reason codes;
- recoverable issues;
- required boundary actions;
- evidence refs;
- config version.

### CandidateScore

Must contain:

- raw score;
- penalties with evidence;
- final score;
- scoring config version;
- strengths/risks.

### DiversityDecision

Must contain:

- requested count;
- selected candidate IDs;
- exclusions with similarity/reason;
- lambda;
- config version.

### BoundaryDecision

Must contain:

- rough/refined timestamps;
- start/end reasons;
- word/sentence/semantic completeness flags;
- pre/post-roll;
- confidence;
- evidence;
- fallback flag.

## 5. Deterministic Edit Plan

### Migration rule and example scope

This section describes the target state, not a current production contract.
The current renderer still consumes several typed inputs, including the existing
`ProductionPlan`, `AudioProject`, source/transcript data and visual-analysis
data, and it persists or uses `ReframePlan`, `SubtitleProject` and `VideoProject`.

The future edit-plan envelope must evolve and reference those existing contracts.
`EDIT_PLAN_SCHEMA_EXAMPLE.json` is an illustrative future envelope around
`ProductionPlan`, `AudioProject`, `ReframePlan`, `SubtitleProject` and
`VideoProject`; it does not replace them today and must not be used to create a
parallel EditPlan renderer.

The edit plan is the only renderer input.

Required fields:

- schema version;
- edit plan ID;
- project/source/analysis/candidate IDs;
- boundary decision reference;
- preset ID/version;
- target format;
- segments;
- composition plan;
- subtitle plan;
- audio plan;
- platform mask;
- expected duration;
- warnings;
- input fingerprints;
- created timestamp.

After Goal 5F hardening, the renderer must reject invalid or identity-mismatched
plans. Until then, the existing multi-input renderer remains the production path
and must be evolved through compatible adapters/migrations.

## 6. Validation pipeline

```text
schema validation
→ identity validation
→ source fingerprint validation
→ timestamp validation
→ overlap/gap validation
→ boundary invariants
→ composition coverage
→ subtitle geometry precheck
→ audio licensing
→ preset constraints
→ expected output contract
```

Required errors:

```text
EDIT_PLAN_SCHEMA_INVALID
IDENTITY_MISMATCH
SOURCE_FINGERPRINT_MISMATCH
TIMESTAMP_OUT_OF_RANGE
SEGMENT_OVERLAP_INVALID
BOUNDARY_WORD_CUT
COMPOSITION_GAP
SUBTITLE_TIMING_INVALID
LICENSE_BLOCKED
PRESET_CONSTRAINT_VIOLATION
OUTPUT_CONTRACT_INVALID
```

### 6.1 Decision: shot-aware subject-lock virtual camera

`CompositionPlanner` remains the sole owner of vertical crop geometry.  It
plans one source shot at a time from existing verified observations; it does
not turn detector samples into per-frame crop commands.

- A shot starts with an `ACQUIRE → LOCKED` subject decision keyed by the
  stable evidence track ID.  A loss of evidence is `TEMPORARILY_OCCLUDED`, not
  a target switch.  A new subject needs sustained, audio-visual evidence and
  an editorial reason before `SWITCH_PENDING → SWITCH_CONFIRMED`.
- The hard geometry is the semantic `must_keep_core` (face, head and critical
  shoulders).  Upper body, hands and context are soft objectives.  A planner
  must emit an evidence gap or `BLOCKED` decision when the hard region cannot
  be shown; it must not claim success by measuring only a convenient person
  box.
- Static-first feasibility intersects the safe crop-centre ranges over the
  whole shot.  A non-empty intersection produces exactly one stationary crop.
  Otherwise the deterministic planner partitions the shot into the minimum
  practical sequence of static holds and only joins adjacent holds with an
  explicit eased `HOLD → MOVE → HOLD` episode.
- Source cuts reset crop and subject state.  Crop interpolation, identity
  inheritance and motion velocity never cross a source-cut boundary.
- The lexicographic priority is: core visibility; justified identity; fewest
  camera moves; shortest travel; lowest acceleration; then preferred/context
  preservation.  Talking-head planning locks Y and scale unless the existing
  approved layout policy requires a different mode.

Rendered-output verification is independent of planner boxes and remains a
separate gate.  Its provisional result is never a visual GO: until a human
reviews the real MP4, the artifact state is `AWAITING MANUAL VISUAL ACCEPTANCE`.

## 7. AI boundary

Allowed:

- semantic candidate proposal;
- claim/hook/payoff extraction;
- content-type classification;
- context-debt proposal;
- style recommendation;
- natural-language explanation.

Required safeguards:

- strict JSON schema;
- evidence refs;
- bounded enums;
- confidence;
- request/response version;
- deterministic fallback;
- no direct filesystem writes;
- no final status mutation;
- no ungrounded timestamps;
- idempotency for paid calls.

## 8. Caching and invalidation

Reuse keys:

| Layer | Key |
|---|---|
| Source probe | source fingerprint |
| Transcript | source fingerprint + ASR config |
| Scene/face analysis | source fingerprint + model/config |
| Semantic analysis | transcript/visual fingerprints + prompt/schema version |
| Boundary decision | candidate + timing + config |
| Edit plan | parent IDs + preset + overrides |
| Render | edit-plan hash + renderer version |
| Quality report | artifact fingerprint + QC config |

Examples:

- subtitle color change → subtitle plan/edit plan/render/QC only;
- crop change → composition plan/edit plan/render/QC;
- scoring weight change → scoring/diversity only;
- source replacement → invalidate all dependents.

## 9. Artifact identity

Suggested identity-based structure:

```text
projects/<project_id>/
  sources/<source_id>/
  analysis/<analysis_run_id>/
  candidates/<candidate_id>/
  edit-plans/<edit_plan_id>.json
  renders/<render_run_id>/
    output.mp4
    render-metadata.json
    quality-report.json
```

UI resolves output by `artifact_id`, never filename order.

## 10. Desktop constraints

Do not run in UI thread:

```text
subprocess.run(...)
process.waitForFinished(...)
full ffprobe scan
thumbnail extraction
model inference
frame-by-frame QC
font probing
```

Use `QProcess`, workers/QThreadPool, Qt signals, explicit cancellation and process-tree termination.

## 11. Recovery and idempotency

- validated artifacts remain reusable;
- missing outputs resume individually;
- crash/cancel marks run interrupted;
- file existence alone never means complete;
- output must match expected candidate/run and validate;
- stale results cannot update newer run;
- old interrupted runs cannot override newer success.

Recommended idempotency key:

```text
operation + project_id + run_id + candidate_id + input_hash + config_version
```

## 12. QualityGate integration

Readiness requires:

```text
render completed
AND artifact identity valid
AND technical QC passed
AND boundary QC passed
AND composition QC passed
AND subtitle QC passed
AND audio QC passed
AND semantic/duplicate QC passed
```

UI reads persisted quality report.

## 13. Testing

Unit:

- score;
- penalties;
- eligibility;
- similarity;
- MMR;
- boundary snapping;
- plan validation;
- identity;
- quality aggregation.

Integration:

- source → candidate → plan → render → QC;
- partial failure/resume;
- stale result rejection;
- same-named outputs;
- Cyrillic paths;
- CPU fallback;
- rerender without repeated analysis.

Real fixtures:

- podcast;
- interview;
- screen demo;
- wide face-at-edge;
- dramatic pauses;
- noisy audio;
- duplicates;
- subtitle stress.

## 14. Migration sequence

1. repository audit;
2. eligibility + scoring V2;
3. diversity;
4. boundaries;
5. edit-plan hardening;
6. composition QC;
7. subtitle QC;
8. final quality gate;
9. real-media suite;
10. preset calibration.

Each Goal preserves backward compatibility or includes explicit migration.
