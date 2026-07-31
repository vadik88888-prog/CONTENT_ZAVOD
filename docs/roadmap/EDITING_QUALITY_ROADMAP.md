# CONTENT FACTORY — EDITING QUALITY ROADMAP

**Status:** Execution plan

## Audit 0 — Repository and real-run audit

No code changes, no commit, no push.

Inspect current scoring, artifacts, boundaries, composition, subtitles, quality checks, invalidation, recovery and three real outputs.

Output:

- map current modules to V2 architecture;
- evidence-backed root causes;
- reusable components;
- gaps;
- migration risks;
- target files per Goal.

## Goal 5B — Candidate Selection Quality V2

Implement:

- eligibility;
- self-containment;
- context debt;
- hook/payoff;
- versioned score;
- penalties/evidence;
- schema validation;
- deterministic fallback.
- mandatory regression: reject a `ProductionPlan` containing the same exact
  source range more than once; that candidate must not reach
  `AudioCompositionService` with the repeated range.

Done when incomplete candidates are rejected, invalid AI output cannot enter
ranking, and a candidate plan with a duplicate exact source range is blocked
before audio composition.

## Goal 5B-2 — Semantic Diversity

Implement:

- semantic clustering;
- key-claim/payoff similarity;
- time overlap;
- MMR reranking;
- duplicate reason codes.

Done when automatic selection avoids three near-identical clips.

## Goal 5C — Boundary Refinement

Implement:

- word-safe start/end;
- clause/sentence boundaries;
- pre/post-roll;
- payoff completion;
- question context;
- intentional pause preservation;
- persisted decision.

Done when no output starts/ends mid-word and complete thought survives.

## Goal 5F — Deterministic Edit Plan Hardening

Extend current plan if it exists.

Implement:

- schema version;
- identity graph;
- segment contract;
- plan refs;
- fingerprints;
- deterministic serialization;
- validation;
- rerender cache keys.

Done when renderer accepts validated plan only and visual rerender does not repeat semantic analysis.

## Goal 5D — Composition Quality

Implement:

- face/target metrics;
- hysteresis;
- empty-frame detection;
- crop movement/switch checks;
- safe fallback;
- blocking critical failures.

Prerequisite: first fix or stabilize the strict JSON schema used by visual
analysis and cover its fallback in regression. Missing visual evidence must not
automatically be treated as safe composition; it requires an explicit
uncertainty/fallback decision or a block under the applicable quality policy.

Do not rebuild scene-aware composition.

## Goal 5E — Subtitle Quality V2

Implement:

- language profiles;
- syntax segmentation;
- max two lines;
- CPS checks;
- rendered geometry validation;
- overlap checks;
- approved motion primitives.

## Goal 5G — Final Quality Gate

Implement:

- `PASS / PASS_WITH_WARNINGS / BLOCKED`;
- blocker/warning catalog;
- QualityReport;
- identity-first validation;
- technical/semantic/duplicate/composition/subtitle/audio aggregation;
- user explanations;
- safe auto-fix orchestration.

## Goal 5H — Real-media Quality Suite

Fixture categories:

- podcast;
- interview;
- lecture/screen demo;
- face-at-edge;
- dramatic scene;
- noisy audio;
- duplicate candidates;
- subtitle stress.

## Goal 5I — Preset Calibration

Only after foundations:

- Clean Podcast;
- Viral Expert;
- Educational;
- Minimal Premium;
- effect limits;
- crop/subtitle/audio parameters;
- config versioning;
- manual A/B evaluation.

## Deferred

- B-roll retrieval;
- generative inserts;
- retention learning;
- analytics ingestion;
- advanced kinetic typography;
- full gameplay/cinematic engines;
- direct publishing;
- full timeline editor;
- huge bundled media library;
- hook variants.

## Goal protocol

For each Goal:

1. Read source-of-truth docs.
2. Audit current code.
3. Reuse existing contracts.
4. Define target files.
5. Add focused regression tests.
6. Run focused/full tests.
7. Run project checks and doctor.
8. Run relevant Windows/media smoke.
9. Inspect diff.
10. Commit only target files.
11. Never use `git add .`.
12. Do not push unless asked.
13. Report root cause, files, tests, commit SHA and git status.

Priority:

```text
artifact correctness
> semantic completeness
> boundary quality
> composition safety
> subtitle readability
> audio intelligibility
> stylistic effects
```

Release gate:

- three distinct outputs;
- all independently understandable;
- no cut words;
- no critical face/target loss;
- readable subtitles;
- correct MP4 per UI card;
- restart recovery;
- rerender reuses expensive analysis;
- statuses match manual inspection.
