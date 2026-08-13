# Content Factory repository guide

## Source of truth

Before changing code, audit the current implementation and read [the editing-system index](docs/README_EDITING_SYSTEM.md). Keep target and current-state claims separate:

- **TARGET behavior:** latest explicit user decision → current SOT docs ([quality](docs/quality/OUTPUT_QUALITY_STANDARD.md), [architecture](docs/architecture/EDITING_PIPELINE_V2.md), [product](docs/product/CONTENT_FACTORY_EDITING_SYSTEM.md), and the UI [screen behavior contract](docs/CONTENT_FACTORY_SCREEN_BEHAVIOR_CONTRACT.md)) → detailed specifications and execution plans → historical research and audit snapshots.
- **CURRENT behavior:** runtime artifacts and current code → validation/audit reports → docs. Never use a target document alone to claim that behavior is implemented.

For UI detail, use the [creative UX spec](docs/CREATIVE_UX_SPEC.md) and [approved visual references](docs/CONTENT_FACTORY_VISUAL_REFERENCES.md). For creative rendering, use [Phase 7](docs/PHASE_7_CREATIVE_SYSTEM.md) and its [rendering architecture](docs/CREATIVE_RENDERING_ARCHITECTURE.md); use the [roadmap](docs/roadmap/EDITING_QUALITY_ROADMAP.md) for execution order. Discovery and historical documents guide design but do not authorize extra scope.

## Non-negotiable invariants

- **AI proposes; code decides.** AI output is evidence-grounded, schema-validated input. Deterministic code owns executable parameters, identities, validation, lifecycle transitions, readiness, filesystem writes, and renderer commands. Raw AI output never reaches FFmpeg or the shell.
- Reuse transcript, Brain/Vision analysis, evidence, and validated artifacts. Style, caption, crop, motion, preview, or final-only changes must invalidate only their dependency descendants and must not silently rerun expensive analysis.
- Structural Preview is not Creative Preview. Creative Preview and Final share the approved intent revision, compiled plan/hash, 30 fps time base, cue and event frames, normalized geometry, fonts, and asset hashes; render profiles may change only resolution, bitrate, encoder, and declared precision. A parity mismatch blocks Final approval.
- Preserve stable identity and lineage from project/source/analysis/candidate through plan, render, artifact, and quality report. A file path or list position is never identity.
- One failed or `BLOCKED` candidate must not kill the batch. Preserve valid candidates and artifacts; isolate fallback, retry, and failure state to the affected candidate.
- Extra source shots are same-source only, require explicit opt-in, and are **OFF by default**. Stock and generative B-roll are forbidden.
- Do not translate operational `completed`/`warning` states into target `PASS`/`BLOCKED` claims. Use warnings and explicit safe fallbacks for minor or recoverable issues. Use `BLOCKED` only for a real publishability or safety failure such as invalid identity/media, broken meaning, unreadable or unsafe layout, unacceptable crop, license risk, or Preview/Final parity failure.

## Working protocol

- Ordinary Codex work is the default. Use a Goal only for systemic verification that genuinely needs a persistent, cross-layer audit or real-media benchmark; never for routine implementation or focused tests.
- Parallel work may use 2–3 lanes only when they have no file, contract, or dependency overlap. Otherwise work sequentially; one lane owns integration.
- Follow: implementation → focused tests (then broader tests as risk requires) → inspect the complete diff → stage explicit target paths → isolated commit → **NO PUSH** → post-commit audit.
- Check `git status` before and after work. Dirty and untracked files are user-owned: do not edit, delete, reset, overwrite, stage, or include them. Never use `git add .`. Without explicit user permission, never use `reset`, `clean`, `--force`/force-push, `rebase`, `commit --amend`, or `stash`. Push only when the user explicitly asks.
- UI changes require real-window QA on the target Windows flow; headless/widget tests alone are insufficient. Media changes require a real MP4, probe plus visual/audio review, and verified lineage/Preview–Final parity; synthetic media alone is not publishability proof.

## Forbidden scope

Do not create a parallel pipeline, plan store, renderer, quality gate, or project/artifact lifecycle. Do not add direct publishing/cloud integrations, unrequested dependencies or schema migrations, or deferred product features. Do not change selection, composition, or caption algorithms inside a UI-only task. Do not rewrite SOT docs or broaden the task unless the user explicitly authorizes it.

## Final report

Keep it short: outcome/root cause; changed files; tests and real QA evidence; commit SHA; remaining warnings or genuine blockers; final git status; explicit `NO PUSH`.
