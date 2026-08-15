# Friend Beta Desktop corrective pass — QA evidence

## Root causes and integration map

This pass preserves the engine/backend integration from `608c0a38`. It changes the desktop projection and narrowly bounds one proven provider wait; it does not replace Brain/Vision, Editorial Policy, BoundaryDecision, the creative renderer, thresholds, or artifact lineage.

| Defect | Root cause | Corrective integration |
|---|---|---|
| Posters disappeared or remained stale | `DesktopProject.thumbnail_path` was persisted but Projects did not consume it; preview posters lived in OS temp; the serial candidate thumbnail process retained work from the previous screen projection | Persist the exact source-revision poster identity under the project, reuse project-owned poster caches, and replace stale pending queues on screen refresh |
| Settings looked like a form and its sample was not production-equivalent | Profile/style/caption owners were hidden behind combo boxes; the earlier sample used local widget animation | Project Auto + 15 profiles, 4 creative families and 7 caption presets as visual choices. Package 28 canonical 4×7 MP4 samples through the existing creative-plan, caption-plan, ASS/libass and bundled-font owners. Runtime resolves exact style/preset/version identity from the manifest; no filename scan |
| Settings sample was black in Windows evidence | `QVideoWidget` uses a native surface that is not captured by `grabWindow` while playing | The same `QMediaPlayer` decodes the canonical MP4 through `QVideoSink` into the QWidget tree for this compact sample. It is real decoded media, not a Qt animation or synthetic poster |
| Advanced crop/reframe did not reach runtime | the resolved product preset overwrote persisted `composition_strategy` | Apply the existing resolved baseline first and the explicit persisted project override last; regression proves UI → project settings → runtime config → existing composition owner |
| Draft overrides rendered immediately | view-model revisions called `build_drafts()` per click | Candidate-scoped Style/Captions/Crop/Boundaries/Extra shots now become pending. One CTA rebuilds one/all changed Drafts; the previous immutable Preview remains visible until replacement. Analysis identity is retained and Brain/Vision are not called |
| Final and normal flow exposed technical noise | raw warnings and a generic critical-count badge were projected directly | Human-readable quality/status/action copy and the first actionable doctor issue replace internal dumps; structural failures remain intact |
| UI/media refresh stalled | stale serial thumbnail work and media setup were retained across rebuilt screens | queued thumbnail work is coalesced by exact identity, FFmpeg extraction is one-threaded, and Qt media handoff remains asynchronous |
| A 51.633 s Draft took 631.6 s | 551.62 s was nested OpenAI SDK connection retry time before the existing local fallback | only the transformation client now has a 45 s request timeout and `max_retries=0`; the application's three attempts and deterministic fallback remain. Brain/Vision/scoring providers and renderer are unchanged |
| Gameplay produced 1 AVAILABLE / 13 BLOCKED | the inspected run had no provider evidence | See `gameplay-analysis-provider-audit.json`: both credential variables were absent; semantic AI was disabled with 0 tokens; Vision was not called and sent 0 frames. This run is not a production benchmark and caused no Boundary/Editorial/threshold change |

Existing capability owners reused by the UI:

| Capability | Existing owner |
|---|---|
| Auto + 15 profiles | `app/content_profile_taxonomy.py`, `app/product_flow.py` |
| “Что искать?” and policy intent | `app/product_flow.py`, `app/editorial_profile_policy.py` |
| Deep-analysis and production controls | `ProjectOptions` → `ProcessingIntent` → `ResolvedProcessingConfig` → existing runtime config |
| RECOMMENDED / AVAILABLE / BLOCKED | persisted Analysis recommendation, eligibility and production-feasibility decisions |
| Four creative families | `app/creative_policy.py` |
| Seven caption presets and fonts | `app/caption_presets.py`, `app/font_assets.py`, `assets/fonts/` |
| Creative Preview and Final | persisted `DraftArtifact.preview`, `ClipResult`, run manifest and quality report |
| Candidate overrides/recovery | existing candidate settings/status/artifact registries in `DesktopProject` and `DesktopServices` |

## Changed target files

Runtime/UI:

- `app/ai.py`
- `app/settings_preview_assets.py`
- `app/gui/components/candidate_thumbnail.py`
- `app/gui/components/final_results.py`
- `app/gui/components/processing_progress.py`
- `app/gui/components/video_preview.py`
- `app/gui/main_window.py`
- `app/gui/screens/onboarding_screen.py`
- `app/gui/screens/project_screen.py`
- `app/gui/screens/projects_screen.py`
- `app/gui/services/desktop_services.py`
- `app/gui/services/pipeline_facade.py`
- `app/gui/styles/theme.qss`
- `app/gui/viewmodels/project_viewmodel.py`

Packaging/assets:

- `packaging/generate_settings_previews.py`
- `assets/settings-previews/manifest.json`
- 28 canonical MP4 samples under `assets/settings-previews/{documentary,dynamic,educational,minimal}/`

Regression/evidence:

- `tests/test_content_transformation.py`
- `tests/test_friend_beta_desktop_integration.py`
- `tests/test_projects_responsive.py`
- `tests/test_candidate_workspace.py`
- `tests/test_final_results_workspace.py`
- `validation/friend_beta_desktop_real_window_qa.py`
- this evidence directory

## Real-window screenshots

Each screenshot is a shown native Windows `MainWindow` captured from its real HWND. The QA transaction reads exact persisted Analysis, DraftArtifact, Creative Preview, ClipResult and Final identities; it copies only desktop metadata into an isolated temporary data directory.

| Screen | 100% | 125% | 150% |
|---|---|---|---|
| Source / Projects | [PNG](source-dpi100.png) | [PNG](source-dpi125.png) | [PNG](source-dpi150.png) |
| Settings | [PNG](settings-dpi100.png) | [PNG](settings-dpi125.png) | [PNG](settings-dpi150.png) |
| Processing | [PNG](processing-dpi100.png) | [PNG](processing-dpi125.png) | [PNG](processing-dpi150.png) |
| Moments | [PNG](moments-dpi100.png) | [PNG](moments-dpi125.png) | [PNG](moments-dpi150.png) |
| Drafts | [PNG](drafts-dpi100.png) | [PNG](drafts-dpi125.png) | [PNG](drafts-dpi150.png) |
| Final | [PNG](final-dpi100.png) | [PNG](final-dpi125.png) | [PNG](final-dpi150.png) |

Machine-readable metrics: [100%](runtime-evidence-dpi100.json), [125%](runtime-evidence-dpi125.json), [150%](runtime-evidence-dpi150.json). All 18 states report zero horizontal scrollbar range, no clipped primary CTA, one primary CTA per actionable screen (none while Processing is running), correct DPR 1.0/1.25/1.5, and the normal three-column Draft composition at each profile.

## Persisted artifact lineage

The real-window transaction resolves:

`project aa71f33b… → Analysis analysis-33eaad… → Draft draft-ab56c7… → candidate-chapter-008-story-004 → Creative Preview SHA 3fe2f7… → ClipResult candidate…:production… → run cb2d9728… → Final SHA cfa347…`

The independent AVAILABLE proof is [available-to-final-e2e.json](available-to-final-e2e.json):

- non-recommended but eligible `candidate-chapter-009-story-001`, source interval `521.04–542.17`;
- candidate-scoped Dynamic + Word Pop 2.1.0 + fit-background, same-source extra shots OFF and zero B-roll segments;
- Draft `draft-ffdb85dbf7e1b077` with Creative Preview/parity identity;
- Final run `881b54b7db2c4cb598ffaaf6a5d1f7e3`, canonical 1080×1920 output, 10,828,011 bytes;
- Final SHA-256 `28c48f…` equals both the persisted output hash and manifest artifact checksum;
- Analysis, Draft artifact, Preview and Final hashes were re-read from disk and all match; Draft/Final contain zero Analysis/Vision stages.

## Performance and gameplay audit

See [performance-before-after.json](performance-before-after.json).

| Transaction | Before | After |
|---|---:|---:|
| Moments verified Analysis loads | 5 | 1 |
| Moments project-open wall time | 2.1803 s | 0.0357 s |
| Initial cards | 1 | 12 |
| Draft total | 631.6 s measured | ≤214.98 s conservative regression bound; fresh measured run pending |
| Draft provider/fallback wait | 551.62 s measured | ≤135 s by 3 × 45 s application attempts |

Integrity verification is unchanged for Moments. The Draft “after” value is deliberately labelled as a bound, not a fabricated measurement.

The gameplay audit is [gameplay-analysis-provider-audit.json](gameplay-analysis-provider-audit.json). Required follow-up is a fresh Analysis identity with working credentials, explicit Vision opt-in, semantic provider provenance and non-zero provider calls/tokens before evaluating selection quality.

## Tests

- Focused UI/identity/transactional/recovery/provider regression: 145 passed.
- Candidate workspace full file after transactional updates: 46 passed.
- Real-window QA: 3 DPI processes passed; 18 screenshots and three evidence files regenerated.
- Canonical Settings media: all 28 MP4 files decoded end-to-end with FFmpeg, 0 failures.
- AVAILABLE lineage: all four persisted hashes revalidated; manifest checksum matches Final.
- `git diff --check`: passed (only repository LF→CRLF checkout notices).
- Broader `tests/test_content_transformation.py` still has one pre-existing, out-of-scope failure in `test_local_generation_keeps_complete_boundary_when_required_facts_exceed_word_budget`; neither the failing production-plan mapping code nor that assertion was changed by this pass. The new provider-timeout regression passes.

## Reference control without an existing owner

No existing production owner was found for arbitrary caption vertical-position control. It was not implemented. Caption lanes and safe bounds continue to come from the existing caption/composition planner.
