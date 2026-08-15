# Friend Beta Desktop redesign — QA evidence

## Root causes and integration map

The UI previously exposed the engine as one long technical form. Friend Beta now projects the same persisted project through six focused routes: Source → Settings → Processing → Moments → Drafts → Final. No replacement analysis, creative, render, or artifact subsystem was added.

The proven Moments delay came from `ProjectScreen._analysis_artifact()` verifying Analysis before consulting its UI transaction cache. Setup estimate and repeated review consumers therefore reopened and reverified the same artifact. The fixed path performs one verified load, stores a signature-keyed projection, and incrementally adds 12 cards at a time from that projection. Artifact/reference stat changes invalidate the projection; verification policy is unchanged.

The AVAILABLE end-to-end run found one further integration defect: Draft correctly applied a candidate `dynamic` override, while selected-render established its defensive constraint from the project default `documentary`. The renderer rejected the valid immutable plan with `PRESET_CONSTRAINT_VIOLATION`. Selected-render now resolves the same candidate-scoped option overlay as Draft. The pre-fix and post-fix run identities are retained in `available-to-final-e2e.json`.

This UI-only follow-up starts from commit `608c0a38` and keeps that integration intact. Settings now projects the existing profile/style/caption owners as visual choices instead of a dropdown form. Its seven caption cards use the exact bundled production font families and typography; the selected preset is demonstrated in a local 9:16 widget, with UI-only motion for Active/Karaoke and Word Pop. That sample does not open media or invoke FFmpeg, Brain, Vision, Draft, or render. Moments no longer exposes transcript/scoring dumps. Drafts keeps the exact persisted Creative Preview in a compact list + 9:16 preview + candidate-scoped inspector. Final binds the exact persisted `ClipResult` and `QualityReport` to a real player and human-readable summary.

| Capability | Existing owner reused by the desktop flow |
|---|---|
| Auto + 15 content profiles | `app/content_profile_taxonomy.py`, `app/product_flow.py` |
| “Что искать?” / editorial intent | `app/product_flow.py`, `app/editorial_profile_policy.py` |
| Processing/deep-analysis controls | `ProjectOptions` → `ProcessingIntent` → `ResolvedProcessingConfig` |
| RECOMMENDED / AVAILABLE / BLOCKED | persisted Analysis candidate recommendation + eligibility decisions |
| Selectable moments and boundaries | existing candidate selection and `candidate_boundary_overrides` |
| Four creative families | `app/creative_policy.py` |
| Seven caption presets | `app/caption_presets.py`, existing `CreativePolicy.user_override_ids` slot |
| Bundled fonts | `app/font_assets.py`, `assets/fonts/` |
| Real Creative Preview | `DraftArtifact.preview` + creative parity identity |
| Crop/reframe and same-source extra shots | existing production composition/source-B-roll config; explicit opt-in remains false |
| Final outputs | canonical `ClipResult` / run manifest projection |
| Retry, recovery, partial success | existing candidate lifecycle, `DesktopServices`, run report recovery |

## Changed target files

UI-only delta from `608c0a38`:

- `app/gui/components/final_results.py`
- `app/gui/components/video_preview.py`
- `app/gui/screens/project_screen.py`
- `app/gui/styles/theme.qss`

Verification:

- `tests/test_friend_beta_desktop_integration.py`
- `tests/test_candidate_workspace.py`
- `tests/test_final_results_workspace.py`
- `validation/friend_beta_desktop_real_window_qa.py`
- this evidence directory

## Real-window screenshots

Every screenshot is a native shown Windows `MainWindow`, captured through its real `winId`. The QA transaction uses exact persisted Analysis, DraftArtifact, Creative Preview, ClipResult, and Final identities; only desktop metadata is copied to an isolated temporary data directory.

| Screen | 100% (1920×950 logical) | 125% (1536×760 logical) | 150% (1280×630 logical) |
|---|---|---|---|
| Source | [PNG](source-dpi100.png) | [PNG](source-dpi125.png) | [PNG](source-dpi150.png) |
| Settings | [PNG](settings-dpi100.png) | [PNG](settings-dpi125.png) | [PNG](settings-dpi150.png) |
| Processing | [PNG](processing-dpi100.png) | [PNG](processing-dpi125.png) | [PNG](processing-dpi150.png) |
| Moments | [PNG](moments-dpi100.png) | [PNG](moments-dpi125.png) | [PNG](moments-dpi150.png) |
| Drafts | [PNG](drafts-dpi100.png) | [PNG](drafts-dpi125.png) | [PNG](drafts-dpi150.png) |
| Final | [PNG](final-dpi100.png) | [PNG](final-dpi125.png) | [PNG](final-dpi150.png) |

Machine-readable metrics: [100%](runtime-evidence-dpi100.json), [125%](runtime-evidence-dpi125.json), [150%](runtime-evidence-dpi150.json). All 18 states have zero horizontal scroll and no clipped primary CTA. Source, Settings, Moments, Drafts, and Final expose exactly one primary CTA; Processing intentionally exposes none while work is running. Drafts preserves the normal three-column composition at all three desktop profiles and uses the outer workspace as its sole vertical scroll owner. Moments and Final retain bounded list/inspector panes rather than becoming a giant vertical stack.

## Persisted artifact lineage

The screenshot transaction resolves:

`project aa71… → Analysis analysis-33ea… → Draft draft-ab56… → candidate-chapter-008-story-004 → Creative Preview SHA 3fe2… → ClipResult candidate…:production… → run cb2d… → Final SHA cfa347…`

The independent AVAILABLE proof is [available-to-final-e2e.json](available-to-final-e2e.json):

- candidate `candidate-chapter-009-story-001`: `recommended=false`, `eligible=true`;
- source interval `521.04–542.17`;
- candidate-scoped Dynamic + `caption-preset:word_pop:2.1.0` + fit-background;
- same-source extra shots false, zero B-roll segments;
- Draft `draft-ffdb85dbf7e1b077`, real creative preview with parity signature;
- Final run `881b54b7db2c4cb598ffaaf6a5d1f7e3`, canonical 1080×1920 output, 10,828,011 bytes;
- final SHA-256 `28c48f298b37aeeb73bd9807b5505b82f400dbd9b6923242217d448a527555f0`, identical to the manifest artifact checksum;
- zero Brain/Vision/analysis stages in both Draft and Final runs.

## Performance before / after

See [performance-before-after.json](performance-before-after.json).

| Initial Moments transaction | Before | After |
|---|---:|---:|
| Verified Analysis loads | 5 | 1 |
| Wall time | 2.1803 s | 0.0357 s |
| Initial cards | 1 | 12 |

## Tests

- UI/candidate/identity/recovery/caption/quality focused regression: 111 passed.
- Real-window QA: 3 DPI processes passed; 18 screenshots and three runtime evidence files produced.
- AVAILABLE → Draft → Final: persisted proof revalidated against its exact existing artifacts; the UI-only delta does not touch its owners.
- `compileall`: passed.
- `git diff --check`: passed (Git reports only the repository's existing LF→CRLF checkout warnings).

## Reference control with no existing owner

No existing production owner was found for arbitrary caption vertical-position control. It was not implemented. Caption safe lanes/bounds continue to be owned by the current caption/composition planner.
