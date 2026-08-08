# Goal 6B.1 — Vision Budget Calibration

## Outcome

Dry-run calibration passed on two real local sources. No production render and no external provider call was made; test spend was `$0.00`. The canonical machine-readable result, including every keyframe ID, timestamp, coverage ID, projected reservation and cache counter, is `vision_budget_calibration.json`.

The fixed 12/32 frame caps were too small for long sources and their original supporting call/token caps could reserve only 9/24 low-detail frames in the worst case. They now remain short-source defaults. A bounded duration/content multiplier raises the effective frame ceiling to at most 2x; supporting call/token ceilings are derived from that frame plan, while the configured hard dollar budgets remain unchanged at `$0.05` / `$0.15`.

## Sources and found keyframes

| Case | Real source | Duration | Local evidence | Found keyframes |
| --- | --- | ---: | ---: | ---: |
| gameplay | PUBG first-person gameplay, WebM | `1259.421 s` (20:59) | 72 scenes / 47 StoryUnits | 64 |
| podcast_interview | long-form business interview, AV1 4K WebM | `9821.821 s` (2:43:41) | 989 scenes / 255 StoryUnits | 64 |

Gameplay found timestamps (seconds): `30.233, 31.317, 34.200, 90.100, 98.667, 132.890, 145.100, 191.500, 211.100, 213.267, 215.350, 221.500, 231.450, 236.683, 245.217, 247.783, 266.450, 300.400, 339.017, 353.460, 399.233, 401.717, 404.900, 408.367, 409.367, 440.100, 460.367, 462.317, 465.633, 466.517, 468.650, 470.850, 475.767, 484.167, 485.017, 502.650, 504.600, 526.600, 528.317, 531.050, 561.567, 586.867, 648.717, 664.350, 677.833, 681.167, 702.783, 705.117, 706.867, 769.667, 867.933, 871.250, 875.367, 903.567, 1022.800, 1034.383, 1111.083, 1132.667, 1136.883, 1137.700, 1148.267, 1187.817, 1243.617, 1246.540`.

Podcast/interview found timestamps (seconds): `378.640, 385.000, 702.080, 715.400, 893.960, 925.200, 1516.000, 1643.480, 1647.400, 1655.280, 1659.920, 1663.160, 1687.240, 1739.280, 1768.000, 1780.000, 1799.480, 1860.480, 2240.320, 2251.560, 2314.480, 2351.160, 2465.960, 2530.440, 2704.440, 2706.560, 2708.560, 2726.120, 2989.480, 3007.880, 3010.200, 3058.000, 3059.720, 3062.440, 3070.480, 3072.480, 3074.440, 3092.720, 3190.560, 3223.040, 4056.200, 4668.520, 4818.600, 6686.840, 7111.320, 7123.320, 7265.320, 7405.600, 7565.200, 7711.040, 7788.080, 7852.560, 7956.080, 7984.560, 8076.080, 8108.080, 8221.040, 8359.120, 9253.960, 9256.480, 9625.200, 9660.920, 9767.440, 9808.440`.

## Budget, coverage and cost

Projected tokens/cost are conservative pre-I/O reservations, not measured fake-provider usage.

| Case / mode | Configured → effective ceiling | Selected | Scene coverage | StoryUnit coverage | Temporal coverage | Projected calls / tokens / cost | Hard USD cap | Stop reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| gameplay / Fast | 0 → 0 | 0 | 0 / 72 | 0 / 47 | none | 0 / 0 / `$0.00000` | `$0.00` | `fast_mode_zero_calls` |
| gameplay / Standard | 12 → 20 | 20 | 20 / 72 | 15 / 47 | all 10 deciles | 7 / 27,700 / `$0.036325` | `$0.05` | `duration_content_ceiling_reached` |
| gameplay / Maximum | 32 → 51 | 39 | 35 / 72 | 25 / 47 | all 10 deciles | 13 / 52,000 / `$0.06760` | `$0.15` | `minimum_temporal_spacing_reached` |
| podcast / Fast | 0 → 0 | 0 | 0 / 989 | 0 / 255 | none | 0 / 0 / `$0.00000` | `$0.00` | `fast_mode_zero_calls` |
| podcast / Standard | 12 → 24 | 24 | 24 / 989 | 24 / 255 | 9 / 10 deciles | 8 / 32,000 / `$0.04160` | `$0.05` | `duration_content_ceiling_reached` |
| podcast / Maximum | 32 → 64 | 36 | 36 / 989 | 35 / 255 | 9 / 10 deciles | 12 / 48,000 / `$0.06240` | `$0.15` | `minimum_temporal_spacing_reached` |

The podcast gap in temporal decile 5 is already present in the 64 found keyframes (there are no candidates between `4818.600` and `6686.840`). It is therefore an upstream sparse-timeline limitation, not budget truncation. Candidate generation/ranking was intentionally left unchanged.

## Selected keyframes

- Gameplay / Standard (20): `31.317, 98.667, 215.350, 236.683, 266.450, 401.717, 462.317, 485.017, 502.650, 528.317, 586.867, 664.350, 705.117, 769.667, 875.367, 903.567, 1034.383, 1111.083, 1148.267, 1243.617`.
- Gameplay / Maximum (39): `31.317, 90.100, 98.667, 132.890, 145.100, 191.500, 215.350, 236.683, 247.783, 266.450, 300.400, 339.017, 353.460, 401.717, 409.367, 440.100, 462.317, 468.650, 475.767, 485.017, 502.650, 528.317, 561.567, 586.867, 648.717, 664.350, 677.833, 705.117, 769.667, 867.933, 875.367, 903.567, 1022.800, 1034.383, 1111.083, 1137.700, 1148.267, 1187.817, 1243.617`.
- Podcast / Standard (24): `378.640, 702.080, 893.960, 1516.000, 1643.480, 1799.480, 2314.480, 2530.440, 2704.440, 3007.880, 4056.200, 4818.600, 6686.840, 7123.320, 7265.320, 7405.600, 7565.200, 7711.040, 7852.560, 7956.080, 8108.080, 9253.960, 9660.920, 9808.440`.
- Podcast / Maximum (36): `378.640, 702.080, 893.960, 1516.000, 1643.480, 1687.240, 1739.280, 1799.480, 1860.480, 2251.560, 2314.480, 2465.960, 2530.440, 2704.440, 3007.880, 3070.480, 3223.040, 4056.200, 4668.520, 4818.600, 6686.840, 7123.320, 7265.320, 7405.600, 7565.200, 7711.040, 7788.080, 7852.560, 7956.080, 8108.080, 8221.040, 8359.120, 9253.960, 9660.920, 9767.440, 9808.440`.

Fast selects no keyframes and makes zero calls in both sources.

## Fixed-cap comparison

| Case / mode | Fixed-cap selected / scenes / StoryUnits | Calibrated selected / scenes / StoryUnits |
| --- | ---: | ---: |
| gameplay / Standard | 12 / 12 / 10 | 20 / 20 / 15 |
| gameplay / Maximum | 32 / 30 / 22 | 39 / 35 / 25 |
| podcast / Standard | 12 / 12 / 12 | 24 / 24 / 24 |
| podcast / Maximum | 29 / 29 / 28 | 36 / 36 / 35 |

## Cache behavior

The first fake-provider dry-run produced one miss per selected frame and the projected batch count. Repeating the identical run produced one hit per selected frame and zero new calls: gameplay Standard `20/0`, gameplay Maximum `39/0`, podcast Standard `24/0`, and podcast Maximum `36/0` (hits/new calls). Fast remained `0/0` on both runs.

The dry provider and deterministic frame loader exist only in `validation/vision_budget_calibration.py`; production provider selection, candidate generation, scoring, ranking, UI, and rendering were not changed.

## Verification

- Full `pytest -q`: exit `0`, reached 100%; platform-dependent tests skipped normally.
- Focused `pytest tests/test_vision_intelligence.py -q`: `9 passed`.
- Mypy for both changed executable modules with imports isolated: `Success: no issues found in 2 source files`.
- `python -m app doctor --config config.yaml.yaml`: 0 errors, 0 warnings.
- `python -m build`: sdist and wheel built successfully.
- Full repository mypy is not configured as a green gate and currently reports 457 pre-existing errors across 41 unrelated legacy/GUI/provider files. No global ignores or out-of-scope type rewrites were introduced.
