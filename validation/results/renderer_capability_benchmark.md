# Goal 7B — Creative Renderer Capability Benchmark

Generated: `2026-08-09T10:32:18Z`

## Decision

**NO_GO_TIER_2_KEEP_TIER_1_LIBASS** — Tier 1 libass; retain the RGBA architecture seam.

This is a benchmark/decision artifact only. No production renderer enum, visual style, Phase 7C planner or UI was changed.

## Controlled contract

- Fixture duration: 3.0 s; one unmeasured warm-up + 3 measured repeats per cell.
- Same fixed RU/EN/mixed text, 30 fps, normalized placement, source, duration, output resolution and encoder target per backend pair.
- Sources: synthetic deterministic H.264 1080p, H.264 4K and AV1 4K. Generated media stays ignored under validation/artifacts.
- CPU is system utilization during the run; RSS is benchmark process plus direct FFmpeg child; NVIDIA values are global utilization and baseline-relative VRAM.
- Short clips emphasize startup overhead; startup is also reported separately. Measurements are local-machine evidence, not a fleet release promise.

## Decision gates

| Gate | Result |
|---|---|
| `all_launch_primitives` | PASS |
| `deterministic_frame_hashes` | PASS |
| `ru_en_mixed_shaping` | PASS |
| `real_per_glyph_bounds` | PASS |
| `preview_final_geometry_parity` | FAIL |
| `font_identity_and_checksum` | PASS |
| `preview_gpu_rtf_initial_budget` | FAIL |
| `preview_cpu_fallback_rtf_budget` | PASS |
| `final_gpu_rtf_initial_budget` | FAIL |
| `final_cpu_fallback_rtf_budget` | PASS |
| `preview_ram_budget` | PASS |
| `final_ram_budget` | PASS |
| `preview_vram_budget` | PASS |
| `final_vram_budget` | PASS |
| `gpu_telemetry_collected` | PASS |
| `cold_startup_budget` | PASS |
| `all_1080_4k_av1_matrix_cells_completed` | PASS |
| `safe_fallbacks_complete` | PASS |
| `no_new_runtime_dependency` | PASS |
| `no_arbitrary_execution_surface` | PASS |

### Decision interpretation

- Tier 1 libass passed the launch primitives, deterministic hashes and parity gate. Its worst measured GPU Preview RTF was 0.916, so retaining it is a correctness/safe-baseline decision—not a claim that the aspirational 0.5 Preview RTF was met on every stress source.
- Qt RGBA reached worst GPU Preview/Final RTF 1.109/1.596; it also exceeded normalized geometry tolerance. Capability alone therefore does not qualify Tier 2.
- Both CPU fallback profiles stayed inside the pre-registered initial RTF budgets; RAM/VRAM and startup gates also passed.

## Capability

| Backend | Fade | Scale | Slide | Karaoke | Deterministic | Glyph bounds |
|---|---|---|---|---|---|---|
| libass | PASS | PASS | PASS | PASS | PASS | rendered block only |
| qt_rgba | PASS | PASS | PASS | PASS | PASS | per-glyph |

### RU/EN shaping and real bounds

- `ru`: Qt shaped 38 glyphs; 31 non-empty per-glyph ink bounds; Qt observed block `[69, 942, 1012, 985]`; libass observed block `[161, 899, 922, 1032]`.
- `en`: Qt shaped 33 glyphs; 27 non-empty per-glyph ink bounds; Qt observed block `[71, 938, 1006, 993]`; libass observed block `[14, 935, 1063, 996]`.
- `mixed`: Qt shaped 35 glyphs; 28 non-empty per-glyph ink bounds; Qt observed block `[73, 941, 1007, 985]`; libass observed block `[226, 899, 854, 1020]`.

## Preview / Final parity

| Backend | Preview normalized bounds | Final normalized bounds | Max delta | Gate |
|---|---|---|---:|---|
| libass | `[0.20925926, 0.46875, 0.79074074, 0.53125]` | `[0.20925926, 0.46822917, 0.79074074, 0.53125]` | 0.00052083 | PASS |
| qt_rgba | `[0.07222222, 0.490625, 0.92592593, 0.5125]` | `[0.06759259, 0.49010417, 0.93240741, 0.51302083]` | 0.00648148 | FAIL |

## Performance matrix

Median values after warm-up.

| Source | Backend | Profile | RTF | Wall s | CPU % | GPU % | Peak RSS MB | Peak VRAM Δ MB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| h264_1080 | libass | preview_cpu | 0.535 | 1.606 | 26.38 | 0.00 | 204.27 | 0.00 |
| h264_1080 | qt_rgba | preview_cpu | 0.706 | 2.117 | 42.29 | 0.00 | 264.66 | 0.00 |
| h264_1080 | libass | preview_gpu | 0.802 | 2.406 | 24.13 | 0.00 | 310.80 | 8.00 |
| h264_1080 | qt_rgba | preview_gpu | 0.990 | 2.970 | 36.99 | 5.00 | 397.59 | 123.00 |
| h264_1080 | libass | final_cpu | 0.826 | 2.478 | 44.65 | 0.00 | 844.26 | 0.00 |
| h264_1080 | qt_rgba | final_cpu | 1.408 | 4.224 | 57.88 | 0.00 | 1155.68 | 0.00 |
| h264_1080 | libass | final_gpu | 0.893 | 2.680 | 23.75 | 25.00 | 478.36 | 216.00 |
| h264_1080 | qt_rgba | final_gpu | 1.489 | 4.468 | 47.83 | 6.40 | 750.96 | 216.00 |
| h264_4k | libass | preview_cpu | 0.652 | 1.956 | 39.26 | 0.00 | 478.92 | 0.00 |
| h264_4k | qt_rgba | preview_cpu | 0.838 | 2.515 | 41.98 | 0.00 | 534.90 | 0.00 |
| h264_4k | libass | preview_gpu | 0.899 | 2.696 | 32.07 | 5.00 | 585.63 | 123.00 |
| h264_4k | qt_rgba | preview_gpu | 1.109 | 3.328 | 40.58 | 3.33 | 639.68 | 123.00 |
| h264_4k | libass | final_cpu | 0.935 | 2.806 | 47.44 | 0.00 | 1110.87 | 0.00 |
| h264_4k | qt_rgba | final_cpu | 1.532 | 4.595 | 67.46 | 0.00 | 1435.32 | 0.00 |
| h264_4k | libass | final_gpu | 0.943 | 2.830 | 32.63 | 13.50 | 744.86 | 216.00 |
| h264_4k | qt_rgba | final_gpu | 1.594 | 4.781 | 51.29 | 5.17 | 1014.59 | 216.00 |
| av1_4k | libass | preview_cpu | 0.651 | 1.952 | 36.83 | 0.00 | 399.75 | 0.00 |
| av1_4k | qt_rgba | preview_cpu | 0.828 | 2.483 | 42.23 | 0.00 | 472.84 | 0.00 |
| av1_4k | libass | preview_gpu | 0.916 | 2.749 | 28.92 | 5.00 | 501.30 | 123.00 |
| av1_4k | qt_rgba | preview_gpu | 1.107 | 3.321 | 38.59 | 3.00 | 572.24 | 123.00 |
| av1_4k | libass | final_cpu | 0.993 | 2.980 | 52.34 | 0.00 | 1038.10 | 0.00 |
| av1_4k | qt_rgba | final_cpu | 1.596 | 4.789 | 66.63 | 0.00 | 1262.18 | 0.00 |
| av1_4k | libass | final_gpu | 0.954 | 2.861 | 33.75 | 14.50 | 664.25 | 216.00 |
| av1_4k | qt_rgba | final_gpu | 1.596 | 4.787 | 51.55 | 4.50 | 914.50 | 216.00 |

## Startup / cache

- libass: cold process wall 1.398s; warm process median 1.345s across 3 repeats.
- qt_rgba: cold process wall 0.303s; warm process median 0.287s across 3 repeats.

## Double-encode bottleneck

On `h264_4k` → `final_cpu`, single-pass median was 2.778s (RTF 0.926); the current prepare+final equivalent was 4.569s (RTF 1.523), **64.5% extra wall time**. Reusing a prepared visual for caption-only final would cost 2.302s.

Code inspection agrees with the measurement: `_prepare_visual_clip` encodes H.264 and `_mux_final` decodes/encodes it again; the full-render cache key contains subtitles, while prepared clips are not independent cache nodes.

## Capability registry and safe fallbacks

Selected backend: `libass`; candidate status: `benchmark_only_unqualified`.

| Requested | Effective | Mapping | Degraded | Fallback |
|---|---|---|---|---|
| static | static | static | False | static |
| fade | fade | fad | False | static |
| scale | scale | bounded_transform | False | fade |
| slide | slide | bounded_move | False | fade |
| karaoke | karaoke | karaoke_fill | False | static |
| per_glyph_motion | karaoke | karaoke_fill | True | karaoke |
| masked_highlight | karaoke | karaoke_fill | True | karaoke |
| coordinated_layers | fade | fad | True | fade |

## Limitations

- GPU telemetry is device-global and may include unrelated desktop activity; exact samples are retained in JSON.
- The candidate is Qt raster RGBA piped into FFmpeg, not a production integration. The decision proves or rejects this exact graph only.
- SSIM/VMAF is not used to rank typography backends because their pixels intentionally differ; correctness uses shaping, real bounds, frame hashes, primitive behavior and parity invariants.
- CPU-only fallback was measured locally, not on a separate deployment machine; fleet qualification remains a later rollout gate.

## Reproduction

```powershell
.\.venv\Scripts\python.exe validation\renderer_capability_benchmark.py
```
