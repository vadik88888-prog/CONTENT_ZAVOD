# Preset/font real-MP4 acceptance — `de4d605`

Date: 2026-08-13

Foundation: `de4d605c84b9152b60389afaa88a8cbacd6e342a`

Result: **PASS for locally available exact system faces; curated Golos Text / Inter / PT Sans Narrow remain unverified on real render because their files are not installed or bundled on this host.**

## Scope and source

- Cached real source only: `validation/artifacts/goal7i/food/fallback-source-window.mp4`
- Source SHA-256: `0f9f2045baf68e31528b411768b809c2601e321610445b95e6e70dafb4cb7f39`
- Source probe: H.264 960×540 at 25 fps + AAC stereo 48 kHz; 18.56 s. Every acceptance render uses its first 8 s.
- Five Preview/Final pairs cover RU and EN captions, all four creative preset IDs, four versioned caption preset tokens, two font families, and exact Regular/Bold faces.
- Provider calls: `0`; Brain rerun: `false`; Vision rerun: `false`; same-source extra shots: OFF.
- EN text is a typography/shaping acceptance script over the same cached RU real footage. This report does not claim EN semantic-content acceptance.

## Result matrix

| Case | Creative / caption preset | Exact font identity selected by libass | Preview SHA-256 | Final SHA-256 | Parity / SSIM |
|---|---|---|---|---|---|
| `ru-minimal-arial-regular` | `minimal:1.0.0` / `caption-preset:minimal_light:1.0.0` | `font.system.arial.normal.normal`; `arial.ttf`; `b3658eadae55e682b5f69eb64c439c1ecc8f196c0bb8d4756d145d13bc86476a`; `(Arial, 400, 0) -> ArialMT` | `b1d948479a6e7a375d77d729fd599c453bac4d704f2e2a8b4af99089b7c2f716` | `bbee58a70aa8e91dfa2b81b226e6c6e557e52e67beae5cedba910141cee7afce` | matched / `0.996547` |
| `ru-dynamic-arial-bold` | `dynamic:1.0.0` / `caption-preset:accent_yellow:1.0.0` | `font.system.arial.normal.bold`; `arialbd.ttf`; `e8f4e3baf6cc35fed6fcce3a540e8b39e8f6cda1d22a28f2ec8f526fef7a43f5`; `(Arial, 700, 0) -> Arial-BoldMT` | `bdc3dca7e3410b0ec42cb7d9bdc0b3c8cdc6084f21bbf5f8e22d7c72b7b5e1ce` | `c07990f38a9d44d32667c2f7be8b19f6719a96df4a551aa966dc3a18875abe0d` | matched / `0.996406` |
| `ru-documentary-segoe-bold` | `documentary:1.0.0` / `caption-preset:editorial_narrow:1.0.0` | `font.system.segoe-ui.normal.bold`; `segoeuib.ttf`; `aeb9e4a6ec5cc59f4d72df8189032d7dbb28f45161cf1552174818b5465dac4e`; `(Segoe UI, 700, 0) -> SegoeUI-Bold` | `ee450e845d4728e0ede34daf30d2e349db6117c7e470186e1f5d11af51036e5c` | `41f99e8cf8b7a0afc40ae5fc53f0e32cd66c07aa7bb24dd9c8c19178e81d1671` | matched / `0.997185` |
| `en-clean-segoe-bold` | `clean:1.0.0` / `caption-preset:clean_white:1.0.0` | `font.system.segoe-ui.normal.bold`; `segoeuib.ttf`; `aeb9e4a6ec5cc59f4d72df8189032d7dbb28f45161cf1552174818b5465dac4e`; `(Segoe UI, 700, 0) -> SegoeUI-Bold` | `f39ccd7dd49504129bb5855526c2b3cf35a6d362da7c79366a337096f6d71c34` | `5e939de60e511d7cf844fa95dcb62b01f3fdfae11b354dbc72541c01b287ef58` | matched / `0.996949` |
| `en-minimal-segoe-regular` | `minimal:1.0.0` / `caption-preset:minimal_light:1.0.0` | `font.system.segoe-ui.normal.normal`; `segoeui.ttf`; `8134dbcd09e7b123c9a7f229d49cffbcb01352cc72ea5e1076b65d0dca9f73cd`; `(Segoe UI, 400, 0) -> SegoeUI` | `b35fe5e3b53317294361b7b6fbe4081fc313e193dc7cd2d500c37096b445c38a` | `fb014f505026c1bdd19802b88810bf4a4ecb97ef2933ea401b0d725b33ce08f2` | matched / `0.996791` |

SSIM compares each Preview with its Final downscaled to 540×960; it is supporting visual evidence, not the semantic parity decision. Semantic/timing/layout parity is `matched` from the two emitted `RenderParityManifest` objects built from one exact compiled plan per case.

## Font and audio evidence

- Preview and Final each contain a renderer-controlled `fonts/` directory with exactly one staged face named by stable asset ID plus checksum prefix. The staged SHA-256 equals the manifest SHA-256 in every profile.
- FFmpeg logs explicitly contain `:fontsdir=`, `Loading font file`, and the expected Regular/Bold `fontselect` result shown above.
- All ten MP4s pass full video+audio decode with `-xerror`; each contains H.264 video at 30 fps and AAC stereo at 48 kHz.
- Decoded PCM SHA-256 is identical for Preview and Final in every pair: `859ba848be1efb3fc87d96435cff0061b5d45d2f1d3dcd2c1522ad5a25b0d6d4`.

## Visual review

- RU: Cyrillic glyphs render without tofu or missing characters; no clipping or unsafe line overflow observed.
- EN: glyph shaping and line breaks are readable; no clipping observed.
- Regular is visibly lighter than Bold for both Arial and Segoe UI.
- Side-by-side normalized Preview/Final frames show the same caption text, line breaks, relative geometry, face weight, and treatment. Resolution/bitrate are the visible profile differences.

Visual evidence:

- `validation/artifacts/preset-font-real-mp4-de4d605/comparison-contact-sheet.png` — SHA-256 `fd6d9bf0bfec419ebebcba77d59600a59fc9d1470aa01190c25c8569de362583`
- `validation/artifacts/preset-font-real-mp4-de4d605/<case>/creative-preview-contact-sheet.png`
- `validation/artifacts/preset-font-real-mp4-de4d605/<case>/final-contact-sheet.png`
- `validation/artifacts/preset-font-real-mp4-de4d605/<case>/preview-final-paired-frame.png`

## Exact artifacts

Root: `validation/artifacts/preset-font-real-mp4-de4d605/`

For every case listed in the matrix:

- Preview: `<case>/creative_preview/creative-preview.mp4`
- Final: `<case>/final/final-short.mp4`
- Shared identity: `<case>/compiled-render-plan.json`
- Profile parity: `<case>/creative_preview/parity-manifest.json` and `<case>/final/parity-manifest.json`
- Exact ASS/font evidence: `<profile>/captions.ass`, `<profile>/fonts/`, and `<profile>/ffmpeg.log`
- Audio/visual comparison: `<case>/preview-final-ssim.log` and the three PNG sheets listed above.

Machine-readable report: `validation/artifacts/preset-font-real-mp4-de4d605/acceptance-report.json` — SHA-256 `a22d1cb1ce3a0cc219c0c8c58eeb77b20257c43ce4154b6c3042cea30be7dea0`.

Harness: `validation/preset_font_real_mp4_acceptance.py` — SHA-256 `9384fa271a1e1af9a6cd71a538c2200a56f46454172370d982699b82a02258e5`. FFmpeg 8.0.1 executable SHA-256: `5af82a0d4fe2b9eae211b967332ea97edfc51c6b328ca35b827e73eac560dc0d`.

## Remaining warning

The friend-beta registry names Golos Text, Inter, and PT Sans Narrow, but this checkout/host has no corresponding font files. Their registry metadata and fallback behavior are covered by focused tests, but their exact curated bytes and libass face selection are **not** real-render accepted here. No font was downloaded or substituted under a curated asset ID.
