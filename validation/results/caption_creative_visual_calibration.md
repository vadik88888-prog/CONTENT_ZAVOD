# Caption / creative preset real-media visual calibration

Date: 2026-08-14
Scope: validation artifacts only; no production, GUI, P1, selection, feasibility, Brain, Vision, or preset-definition changes.

## Outcome

The comparison pack is ready for visual Accept / Reject decisions. It uses one cached 27.59-second real RU excerpt (88 timed words, 3.1896 words/s) across every case, with dark studio, light animation/product footage, long words, 11 multiline cues, and three semantic-emphasis cues. All renders use the same fixed center crop so this pack compares caption styling, not composition quality.

Start with:

- `validation/artifacts/caption-creative-visual-calibration/comparison-grid.mp4`
- `validation/artifacts/caption-creative-visual-calibration/comparison-contact-sheet.png`
- `validation/artifacts/caption-creative-visual-calibration/comparison-semantic-emphasis-dark.png`
- `validation/artifacts/caption-creative-visual-calibration/comparison-semantic-emphasis-light.png`
- record decisions in `validation/artifacts/caption-creative-visual-calibration/decision-table.csv`

## Comparison matrix

| # | Scope | Creative preset | Caption preset | Exact rendered font | Visual style | Cues / multiline / emphasis | MP4 SHA-256 |
|---:|---|---|---|---|---|---:|---|
| 1 | creative default | `minimal` | `minimal_light` | `font.system.arial.normal.normal`, Arial Regular, `arial.ttf` | regular white, subtle outline/shadow | 17 / 11 / 3 | `3791213f7decbbd3e05a22450cb992554b2257bab4666bcf7ff68bd2311534c5` |
| 2 | creative default | `clean` | `clean_white` | `font.system.arial.normal.bold`, Arial Bold, `arialbd.ttf` | bold white, clean outline | 17 / 11 / 3 | `c8a73789e87669b38d64cbf5f153e1d1ca8a9468a390c4a03c194f2c36785a1e` |
| 3 | creative default | `documentary` | `editorial_narrow` | `font.system.arial.normal.bold`, Arial Bold, `arialbd.ttf` | editorial/documentary typography | 17 / 11 / 3 | `013f0c28be7ec036bbb94d365055a3ef3a5d8da68174ded12b273236cd4250e0` |
| 4 | creative default | `dynamic` | `accent_yellow` | `font.system.arial.normal.bold`, Arial Bold, `arialbd.ttf` | uppercase yellow semantic emphasis | 17 / 11 / 3 | `dbf0d85825dec593debf192a4ae86c476368a32586e9846c3963a63d23f87a05` |
| 5 | caption only | — | `karaoke_yellow` | `font.system.arial.normal.bold`, Arial Bold, `arialbd.ttf` | yellow karaoke semantic emphasis | 17 / 11 / 3 | `05f73191ae2c04c0d5831c52006fdc19c52ed7362c272544ec6b7e7754de0b7a` |
| 6 | caption only | — | `contrast_box` | `font.system.arial.normal.bold`, Arial Bold, `arialbd.ttf` | white text, dark translucent box | 17 / 11 / 3 | `e37391d504581c72de78cfb721e250b3221b599599f13b26522f9d93d0f84850` |

The caption-only cases use current registry definitions through a validation-time mapping; production defaults were not edited.

## Font and media evidence

- Regular file SHA-256: `b3658eadae55e682b5f69eb64c439c1ecc8f196c0bb8d4756d145d13bc86476a`; libass selected `(Arial, 400, 0) -> ArialMT`.
- Bold file SHA-256: `e8f4e3baf6cc35fed6fcce3a540e8b39e8f6cda1d22a28f2ec8f526fef7a43f5`; libass selected `(Arial, 700, 0) -> Arial-BoldMT`.
- Every case loaded only its checksummed file from its controlled `fontsdir`; no libass fallback was observed.
- All six decoded audio streams match SHA-256 `264ca6c0cc8dd738f4299971ef657a62fd4e00fb271a109ea85a3ceb6ae506eb`.
- Every case probes as H.264, 1080×1920, 30 fps, 27.600 s. The 3×2 grid is H.264/AAC, 1080×1280, 30 fps, 27.600 s; SHA-256 `969d19d97882a6715f461a23a2f95311ef71eae9d3558cf2fed75f582f1e37af`.
- Source window SHA-256: `bc4ee7a824ae4b91eac854c1f1f0a6e82cd4bb2031d34f1bc56442ad514c244a`. Cached transcript and source were reused; provider calls: 0.

Golos Text, Inter, and PT Sans Narrow Regular/Bold files are not available on this host. They were not downloaded, not rendered, and not substituted under their curated asset IDs. The explicit system Arial identities above are used only so the current preset styles remain directly comparable.

## Review notes

Visual inspection found no burned-in source-caption conflict or frame-edge clipping in the sampled dark/light and individual contact sheets. Yellow emphasis is visible on `ИНДУСТРИЯ ДОДУМАЛАСЬ` and `ПРИВЫЧКИ ЗАКЛАДЫВАЮТСЯ`; contrast-box behavior is visible on both background classes.

This is a visual calibration pack, not a publishability PASS. Current caption-plan reports remain `PASS_WITH_WARNINGS`: some cached word timings degrade safely to phrase/static timing, and dense speech produces CPS/readability warnings. No preset was changed by taste.

Machine evidence: `validation/artifacts/caption-creative-visual-calibration/calibration-report.json` (SHA-256 `6e318aa880ea45a14133e9a106d1e2fe502a60874661125284fe4333fdb96359`).
