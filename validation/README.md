# Production validation (Goal 3E)

This directory contains reproducible, local-only evidence for the full Content
Factory production pipeline. Generated videos, WAV files, reports, metrics and
temporary clips intentionally remain ignored by Git.

```text
validation/
  fixtures/   # licensed user media or synthetic technical fixtures
  reports/    # copied report.json evidence for completed runs
  artifacts/  # selected final MP4 / project manifests for inspection
  metrics/    # generated health JSON and Markdown
```

## Fixture policy

Use only media that the operator is permitted to process. The current repository
ships synthetic technical fixtures only; they are suitable for codec, cache,
timeline, crop, subtitle, audio, fallback and recovery checks, but **not** for
semantic quality claims about films, series, interviews, podcasts, lectures,
gameplay or news.

For complete production validation, place licensed examples in `fixtures/` and
record their category in the validation run manifest. Do not commit source media.

## Aggregating a completed run

```powershell
python validation\collect_health.py `
  --report .\output\<source>\report.json `
  --report .\output\<source-2>\report.json
```

The command writes `metrics/production-health.json` and
`metrics/production-health.md`. It reads only local reports; it never invokes
AI, TTS, Audio Composition or render stages.

## Bounded cache stress check

```powershell
python validation\run_stress.py `
  --input .\validation\fixtures\synthetic-proxy-interview.mp4 `
  --config .\validation\config.synthetic.yaml `
  --iterations 10
```

It starts a fresh CLI process for each isolated production-render-only cache
run and stores bounded stdout/stderr tails, runtime and Windows peak working
set memory in `metrics/stress.json`. It does not start a provider request.

## Synthetic source-format fixtures

```powershell
python validation\generate_variants.py --source .\input\smoke-test.mp4
```

It creates 720p/1080p/1440p/2160p, 24/25/30/50/60 FPS, vertical and square
technical fixtures under `fixtures/`. They are derived from one permitted
synthetic source and are intended for decoder/crop/timeline checks only.

To render the format matrix against an already validated plan and audio project:

```powershell
python validation\render_source_variants.py --config .\validation\config.synthetic.yaml `
  --plan .\output\synthetic-proxy-interview-...\production-plan.json `
  --audio-project .\output\synthetic-proxy-interview-...\audio\audio-project.json `
  --transcript .\work\synthetic-proxy-interview-...\transcript.json `
  --source .\validation\fixtures\synthetic-format-720p30.mp4
```

Repeat `--source` for every generated profile. The result is recorded in
`metrics/source-format-render.json`.

## Required evidence before GUI work

- `pytest` and `python -m app doctor` are green;
- each selected final MP4 passes ffprobe validation and a visual/audio review;
- render-only cache hits are recorded without upstream calls;
- CPU fallback and failure/recovery evidence are retained;
- category coverage is marked `synthetic` or `licensed-real` honestly.
