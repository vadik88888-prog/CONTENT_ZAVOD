# Audio Evidence v1 — Fresh Gameplay validation

This is a deterministic replay over the saved Fresh Gameplay PCM/transcript/Vision artifacts. It made zero AI and zero Vision provider calls.

## Rejected 746.30–782.59 candidate

Before: `candidate-chapter-034-story-001`, 36.29s, score 49, `RECOMMENDED`, selected=true.

After: score 49, local quality 30.177, `AVAILABLE` (still selectable, but no longer recommended).

Speech: longest internal gap 29.37s. Audio: activity 0.058, dead-zone 0.832; no meaningful gameplay audio event in the range. Vision PASS 1 reported movement but no reaction/payoff. The code-owned `SPARSE_MULTIMODAL_CONTENT` penalty is soft and does not create `BLOCKED`.

## RECOMMENDED after replay

- `candidate-chapter-021-story-001` — 461.53–481.82 (20.29s), profile `gameplay`, score 52. Text: Ну, не знаю, да, игрывать как-то можно. А, там есть, да. А этого что, мне убивать надо, что ли? Мой бинок какой-нибудь, да, мне знаю. Залетает он? 17 секунд. Почему? Потому что там 9 секунд.
  Audio: activity=0.291, meaningful_events=0; Visual: movement. Reason: tight self-contained candidate retained by existing Brain/virality/editorial owners; no sparse multimodal penalty.

## Audio seeds

Signal peak regions: 12; bounded semantic regions: 15; candidate seeds resolved by the existing SemanticBoundaryEngine: 10.

## Performance

One signal pass took 4.350s. Bounded ONNX took 1.088s over 180.0s / 1259.4s (15 regions). Full-source semantic scan=false, source video decodes=0.

The complete machine-readable evidence, including peak/semantic regions and events, is in `docs/audits/AUDIO_EVIDENCE_V1_GAMEPLAY_VALIDATION.json`.
