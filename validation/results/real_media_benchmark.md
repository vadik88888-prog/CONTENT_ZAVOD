# Goal 6E — Real Media Benchmark & Calibration

Generated: `2026-08-08T16:08:43Z`

## Benchmark contract

OLD: `5B.1-pre-6D@f70ba0e^`. NEW: `6D.1`. Both use identical persisted transcript/audio/scene evidence, duration constraints, score threshold, diversity/coverage selection and three-clip limit. LLM reranking and rendering are excluded. Media is local user-supplied evaluation material and is not committed or redistributed.

## OLD vs NEW

| Source | Type | OLD candidates | NEW candidates | OLD selected | NEW selected | Vision calls | Tokens | Cost USD | Cache |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Большинство бизнесов провалит ИИ-трансформацию — Андрей Дороничев | podcast / talking head | 255 | 255 | 3 | 3 | 8 | 19698 | 0.025777 | confirmed |
| Еда, которая нас убивает | interview | 56 | 59 | 3 | 3 | 9 | 23224 | 0.031839 | confirmed |
| Пробую пельмени Зубарева в Китае | vlog / food / travel | 68 | 68 | 3 | 3 | 6 | 17432 | 0.024667 | confirmed |
| PUBG local gameplay source | visual-heavy / gameplay | 47 | 47 | 3 | 3 | 7 | 20585 | 0.029268 | confirmed |

Aggregate real Vision usage: **30 calls, 88 frames sent, 80939 tokens, $0.1115515 estimated**. Cold benchmark wall time: **1038.8s**. Every warm repeat used 0 calls and sent 0 frames.

## Processing time

| Source | OLD generation+score+selection | NEW local timeline+generation+2 scores+selection | Vision cold | Vision warm repeat |
|---|---:|---:|---:|---:|
| podcast_talking_head | 24.094s | 29.034s | 210.179s | 34.341s |
| interview | 0.915s | 1.910s | 247.041s | 33.504s |
| food_vlog | 0.503s | 1.013s | 182.301s | 22.346s |
| gameplay | 0.158s | 0.531s | 235.505s | 28.306s |

## Rejected candidates

| Source | OLD sampled reasons | NEW sampled reasons |
|---|---|---|
| podcast_talking_head | DURATION_OUT_OF_RANGE, VISUAL_EVIDENCE_UNAVAILABLE | DURATION_OUT_OF_RANGE, SEMANTIC_INCOMPLETE, SENTENCE_BOUNDARY_UNRECOVERABLE, VISUAL_EVIDENCE_UNAVAILABLE |
| interview | DURATION_OUT_OF_RANGE, SEMANTIC_INCOMPLETE, SENTENCE_BOUNDARY_UNRECOVERABLE, VISUAL_EVIDENCE_UNAVAILABLE | DURATION_OUT_OF_RANGE, SEMANTIC_INCOMPLETE, SENTENCE_BOUNDARY_UNRECOVERABLE |
| food_vlog | DURATION_OUT_OF_RANGE, VISUAL_EVIDENCE_UNAVAILABLE | DURATION_OUT_OF_RANGE, SEMANTIC_INCOMPLETE, SENTENCE_BOUNDARY_UNRECOVERABLE, VISUAL_EVIDENCE_UNAVAILABLE |
| gameplay | DURATION_OUT_OF_RANGE, VISUAL_EVIDENCE_UNAVAILABLE | DURATION_OUT_OF_RANGE, SEMANTIC_INCOMPLETE, VISUAL_EVIDENCE_UNAVAILABLE |

## Side-by-side selections

### Большинство бизнесов провалит ИИ-трансформацию — Андрей Дороничев

| Rank | OLD | NEW | Same | Main factor deltas | Evidence | Usefulness |
|---:|---|---|---|---|---|---|
| 1 | `candidate-chapter-020-story-002` 1841.0–1875.29 | `candidate-chapter-011-story-001` 878.36–909.13 | False | `{"visual_interest": 100.0, "vertical_viability": 82.0, "information_value": -13.9, "emotional_intensity": -13.839}` | pass2=skipped, audio=0, visual=2 | useful |
| 2 | `candidate-chapter-095-story-003` 8028.9–8048.03 | `candidate-chapter-095-story-003` 8028.9–8048.03 | True | `{"vertical_viability": 50.0, "visual_interest": 32.5, "confidence": -19.984, "emotional_intensity": -13.09}` | pass2=None, audio=0, visual=0 | neutral |
| 3 | `candidate-chapter-012-story-001` 989.4–1015.52 | `candidate-chapter-032-story-002` 2500.79–2551.2 | False | `{"vertical_viability": 82.0, "visual_interest": 75.265, "emotional_intensity": -15.689, "audio_energy": -13.935}` | pass2=not_requested, audio=0, visual=1 | mixed |

### Еда, которая нас убивает

| Rank | OLD | NEW | Same | Main factor deltas | Evidence | Usefulness |
|---:|---|---|---|---|---|---|
| 1 | `candidate-chapter-006-story-002` 514.32–551.32 | `candidate-chapter-014-story-002` 1220.05–1259.66 | False | `{"visual_interest": 100.0, "vertical_viability": 38.0, "emotional_intensity": 31.453, "context_debt": -10.583}` | pass2=completed, audio=19, visual=2 | useful_with_continuity_risk |
| 2 | `candidate-chapter-014-story-001` 1200.67–1220.3 | `candidate-chapter-006-story-003` 551.07–566.66 | False | `{"vertical_viability": 50.0, "visual_interest": 32.5, "confidence": -26.864, "information_value": -11.3}` | pass2=None, audio=21, visual=0 | mixed |
| 3 | `candidate-chapter-009-story-002` 735.99–753.5 | `candidate-chapter-009-story-003` 753.25–773.32 | False | `{"visual_interest": 83.0, "vertical_viability": 82.0, "confidence": -14.128, "information_value": -13.9}` | pass2=not_requested, audio=17, visual=1 | useful |

### Пробую пельмени Зубарева в Китае

| Rank | OLD | NEW | Same | Main factor deltas | Evidence | Usefulness |
|---:|---|---|---|---|---|---|
| 1 | `candidate-chapter-008-story-001` 300.43–338.08 | `candidate-chapter-021-story-001` 721.17–740.16 | False | `{"visual_interest": 100.0, "vertical_viability": 82.0, "confidence": -21.328, "emotional_intensity": -16.538}` | pass2=skipped, audio=0, visual=2 | useful |
| 2 | `candidate-chapter-014-story-001` 558.15–575.12 | `candidate-chapter-033-story-001` 1181.64–1205.11 | False | `{"visual_interest": 83.0, "vertical_viability": 82.0, "narrative_completeness": -18.125, "confidence": -12.184}` | pass2=not_requested, audio=1, visual=1 | useful |
| 3 | `candidate-chapter-001-story-002` 32.61–54.77 | `candidate-chapter-012-story-001` 499.78–539.57 | False | `{"visual_interest": 100.0, "vertical_viability": 82.0, "emotional_intensity": -15.339, "information_value": 15.1}` | pass2=not_requested, audio=0, visual=1 | useful |

### PUBG local gameplay source

| Rank | OLD | NEW | Same | Main factor deltas | Evidence | Usefulness |
|---:|---|---|---|---|---|---|
| 1 | `candidate-chapter-010-story-001` 234.79–249.84 | `candidate-chapter-023-story-001` 499.83–525.1 | False | `{"vertical_viability": 82.0, "visual_interest": 62.187, "information_value": 12.0, "audio_energy": 5.5}` | pass2=not_requested, audio=1, visual=1 | useful |
| 2 | `candidate-chapter-007-story-001` 138.28–153.98 | `candidate-chapter-002-story-001` 17.4–56.77 | False | `{"vertical_viability": 82.0, "visual_interest": 69.207, "emotional_intensity": -11.616, "audio_energy": -11.155}` | pass2=not_requested, audio=0, visual=1 | useful |
| 3 | `candidate-chapter-008-story-002` 193.12–219.07 | `candidate-chapter-008-story-002` 193.12–219.07 | True | `{"vertical_viability": 82.0, "visual_interest": 61.563, "emotional_intensity": -17.279, "audio_energy": -15.015}` | pass2=not_requested, audio=0, visual=1 | neutral_same_candidate |

## Failure / fallback and overvaluation matrix

| Case | Final score | Visual interest | Audio energy | Confidence |
|---|---:|---:|---:|---:|
| missing_vision | 60.830 | 32.500 | 0.000 | 66.660 |
| weak_transcript | 44.342 | 32.500 | 0.000 | 36.900 |
| visual_heavy | 73.091 | 55.500 | 33.250 | 88.260 |
| audio_driven | 71.318 | 32.500 | 95.000 | 66.660 |
| no_strong_multimodal_evidence | 60.830 | 32.500 | 0.000 | 66.660 |
| random_motion | 65.220 | 22.500 | 0.000 | 87.660 |
| ordinary_scene_change | 60.830 | 32.500 | 0.000 | 66.660 |
| weak_reaction | 60.830 | 32.500 | 0.000 | 66.660 |
| loud_sound_without_editorial_value | 64.501 | 32.500 | 33.250 | 66.660 |
| low_confidence_visual | 60.830 | 32.500 | 0.000 | 66.660 |

## Confirmed improvements

- Podcast top selection gained story-relevant B-roll rather than another static talking-head frame.
- Food-vlog top selection preserved the actual dish/reveal instead of selecting only presenter commentary.
- Gameplay top selection moved from a downed/static state to active play with source-grounded action evidence.
- Interview selection preserved a relevant numeric visual payoff; PASS 2 also exposed its high continuity risk for review.
- Every repeated Vision analysis used cache entries with zero provider calls and zero frames sent.

## Confirmed regressions and calibration

- Failure matrix proved that sub-0.65 visual confidence could still raise visual interest, payoff and vertical viability. Calibration now excludes that evidence from ranking/composition intent; covered by `test_low_confidence_visual_evidence_cannot_boost_editorial_or_composition_intent`.
- Failure matrix proved raw loudness received the same emotional/audio treatment as a grounded editorial event. Calibration now retains only a weak 35% relevance contribution unless multimodal provenance contains an audio event; covered by `test_raw_loudness_is_weaker_than_grounded_audio_editorial_event`.
- Generic action/reaction weights were not changed: real midpoint inspection showed useful B-roll, food, and gameplay improvements.

## Remaining weaknesses

- Candidate generation exceeds `max_candidates` on the long podcast (255), leaving no room for 6C composites; interview was the only source where count expanded (56→59).
- Side-by-side factor deltas compare differently shaped OLD/NEW contracts and should be read with evidence/frame review, not as isolated quality truth.
- PASS 2 marked the top interview candidate `continuity_risk=high`; current scoring reduces completeness but does not make this a hard rejection.
- Full 4K AV1 cold source analysis is not part of OLD-vs-NEW scoring time; an attempted diagnostic run showed material decode overhead and was stopped rather than conflated with Editorial Brain latency.
- API cost uses the gateway's configured token prices and recorded usage; it is an estimate, not a provider invoice.
