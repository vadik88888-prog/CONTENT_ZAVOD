# Архитектурный аудит — Goal 3E

## Карта движка

```text
source video
  -> metadata / transcription / audio + scene features
  -> candidate generation + local scoring + provider reranking
  -> grounded Content Transformation
  -> ProductionPlan
  -> TTS artifacts -> AudioProject / mixed_audio.wav
  -> VideoProject / ASS subtitles -> final H.264/AAC MP4
```

Границы Goal 2, TTS, Audio Composition и Video Composition изолированы: каждый
нижестоящий этап получает типизированные артефакты предыдущего и имеет свой
cache key. Это хорошая основа для GUI: интерфейсу не нужно знать детали
FFmpeg, провайдеров или структуры timeline.

## Подтверждённые результаты

- `pytest` успешно проходит 121 regression-тест; `doctor` с local validation
  profile зелёный.
- 7 synthetic category-прокси прошли полный local pipeline.
- 10 source-format вариантов прошли isolated production render: 720p, 1080p,
  1440p, 2160p; 24/25/30/50/60 FPS; vertical и square.
- 10 последовательных cache render runs завершились успешно; средняя
  длительность 0.762 с, измеренный рост peak working set — 0 байт.
- 30-минутный source прошёл полный pipeline и final render; 60 и 90 минут
  прошли до ProductionPlan в `--production-plan-only` режиме.
- Direct QA of the current 30-minute synthetic final mix: EBU R128 integrated loudness
  `-18.5 LUFS`, loudness range `10.8 LU`, true peak `-5.3 dBFS`. This is evidence
  for the synthetic dialogue/narration mix, not a replacement for category-specific
  human listening or a billing-backed production acceptance threshold.
- `blackdetect` on the current synthetic final render found no black interval of at
  least `0.1 s` using the validation command's threshold. Frame samples also confirm
  that the scaled subtitles stay inside the 360x640 canvas.

## Найденные и исправленные дефекты

1. **Исправлено, P1.** TTS cache возвращал `segment_id` предыдущего
   ProductionPlan при совпадающем normalized text. Это могло потерять сегменты
   в результатах нового plan. Cache hit теперь перепривязывает plan-local поля;
   добавлен regression-тест.
2. **Исправлено, P1.** ASS subtitle styles были абсолютными и при 360×640
   могли переносить текст за верхнюю границу. Шрифт, outline, shadow и margins
   теперь масштабируются от reference canvas 1080×1920; engine version входит
   в render cache key, поэтому старые кадры не переиспользуются.

## Риски и рекомендации перед GUI

| Приоритет | Риск | Рекомендация и компромисс |
|---|---|---|
| P0 | В workspace нет лицензированных реальных источников. Synthetic clips не доказывают выбор моментов, смысловую корректность, реальные subtitle errors или visual quality для заявленных категорий. | До production sign-off собрать разрешённый dataset по всем категориям и провести human QA. Не заменять его synthetic-прокси. |
| P1 (partially superseded) | `report.json` хранит последний state source, а не неизменяемую историю запусков. Aggregated timing/cache metrics могут смешивать старые и cache-only этапы. | **Historical correction:** the recommendation to add a separate append-only, run-scoped manifest is superseded by the current implementation of run-scoped manifests and canonical result handling. The `report.json` aggregate-metric concern needs independent review. The P0 warning about licensed real-media fixtures and human QA remains current. |
| P1 | Local reports содержат оценку стоимости, а не подтверждённую invoice/billing сумму; mock runs не имеют расходов. | Для production добавить reconciliation с экспортом billing провайдера или явный operator-entered actual cost. Не выводить estimate как actual. |
| P2 | 10-run stress создаёт новый CLI process на итерацию. Он ловит repeat-run failures, но не доказывает отсутствие утечек в долгоживущем GUI process. | После появления GUI добавить soak test одной process/session с telemetry RSS/VRAM и заданным порогом роста. |
| P2 | TTS, dialogue extraction и render caches не имеют policy retention/quota. Длинная пользовательская библиотека может неограниченно занять диск. | До широкого desktop rollout определить limit, LRU/очистку и UI-индикацию размера cache. |
| P2 | Subtitle visual QA проверена на synthetic English speech и fixed samples. | Проверить реальные языки, длинные слова, кириллицу и user font fallback на целевых Windows-машинах. |

## Решение по готовности

Техническое ядро готово к **внутреннему GUI integration**: typed artifacts,
isolated stages, deterministic mock mode, cache recovery и final MP4 проверены.
Production-quality GUI rollout не следует объявлять завершённым до закрытия P0
и P1 рекомендаций выше.
