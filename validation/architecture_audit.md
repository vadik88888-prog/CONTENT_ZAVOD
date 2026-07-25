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
| P1 | `report.json` хранит последний state source, а не неизменяемую историю запусков. Aggregated timing/cache metrics могут смешивать старые и cache-only этапы. | Перед GUI добавить отдельный append-only run manifest с run id и stage provenance; это требует согласования формата, поэтому не внедрялось в Goal 3E. |
| P1 | Local reports содержат оценку стоимости, а не подтверждённую invoice/billing сумму; mock runs не имеют расходов. | Для production добавить reconciliation с экспортом billing провайдера или явный operator-entered actual cost. Не выводить estimate как actual. |
| P2 | 10-run stress создаёт новый CLI process на итерацию. Он ловит repeat-run failures, но не доказывает отсутствие утечек в долгоживущем GUI process. | После появления GUI добавить soak test одной process/session с telemetry RSS/VRAM и заданным порогом роста. |
| P2 | TTS, dialogue extraction и render caches не имеют policy retention/quota. Длинная пользовательская библиотека может неограниченно занять диск. | До широкого desktop rollout определить limit, LRU/очистку и UI-индикацию размера cache. |
| P2 | Subtitle visual QA проверена на synthetic English speech и fixed samples. | Проверить реальные языки, длинные слова, кириллицу и user font fallback на целевых Windows-машинах. |

## Решение по готовности

Техническое ядро готово к **внутреннему GUI integration**: typed artifacts,
isolated stages, deterministic mock mode, cache recovery и final MP4 проверены.
Production-quality GUI rollout не следует объявлять завершённым до закрытия P0
и P1 рекомендаций выше.
