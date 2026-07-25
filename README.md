# Content Factory — Clip Engine

Локальный технический прототип для Windows: он принимает одно длинное видео или поддерживаемую публичную ссылку, распознаёт речь локально, находит кандидаты на короткие ролики, оценивает их и создаёт вертикальные MP4 с субтитрами. Всё хранится на вашем компьютере.

Проект рассчитан на ноутбук с NVIDIA RTX 3050 Ti и 16 ГБ RAM, но умеет перейти на CPU, если CUDA недоступна. OpenAI — основной провайдер смысловой оценки; Gemini остаётся опциональным. Без API-ключей полностью работает режим '--mock-ai'.

## AI-провайдеры

По умолчанию используется **OpenAI** через Responses API и Structured Outputs. Для
оценки кандидатов выбрана `gpt-5-mini`: это актуальная недорогая модель для
строго заданной задачи оценки и JSON-результата. Модель, число повторов и тарифы
за токены задаются только в `ai` в `config.yaml`.

Создайте локальный файл с секретами — он уже исключён из Git:

~~~powershell
Copy-Item .env.example .env
~~~

Для OpenAI заполните `OPENAI_API_KEY` в `.env`. Оплата OpenAI API ведётся
отдельно от подписки ChatGPT; наличие подписки ChatGPT не даёт API-кредиты.

~~~yaml
ai:
  provider: openai
  model: gpt-5-mini
  max_retries: 2
~~~

Чтобы выбрать Gemini, укажите `provider: gemini`, `model` Gemini-модели и
`GEMINI_API_KEY` в `.env`. Gemini остаётся опциональным провайдером.

Для полностью локальной проверки ключи не нужны — флаг ниже всегда принудительно
включает детерминированный `mock`, независимо от `config.yaml`:

~~~powershell
python -m app process --input ".\input\test-video.mp4" --mock-ai
~~~

Обычный запуск берёт провайдер из конфигурации:

~~~powershell
python -m app process --input ".\input\test-video.mp4" --config config.yaml
~~~

`python -m app doctor --config config.yaml` показывает выбранный provider и
проверяет только нужный ему ключ, не раскрывая его значение. В `report.json`
добавлен блок `ai` с provider, model, токенами, приблизительной стоимостью,
повторами и безопасными сообщениями API.

## Что реализовано

- CLI-команда проверки окружения: 'python -m app doctor';
- вход из локального файла или поддерживаемой публичной ссылки через 'yt-dlp';
- безопасные имена файлов и запуск внешних программ без shell-инъекций;
- 'ffprobe'-метаданные и WAV 16 kHz mono для распознавания;
- транскрибация 'faster-whisper' с сегментами и таймкодами слов;
- локальные кандидаты длиной 15–60 секунд, с padding до/после фразы;
- OpenAI-провайдер (основной), Gemini-провайдер (опциональный) и детерминированный local mock-провайдер;
- Grounded Content Transformation Core: отдельный проверяемый сценарий для будущей TTS-стадии;
- отбор до пяти непересекающихся клипов;
- вертикальный рендер 'blur-background' либо 'center-crop';
- ASS-субтитры с кириллицей;
- статусы этапов, кэш и подробный 'report.json'.

## Чего пока нет

Это не SaaS и не полный аналог OpusClip. Здесь нет интерфейса, аккаунтов, публикации в соцсети, облачной базы, оплаты, TTS/новой озвучки, музыки, B-roll, face/speaker tracking и сложного видеоредактора.

Не поддерживаются обход DRM, paywall, авторизации или технических ограничений видеоплатформ. Используйте только видео и публичные ссылки, на обработку которых у вас есть права.

## Установка на Windows

### 1. Установите Python 3.11 или новее

Скачайте Python с [python.org](https://www.python.org/downloads/windows/). Во время установки включите пункт **Add Python to PATH**. Закройте и снова откройте PowerShell, затем проверьте:

~~~powershell
python --version
~~~

Должно быть не ниже 'Python 3.11'.

### 2. Установите FFmpeg

Установите FFmpeg любым удобным способом и добавьте папку 'bin' в переменную окружения 'PATH'. Затем проверьте:

~~~powershell
ffmpeg -version
ffprobe -version
~~~

Обе команды должны показать версию. 'ffprobe' устанавливается вместе с FFmpeg.

### 3. Необязательно: подготовьте NVIDIA CUDA

Установите актуальный драйвер NVIDIA. Проверьте его командой:

~~~powershell
nvidia-smi
~~~

Если команда не работает, программа всё равно сможет использовать CPU — это будет медленнее. Для RTX 3050 Ti начните с модели Whisper 'small', указанной в конфигурации.

### 4. Создайте виртуальное окружение и поставьте пакеты

Откройте PowerShell в папке проекта и выполните:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
~~~

Если PowerShell запрещает запуск 'Activate.ps1', выполните один раз для текущего пользователя:

~~~powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
~~~

Затем повторите активацию окружения.

### 5. Проверьте окружение

~~~powershell
python -m app doctor
~~~

Отчёт объясняет каждую проблему простым текстом. Отсутствие ключа выбранного AI-провайдера — только предупреждение: mock-режим доступен без ключа.

## Конфигурация

Скопируйте пример, если хотите изменить настройки:

~~~powershell
Copy-Item config.example.yaml config.yaml
~~~

В 'config.yaml' можно выбрать модель Whisper, язык, CUDA/CPU, длительности клипов, лимит роликов, порог оценки, режим вертикального рендера, субтитры и удаление скачанного исходника. По умолчанию используется 'blur-background', чтобы не обрезать важные края горизонтального видео.

Секреты не хранятся в YAML и не попадают в Git. Скопируйте '.env.example' в '.env' и добавьте ключ выбранного провайдера:

~~~text
OPENAI_API_KEY=ваш_ключ
# GEMINI_API_KEY=ваш_ключ
~~~

## Быстрый запуск без API

Поместите видео в 'input' или укажите любой путь к нему:

~~~powershell
python -m app process --input ".\input\test-video.mp4" --mock-ai
~~~

Для своей конфигурации добавьте '--config config.yaml':

~~~powershell
python -m app process --input "C:\Videos\source.mp4" --config config.yaml --mock-ai
~~~

Mock-режим оценивает фрагменты детерминированно. Он нужен для настройки и тестирования pipeline, а не заменяет смысловую оценку OpenAI или Gemini.

## Запуск с OpenAI или Gemini

После заполнения '.env' выполните:

~~~powershell
python -m app process --input ".\input\test-video.mp4" --config config.yaml
~~~

OpenAI и Gemini получают только структурированный текст транскрипта и список кандидатов с таймкодами, а не сам видеофайл. OpenAI использует Responses API и Structured Outputs со строгой JSON Schema; для обоих провайдеров при ошибке выполняется ограниченный повтор, после чего pipeline сохраняет безопасное сообщение в отчёте.

## Обработка публичной ссылки

~~~powershell
python -m app process --url "https://example.com/public-video" --mock-ai
~~~

Для ссылки нужен установленный 'yt-dlp' и поддерживаемая публичная страница. Не используйте закрытые, платные или защищённые DRM-видео. Если в конфигурации включить 'delete_downloaded_source: true', загруженный оригинал удалится после успешного создания хотя бы одного клипа.

## Где искать результаты

Для каждого источника создаются отдельные папки:

~~~text
work/<имя-и-id>/
  source.json
  state.json
  metadata.json
  transcript.json
  transcript.txt
  candidates.raw.json
  candidates.scored.json
  render.json

output/<имя-и-id>/
  clip-01-....mp4
  clip-01-....ass
  report.json
~~~

При повторном запуске с тем же локальным файлом этапы с успешным статусом и существующими артефактами не выполняются заново. Это особенно экономит время транскрибации. Если один рендер не удался, остальные клипы и все отчёты сохраняются.

## Как читать report.json

'report.json' содержит источник, длительность, статусы и время этапов, использовались ли GPU/NVENC, число кандидатов и выбранных роликов, пути к MP4, предупреждения, ошибки и блок `ai`: provider, model, входные/выходные токены, приблизительную стоимость, повторы и безопасные ошибки API.

## Clip Intelligence Core (Goal 1.6)

Перед обращением к AI проект локально вычисляет признаки транскрипта, аудио и
смен сцен. Кандидаты формируются у естественных границ предложений, пауз и сцен,
получают объяснимые локальные оценки, дедуплицируются и попадают в shortlist.
OpenAI/Gemini/Mock используются только для semantic reranking этого shortlist.

Локальный `local_quality_score` — взвешенная сумма `hook`, `completeness`,
`clarity`, `speech_density`, `pacing`, `audio_energy`, `scene_structure`,
`context_independence` и `boundary_quality`; после неё отдельно применяются
штрафы повторов и слов-паразитов. Веса валидируются и настраиваются в
`config.yaml`.

Новые cache-артефакты находятся в `work/<source>/`: `transcript_features.json`,
`audio_features.json`, `scene_boundaries.json`, `candidates_v2.json`,
`shortlist.json`, `ai_ranking.json` и `final_selection.json`. Изменение AI
провайдера или весов не запускает повторное извлечение аудио или сцен.

Дополнительные режимы запуска:

~~~powershell
python -m app process --input ".\input\smoke-test.mp4" --no-ai-rerank
python -m app process --input ".\input\smoke-test.mp4" --mock-ai
python -m app process --input ".\input\smoke-test.mp4" --recompute-intelligence
~~~

`--no-ai-rerank` использует только local ranking. При недоступности AI pipeline
автоматически использует тот же local fallback и фиксирует это в `report.json`.
Блок `clip_intelligence` отчёта содержит количество признаков, пауз, сцен,
кандидатов, shortlist, mode/fallback, processing times и объяснения кандидатов.

## AI Content Transformation Core (Goal 2)

После `final_selection` проект может создать отдельный, grounded сценарий для
каждого выбранного клипа. Это дополнительный production artifact для будущей
TTS-стадии, а не замена звука или субтитров текущего ролика.

```text
selected candidate
  → SourceContext (primary evidence + supporting context)
  → SemanticRepresentation (facts с transcript evidence)
  → NarrativePlan
  → ScriptDraft
  → deterministic grounding + quality validation
  → limited repair или conservative local fallback
  → transformed-script.txt / transformed-script.json
```

`primary_evidence` — сам выбранный фрагмент. Соседний
`supporting_context` передаётся только для понимания контекста и не может
автоматически стать фактом сценария. Python валидирует ссылки фактов и сегментов,
числа, проценты, валюты, даты, имена, URL, абсолютные/сравнительные утверждения,
отрицания, модальность и преобразование мнения в факт. Небезопасный AI-draft
никогда не попадает в final script: после ограниченного repair используется
консервативный сценарий из исходных предложений либо безопасный статус `failed`.

По умолчанию `transformation.enabled: false`, чтобы прежний запуск не создавал
дополнительные API-вызовы. Для разового запуска включите artifact флагом:

```powershell
python -m app process --input ".\input\smoke-test.mp4" --mock-ai --transform-script --print-transformed-script
```

Полностью локальный, проверяемый fallback без AI:

```powershell
python -m app process --input ".\input\smoke-test.mp4" --transform-script --no-ai-transformation
```

Полезные параметры:

- `--no-transform-script` — принудительно отключить artifact;
- `--transformation-mode auto|faithful_compression|hook_first|...`;
- `--transformation-ai-strategy compact|staged|local_only`;
- `--target-duration 35`, `--allow-cta`, `--strict-grounding`;
- `--recompute-transformation` — пересчитать только transformation и report;
- `--print-transformed-script` — вывести безопасный final text в CLI.

В `config.yaml` за это отвечает секция `transformation`. Язык сохраняется по
умолчанию; полноценный перевод намеренно не имитируется в Goal 2: при запросе
другого языка pipeline фиксирует `not implemented`, не выдавая фальшивый перевод.

Для OpenAI используется один `compact` Responses API вызов со strict Structured
Outputs (semantic representation + plan + draft), после которого все критичные
проверки всё равно выполняются локально. `staged` сохраняет раздельные typed
этапы в pipeline и report, но может использовать один оптимизированный structured
response. Mock provider воспроизводим и имеет test modes для unsafe facts и
provider failure.

Результаты появляются рядом с MP4:

```text
output/<source>/
  original-transcript.txt
  transformed-script.txt
  transformed-script.json
  transformation-report-<candidate>.json
  clip-01-<candidate>.mp4
  clip-01-<candidate>.ass
  report.json
```

`clip-*.mp4` и `.ass` по-прежнему строятся только по исходному аудио и исходному
transcript. `report.json.content_transformation` содержит SourceContext,
semantics, plan, draft, validation, repair/fallback, scores, prompt versions,
timings, cache и безопасную AI metadata. Ключи и authorization headers туда не
записываются.

## AI Production Foundation (Goal 3A)

После готового `FinalScript` pipeline автоматически создаёт локальный
`ProductionPlan` — единственный источник правды для будущих TTS, audio mix и
subtitle-sync этапов. Он построен на Pydantic-моделях и не генерирует аудио или
видео.

```text
FinalScript
  → ProductionPlan
  → narration / dialogue placeholders / pauses
  → TimelineEstimate + SubtitleTrack placeholders
```

Для каждого narration блока план содержит текст, число слов, расчётную
длительность и WPS. Каждый `original_dialogue` placeholder связан с `fact_id`,
исходным transcript segment, таймкодами, speaker placeholder и confidence.
Dialogue placeholders не входят в master timeline и не вырезаются из видео: они
явно связаны с соответствующими narration blocks для будущего Dialogue Selection.

В план также входят placeholder `VoiceProfile`, четыре будущих `AudioLayer`
(`narration`, `original_dialogue`, `music`, `effects`) и `SubtitleTrack` с
примерными cue timings. Поля metadata явно фиксируют, что TTS, mix и render не
выполнялись.

Новые artifacts рядом с результатом:

```text
production-plan.json
timeline.json
production-summary.txt
```

Production Plan кэшируется отдельно по FinalScript, source evidence и
`production` config. Изменение render-настроек не пересчитывает его.

```powershell
# Обычная обработка: plan создаётся после FinalScript автоматически.
python -m app process --input ".\input\smoke-test.mp4" --mock-ai --transform-script

# Только FinalScript + Production Plan, без запуска render/ASS/FFmpeg.
python -m app process --input ".\input\smoke-test.mp4" --mock-ai --production-plan-only

# Пересчитать только Production Plan и report.
python -m app process --input ".\input\smoke-test.mp4" --mock-ai --transform-script --recompute-production-plan
```

`--production-plan-only` не перезаписывает существующий `render.json` и не
инвалидирует cache готового MP4. В Goal 3A намеренно отсутствуют TTS, OpenAI
Speech, ElevenLabs, voice cloning, audio mix/ducking, FFmpeg audio pipeline,
ASS/subtitle render и translation.

## Тесты

После активации виртуального окружения:

~~~powershell
pytest
~~~

Тесты проверяют валидацию конфигурации, границы кандидатов, mock-оценку и устранение пересечений, ASS-субтитры и валидацию источника.

## Типичные ошибки

| Сообщение | Что сделать |
| --- | --- |
| 'FFmpeg не найден' | Установите FFmpeg и перезапустите PowerShell. |
| 'yt-dlp не найден' | Активируйте '.venv' и повторите 'pip install -r requirements.txt'. |
| 'faster-whisper не установлен' | Активируйте '.venv' и поставьте зависимости. |
| Ошибка CUDA или нехватка памяти | В 'config.yaml' укажите 'device: cpu' либо используйте меньшую модель 'base'/'small'. |
| 'GEMINI_API_KEY не задан' | Добавьте ключ в '.env' или используйте '--mock-ai'. |
| Не найдено кандидатов | Уменьшите 'min_clip_duration'; убедитесь, что в видео есть разборчивая речь. |
| MP4 не создаётся | Посмотрите 'render.json' и 'report.json'; один сбой клипа не удаляет другие результаты. |
