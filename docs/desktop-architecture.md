# Desktop architecture — Goal 4A

## Границы

Desktop-слой не заменяет и не импортирует отдельные стадии движка в widgets.

```text
PySide6 widgets
  -> ViewModels
  -> DesktopServices / stores / error mapping
  -> PipelineFacade + QtPipelineRunner
  -> existing `python -m app process`
  -> existing Content Factory engine
```

`PipelineFacade` — единственная точка, которая знает путь к engine, формирует
аргументы и создаёт временный YAML-конфиг для конкретного запуска. Он передаёт
аргументы массивом в `QProcess`, без shell и без API-ключей в командной строке.

Выбран `QProcess`, а не прямой импорт `Pipeline`, потому что текущий engine не
имеет cooperative cancellation API и рассчитан на изолированный CLI-run.
Изоляция не блокирует Qt event loop, позволяет штатно остановить дочерний
процесс и не требует рефакторинга существующих контрактов pipeline.

Прогресс строится честно: runner читает только известный `state.json` через
service layer и показывает смену этапа, indeterminate indicator и elapsed time.
Процент не симулируется. Widgets не читают engine JSON напрямую.

## Реализованные экраны

- Projects: выбор одного файла, drag-and-drop, список persistent projects,
  открытие папки и подтверждённое удаление только app-data.
- Project: source/result preview с системным fallback, metadata, autosave,
  существующие настройки субтитров/encoder/cache, запуск, отмена и run history.
- Settings: data/config path, Auto/On/Off GPU preference, cache location,
  реальный local mock mode, redacted API-key status и встроенный doctor.
- Onboarding: короткое приветствие, локальное хранение и doctor без требования ключа.

В Goal 4A намеренно отсутствуют очередь нескольких jobs, редактор timeline,
несколько Shorts, сценарный редактор, smart crop, billing UI, аккаунты, web/
cloud services, installer и автопостинг.

## Безопасность

Проекты не копируют source video. Удаление допускается только внутри
`projects/<project_id>` после проверки пути и отсутствия symbolic links/junctions.
Логи редактируют bearer/API-key patterns; settings и run records не имеют полей
для secret values. Открытие файла/папки происходит только после проверки
существования пути.

## Ограничение

`CONTENT_FACTORY_UX_CONCEPT(1).md` не находился в предоставленном workspace,
поэтому для Goal 4A UX source of truth — приложенная спецификация задачи. Его
следует добавить в репозиторий до следующей дизайн-фазы.
