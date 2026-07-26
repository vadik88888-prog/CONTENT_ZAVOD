# Desktop processing lifecycle — Goal 4A

1. Пользователь выбирает supported local video. `PipelineFacade` получает
   metadata через существующий `ffprobe` adapter и создаётся persistent project.
2. При «Создать ролик» создаётся `ProjectRun` со статусом `preparing`, затем
   отдельный runtime config и `QProcess` с существующим `python -m app process`.
3. `QProcess` передаёт stdout/stderr в rotating `pipeline.log`; UI получает
   человекочитаемые stage changes через `state.json` и не блокирует event loop.
4. После exit code 0 facade проверяет report, копирует его в run folder и
   архивирует созданные MP4. Warnings дают `completed_with_warnings`.
5. Ошибка, отсутствие report/output или ненулевой exit code дают `failed`;
   техническая информация остаётся в log, а UI получает mapped user message.

## Cancellation

Кнопка «Отменить» переводит run в `cancelling`, вызывает `QProcess.terminate()`
и через пять секунд — `kill()`, если процесс ещё жив. Частичные файлы не
архивируются как результат. Завершённый run получает `cancelled`; повторный
запуск доступен сразу после завершения процесса.

## Restart recovery

При старте app обходит сохранённые проекты. Каждый `preparing`, `running` или
`cancelling` run, оставшийся от аварийного закрытия, переводится в `interrupted`.
Проект с `processing` также становится `interrupted`. Resume с середины pipeline
не заявляется; следующий run может использовать существующий engine cache.
