# Desktop project storage — Goal 4A

По умолчанию app data расположены в `%LOCALAPPDATA%\ContentFactoryData`.
Папку можно выбрать на Settings screen; source video остаётся по исходному пути.

```text
ContentFactoryData/
  settings.json
  projects/
    <project_id>/
      project.json
      runs/
        <run_id>/
          run.json
          runtime-config.yaml
          pipeline.log
          report.json
          artifacts/
            final-short.mp4
```

`project.json` содержит ID, имя, timestamps, source path, project directory,
статус, текущие реальные project options, latest run ID и source metadata.
Поддерживаемые статусы: `draft`, `ready`, `queued`, `processing`, `completed`,
`completed_with_warnings`, `failed`, `cancelled`, `interrupted`.

Каждый `run.json` append-only: ID запуска, project/source/settings snapshots,
version, лог, report, artifact paths, warnings, безопасную error summary и
`cost_estimate`, понятную `error_summary` и redacted `technical_details`.
`actual_cost` всегда `null`, пока не добавлена отдельная
сверка с billing provider. Предыдущая запись никогда не перезаписывается.

После успеха копируется report и для MP4 создаётся hard link; если проекты и
output находятся на разных volume, используется копия готового артефакта.
Исходный длинный source никогда не копируется. Это сохраняет историю, даже если
следующий engine run обновит общий `output/<source>/report.json`.

Все JSON пишутся атомарно через temporary file + replace. Повреждённый project
или run пропускается при listing и не ломает другие проекты. Отсутствующий
`schema_version` считается совместимой v1 записью; неизвестная будущая версия
безопасно не загружается.
