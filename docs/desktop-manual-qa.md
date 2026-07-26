# Desktop manual QA — Goal 4A

Запуск в development environment:

```powershell
.\.venv\Scripts\python.exe -m app.gui
```

Перед ручной проверкой создайте `.env` при реальном provider либо включите в
Settings «Локальный тестовый режим без внешних API».

| Сценарий | Ожидаемый результат | Статус Goal 4A |
|---|---|---|
| Первый запуск | onboarding открывается, doctor работает без ключа | Проверено headless |
| Новый проект | один MP4 создаёт draft/ready project; metadata видны | Проверено focused tests |
| Restart | project и run history после перезапуска сохраняются | Проверено focused tests |
| Full pipeline | UI не блокируется, стадии меняются, MP4 архивируется | Проверено synthetic mock через QProcess |
| Cancellation | run становится cancelled, повторный запуск доступен | Проверено QProcess test |
| Failure | понятная ошибка, technical log сохранён | Проверено QProcess/nonzero-exit test |
| Кириллица и пробелы | input path принимается | Проверено tests и synthetic GUI run |
| Doctor | Settings показывает FFmpeg/FFprobe/GPU/provider без secret value | Проверить на целевой машине |
| GPU preference | Auto/Вкл./Выкл. меняет существующий `device` в runtime config | Проверено facade test |
| Preview | встроенный player или «Открыть в проигрывателе» | Проверить на целевой машине |
| Window close | confirmation; после restart active run is interrupted | Проверено persistence recovery |

Synthetic GUI run в Goal 4A прошёл по пути
`C:\Users\Вадим\Desktop\КОНТЕТ ЗАВОД\input\smoke-test.mp4`: completed with
warnings из-за ожидаемого NVENC → CPU fallback, `final-short.mp4` сохранён в
run-specific artifacts. Это техническая проверка, не human QA на реальном
лицензированном контенте.
