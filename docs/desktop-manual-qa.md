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

## UX stabilization acceptance

Проверьте на Windows новый путь ровно в этой последовательности: «Источник →
Загрузка → Настройка → Обработка → Моменты → Черновики → Готовые ролики».

| Сценарий | Ожидаемый результат |
|---|---|
| Локальный файл | После выбора сразу открывается «Настройка»; одна основная кнопка запускает поиск моментов. |
| Открытая ссылка на длинное видео | После проверки ссылки открывается отдельная «Загрузка». Во время скачивания видны статус, процент, скорость, объём, ETA и «Отменить»; поиск сам не начинается. |
| Отмена и повтор | Отмена удаляет только неполный файл в папке проекта. После завершения кнопка повторной загрузки доступна и скачивание можно запустить снова. |
| Полный путь | После поиска выбрать моменты, собрать и подтвердить черновики, создать готовые ролики, затем открыть получившийся MP4 из карточки. |
| Перезапуск | Проект открывается на шаге, который следует из сохранённых source/project/artifact states; прерванная загрузка предлагается к повтору, а не выглядит активной. |
