# Desktop UX stabilization

## What was getting in the way

- The project page placed setup, source details, candidates, drafts, finished files, run history, and advanced settings on the same screen. The next action was easy to miss.
- A link download started as part of analysis. It was not clear that a file was being downloaded first, or what would happen when it finished.
- Download feedback omitted the transferred and expected volume. Long processing only showed a spinner and internal stage history, without explaining the current user-facing activity or what follows.
- Persisted project status was sufficient for the pipeline, but the GUI did not consistently translate a reopened URL or review project into its next user step.
- Main-screen labels exposed implementation terms such as "production render" and technical output details.

## Screen structure

The desktop client keeps the established persisted analysis, review, draft, and production contracts. It presents them through one derived, durable flow position:

1. **Источник** — select a local file or inspect a public link in the projects screen.
2. **Загрузка** — for a public link only; download is an explicit action with live yt-dlp status, percentage, speed, volume, ETA, and cancellation.
3. **Настройка** — choose the three common options, then start processing.
4. **Обработка** — show the current plain-language activity, elapsed time, the next activity, and cancellation. Percentages are shown only when the source provides a real one.
5. **Моменты** — review found moments and choose up to three for drafts.
6. **Черновики** — preview and explicitly keep the drafts to turn into finished videos.
7. **Готовые ролики** — watch or open the final files; history and advanced controls stay secondary.

The current position is calculated from the persisted source state, project status, and saved artifacts every time a project opens. A download left active by an application shutdown becomes repeatable rather than appearing to continue without a running process.
