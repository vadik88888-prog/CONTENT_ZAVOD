from __future__ import annotations

import re
from dataclasses import dataclass


_SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+[^\s]+|(?:OPENAI|GEMINI)_API_KEY\s*=\s*[^\s]+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class UserFacingError:
    title: str
    user_message: str
    suggested_action: str
    technical_details: str
    error_code: str


def dialog_message(error: UserFacingError) -> str:
    """Show a friendly next step; technical detail belongs in the run log."""

    message = error.user_message.strip()
    action = error.suggested_action.strip()
    if not action or action == message:
        return message
    return f"{message}\n\n{action}"


def redact_secrets(value: object) -> str:
    return _SECRET.sub("[скрыто]", str(value))


def map_error(error: object) -> UserFacingError:
    detail = redact_secrets(error)
    lowered = detail.lower()
    if "no_draft_previews" in lowered or "no candidate draft could be assembled" in lowered:
        return UserFacingError(
            "Не удалось подготовить выбранные черновики",
            "Ни один из выбранных моментов не прошёл проверку для Draft Preview.",
            "Откройте отмеченные карточки, скорректируйте границы при необходимости и повторите только неуспешные моменты.",
            detail,
            "draft_candidates_failed",
        )
    if "no_renderable_clips" in lowered or "no approved draft could be rendered" in lowered:
        return UserFacingError(
            "Нет готовых черновиков для финального экспорта",
            "Подтверждённые черновики не прошли проверку ProductionPlan.",
            "Повторите только отмеченные Draft Preview, затем снова подтвердите готовые черновики.",
            detail,
            "approved_drafts_invalid",
        )
    if "обработка остановилась и не отвечает" in lowered:
        return UserFacingError(
            "Обработка не отвечает",
            "Обработка остановилась и не отвечает.",
            "Откройте технический журнал: там сохранены команда, состояние процесса и время последней активности.",
            detail,
            "pipeline_stalled",
        )
    if "итоговый видеофайл" in lowered:
        return UserFacingError(
            "Не удалось создать итоговый видеофайл",
            "Не удалось создать итоговый видеофайл.",
            "Проверьте технический журнал и повторите запуск.",
            detail,
            "final_output_missing",
        )
    if "yt-dlp" in lowered or "загрузить видео по этой ссылке" in lowered or "ссылк" in lowered and "видео" in lowered:
        if "компонент" in lowered or "yt-dlp" in lowered:
            return UserFacingError(
                "Не готова загрузка по ссылке",
                "Для загрузки по ссылке требуется дополнительный компонент.",
                "Установите yt-dlp или добавьте видео как локальный файл.",
                detail,
                "ytdlp_missing",
            )
        return UserFacingError(
            "Не удалось загрузить видео",
            "Не удалось получить видео по этой ссылке.",
            "Проверьте, что ссылка публичная, или сохраните видео вручную и добавьте как файл.",
            detail,
            "url_download_failed",
        )
    if "не найден" in lowered and ("видео" in lowered or "source" in lowered or "file" in lowered):
        return UserFacingError("Файл недоступен", "Не удалось найти исходное видео.", "Проверьте путь к файлу и подключённые диски.", detail, "source_missing")
    if "ffmpeg" in lowered or "ffprobe" in lowered:
        return UserFacingError("Не готова среда", "Для создания ролика не хватает компонента обработки видео.", "Откройте «Настройки → Диагностика» и выполните проверку.", detail, "media_dependency")
    if "api key" in lowered or "authentication" in lowered or "401" in lowered:
        return UserFacingError("Не удалось подключиться к AI", "Проверьте настройку ключа API или используйте локальный тестовый режим.", "Добавьте ключ в .env и повторите попытку.", detail, "provider_auth")
    if "cancel" in lowered or "отмен" in lowered:
        return UserFacingError("Создание отменено", "Создание ролика было остановлено.", "Можно изменить настройки и запустить проект снова.", detail, "cancelled")
    if "permission" in lowered or "access is denied" in lowered or "доступ" in lowered:
        return UserFacingError("Нет доступа к файлу", "Приложение не может записать результат.", "Проверьте права на папку данных и наличие свободного места.", detail, "permission_denied")
    return UserFacingError("Не удалось создать ролик", "Обработка завершилась с ошибкой.", "Проверьте исходный файл и откройте технический журнал при необходимости.", detail, "pipeline_failed")
