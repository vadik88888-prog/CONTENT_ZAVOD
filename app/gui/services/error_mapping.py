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


def redact_secrets(value: object) -> str:
    return _SECRET.sub("[скрыто]", str(value))


def map_error(error: object) -> UserFacingError:
    detail = redact_secrets(error)
    lowered = detail.lower()
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
