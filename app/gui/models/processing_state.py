from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProcessingPhase(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


STAGE_LABELS = {
    "download": "Загружаем видео",
    "source": "Подготавливаем видео",
    "media": "Подготавливаем видео",
    "transcription": "Распознаём речь",
    "transcript_features": "Ищем сильные моменты",
    "audio_features": "Ищем сильные моменты",
    "scene_detection": "Ищем сильные моменты",
    "candidate_generation": "Ищем сильные моменты",
    "ai_reranking": "Понимаем содержание",
    "content_transformation": "Создаём сценарий",
    "production_plan": "Создаём сценарий",
    "tts": "Готовим озвучку",
    "audio": "Готовим озвучку",
    "production_render": "Собираем ролик",
    "render": "Собираем ролик",
    "report": "Проверяем результат",
}


@dataclass(slots=True)
class ProcessingSnapshot:
    phase: ProcessingPhase = ProcessingPhase.IDLE
    stage: str | None = None
    message: str = "Готово к созданию ролика"
    elapsed_seconds: float = 0.0
    last_activity_at: str | None = None
    last_activity_reason: str | None = None
    progress_fraction: float | None = None
    transfer_speed: str | None = None
    transfer_downloaded: str | None = None
    transfer_total: str | None = None
    eta_seconds: int | None = None

    @property
    def stage_label(self) -> str:
        if not self.stage:
            return self.message
        parent = self.stage.split(":", 1)[0]
        return STAGE_LABELS.get(parent, "Обрабатываем видео")
