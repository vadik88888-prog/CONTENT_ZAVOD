class ClipEngineError(RuntimeError):
    """Ошибка, которую можно безопасно показать пользователю."""


NO_RENDERABLE_CLIPS = "NO_RENDERABLE_CLIPS"
NO_RENDERABLE_CLIPS_MESSAGE = (
    "Не удалось подготовить ни одного ролика к созданию. "
    "Проверьте отчёт анализа или попробуйте другой режим."
)

DUPLICATE_EXACT_SOURCE_RANGE = "DUPLICATE_EXACT_SOURCE_RANGE"


class DependencyError(ClipEngineError):
    """Не установлена или недоступна внешняя программа."""


class SourceError(ClipEngineError):
    """Источник видео недоступен или не поддержан."""


class StageError(ClipEngineError):
    """Критическая ошибка одного этапа конвейера."""


class SemanticCredentialError(ClipEngineError):
    """Semantic AI cannot run because its configured credential is unusable."""


class SemanticProviderUnavailableError(ClipEngineError):
    """Semantic AI is temporarily unavailable; completed local work is reusable."""

    def __init__(self, message: str, usage: dict[str, object]) -> None:
        self.usage = usage
        super().__init__(message)


class VisionCredentialError(ClipEngineError):
    """Vision AI cannot run because its configured credential is unusable."""


class TransformationConfigurationError(ClipEngineError):
    """Неподдерживаемая или небезопасная настройка transformation."""


class SemanticExtractionError(ClipEngineError):
    """Semantic representation не прошла обязательную валидацию."""


class NarrativePlanningError(ClipEngineError):
    """Narrative plan не может быть построен из подтверждённых фактов."""


class ScriptGenerationError(ClipEngineError):
    """Script draft не имеет безопасной проверяемой структуры."""


class ScriptValidationError(ClipEngineError):
    """Сценарий не прошёл детерминированную проверку качества."""


class GroundingValidationError(ScriptValidationError):
    """Сценарий содержит неподтверждённый или искажённый материал."""


class TransformationProviderError(ClipEngineError):
    """Внешний AI provider не вернул допустимый transformation result."""


class TransformationFallbackError(ClipEngineError):
    """Консервативный local fallback не смог собрать сценарий."""


class ProductionPlanError(ClipEngineError):
    """Невозможно построить безопасный production plan из FinalScript."""


class TTSError(ClipEngineError):
    """TTS provider or generated-audio validation failed without changing the Production Plan."""


class TTSCredentialError(TTSError):
    """A cloud TTS request cannot run because its credential is unusable."""


class AudioCompositionError(ClipEngineError):
    """Audio Project could not be safely composed from existing plan and audio artifacts."""


class ProductionPlanHandoffError(AudioCompositionError):
    """A machine-readable ProductionPlan invariant failed before audio work began."""

    def __init__(self, code: str, evidence: dict[str, object]) -> None:
        self.code = code
        self.evidence = evidence
        super().__init__(f"{code}: ProductionPlan audio handoff rejected.")


class ProductionRenderError(ClipEngineError):
    """Goal 3D could not safely build a final video from existing production artifacts."""

    def __init__(
        self,
        message: str,
        *,
        quality_gate_report: dict[str, object] | None = None,
        artifact_reference: dict[str, object] | None = None,
    ) -> None:
        self.quality_gate_report = quality_gate_report
        self.artifact_reference = artifact_reference
        super().__init__(message)
