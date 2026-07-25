class ClipEngineError(RuntimeError):
    """Ошибка, которую можно безопасно показать пользователю."""


class DependencyError(ClipEngineError):
    """Не установлена или недоступна внешняя программа."""


class SourceError(ClipEngineError):
    """Источник видео недоступен или не поддержан."""


class StageError(ClipEngineError):
    """Критическая ошибка одного этапа конвейера."""


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
