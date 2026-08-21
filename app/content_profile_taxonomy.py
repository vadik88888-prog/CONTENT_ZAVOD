"""Canonical Source Content Profile v2 taxonomy contract.

This module intentionally depends only on the Python standard library.  It is
the single owner of profile schema versions, axis/value order, fallbacks, and
which values may be selected by a user.  ``auto`` is an input sentinel only;
it is never a stored profile value.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


CONTENT_PROFILE_SCHEMA_VERSION = "5A.3"
LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS = ("5A.1", "5A.2")
SUPPORTED_CONTENT_PROFILE_SCHEMA_VERSIONS = frozenset(
    (*LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS, CONTENT_PROFILE_SCHEMA_VERSION)
)

AUTO_PROFILE_INPUT = "auto"
UNKNOWN_PROFILE_ID = "unknown"
PROFILE_AXIS_ORDER = ("format", "editorial_mode", "domain", "traits")

ProfileAxisId = Literal["format", "editorial_mode", "domain", "traits"]


@dataclass(frozen=True, slots=True)
class ProfileTaxonomyValue:
    id: str
    label: str
    user_overridable: bool = True


@dataclass(frozen=True, slots=True)
class ProfileTaxonomyAxis:
    id: ProfileAxisId
    values: tuple[ProfileTaxonomyValue, ...]
    unknown_fallback: str | tuple[()]
    multiple: bool = False


@dataclass(frozen=True, slots=True)
class ContentProfilePreset:
    """Stable user-facing shortcut over the canonical profile axes."""

    id: str
    label: str
    format: str
    editorial_mode: str
    domain: str
    traits: tuple[str, ...]

    def profile(self) -> dict[str, str | list[str]]:
        return {
            "format": self.format,
            "editorial_mode": self.editorial_mode,
            "domain": self.domain,
            "traits": list(order_profile_ids("traits", list(self.traits))),
        }


def _value(value_id: str, label: str, *, user_overridable: bool = True) -> ProfileTaxonomyValue:
    return ProfileTaxonomyValue(value_id, label, user_overridable)


PROFILE_TAXONOMY = MappingProxyType({
    "format": ProfileTaxonomyAxis(
        "format",
        (
            _value("talking_head", "Говорящая голова"),
            _value("dialogue", "Диалог"),
            _value("screen_demo", "Демонстрация экрана"),
            _value("gameplay", "Геймплей"),
            _value("scene_driven", "Сценовое видео"),
            _value("mixed", "Смешанный"),
            _value(UNKNOWN_PROFILE_ID, "Не определено", user_overridable=False),
        ),
        UNKNOWN_PROFILE_ID,
    ),
    "editorial_mode": ProfileTaxonomyAxis(
        "editorial_mode",
        (
            _value("explanatory", "Объяснение"),
            _value("interview", "Интервью"),
            _value("commentary", "Комментарий"),
            _value("motivational", "Мотивация"),
            _value("narrative", "История"),
            _value("demonstration", "Демонстрация"),
            _value("entertainment", "Развлечение"),
            _value("news_analysis", "Новости / аналитика"),
            _value(UNKNOWN_PROFILE_ID, "Не определено", user_overridable=False),
        ),
        UNKNOWN_PROFILE_ID,
    ),
    "domain": ProfileTaxonomyAxis(
        "domain",
        (
            _value("business", "Бизнес"),
            _value("technology", "Технологии"),
            _value("education", "Образование"),
            _value("gaming", "Игры"),
            _value("food", "Еда"),
            _value("health", "Здоровье"),
            _value("finance", "Финансы"),
            _value("lifestyle", "Лайфстайл"),
            _value("entertainment", "Развлечения"),
            _value("news", "Новости"),
            _value("general", "Общее"),
            _value(UNKNOWN_PROFILE_ID, "Не определено", user_overridable=False),
        ),
        UNKNOWN_PROFILE_ID,
    ),
    "traits": ProfileTaxonomyAxis(
        "traits",
        (
            _value("speech_led", "Ведёт речь"),
            _value("visual_led", "Ведёт визуал"),
            _value("single_speaker", "Один спикер"),
            _value("multi_speaker", "Несколько спикеров"),
            _value("question_answer", "Вопрос — ответ"),
            _value("high_pacing", "Высокий темп"),
            _value("low_pacing", "Спокойный темп"),
            _value("high_emotion", "Высокая эмоциональность"),
            _value("dense_information", "Плотная информация"),
            _value("repetitive", "Повторы"),
            _value("screen_content", "Экранный контент"),
            _value("scene_driven", "Ведут сцены"),
            _value("instructional", "Обучающий"),
        ),
        (),
        multiple=True,
    ),
})


CONTENT_PROFILE_PRESETS = MappingProxyType({
    item.id: item for item in (
        ContentProfilePreset(
            "podcast", "Подкаст", "dialogue", "commentary", "general",
            ("speech_led", "multi_speaker", "low_pacing"),
        ),
        ContentProfilePreset(
            "interview", "Интервью", "dialogue", "interview", "general",
            ("speech_led", "multi_speaker", "question_answer"),
        ),
        ContentProfilePreset(
            "talking_head_expert", "Эксперт в кадре", "talking_head", "explanatory", "education",
            ("speech_led", "single_speaker", "dense_information"),
        ),
        ContentProfilePreset(
            "gameplay", "Геймплей", "gameplay", "commentary", "gaming",
            ("visual_led", "scene_driven", "high_pacing"),
        ),
        ContentProfilePreset(
            "stream", "Стрим", "mixed", "commentary", "entertainment",
            ("speech_led", "visual_led", "high_pacing"),
        ),
        ContentProfilePreset(
            "vlog_lifestyle", "Влог / лайфстайл", "scene_driven", "narrative", "lifestyle",
            ("visual_led", "scene_driven"),
        ),
        ContentProfilePreset(
            "food", "Еда", "scene_driven", "demonstration", "food",
            ("visual_led", "scene_driven", "instructional"),
        ),
        ContentProfilePreset(
            "travel", "Путешествия", "scene_driven", "narrative", "lifestyle",
            ("visual_led", "scene_driven"),
        ),
        ContentProfilePreset(
            "tutorial_education", "Обучение / туториал", "screen_demo", "demonstration", "education",
            ("speech_led", "visual_led", "dense_information", "screen_content", "instructional"),
        ),
        ContentProfilePreset(
            "review", "Обзор", "mixed", "commentary", "general",
            ("speech_led", "visual_led", "dense_information"),
        ),
        ContentProfilePreset(
            "reaction", "Реакция", "mixed", "commentary", "entertainment",
            ("speech_led", "visual_led", "high_emotion"),
        ),
        ContentProfilePreset(
            "story_entertainment", "История / развлечение", "scene_driven", "narrative", "entertainment",
            ("visual_led", "scene_driven", "high_emotion"),
        ),
        ContentProfilePreset(
            "movie_series", "Фильм / сериал", "scene_driven", "entertainment", "entertainment",
            ("visual_led", "scene_driven"),
        ),
        ContentProfilePreset(
            "sports_fitness", "Спорт / фитнес", "scene_driven", "demonstration", "health",
            ("visual_led", "scene_driven", "high_pacing", "instructional"),
        ),
        ContentProfilePreset(
            "news_commentary", "Новости / комментарии", "talking_head", "news_analysis", "news",
            ("speech_led", "single_speaker", "dense_information"),
        ),
    )
})


def content_profile_preset_ids(*, include_auto: bool = False) -> tuple[str, ...]:
    preset_ids = tuple(CONTENT_PROFILE_PRESETS)
    return (AUTO_PROFILE_INPUT, *preset_ids) if include_auto else preset_ids


def content_profile_preset(preset_id: str) -> ContentProfilePreset:
    try:
        return CONTENT_PROFILE_PRESETS[preset_id]
    except KeyError as error:
        raise ValueError(f"Unsupported content profile preset: {preset_id!r}") from error


def content_profile_preset_mapping(preset_id: str) -> dict[str, str | list[str]]:
    """Return a fresh deterministic override payload for a manual preset."""

    return content_profile_preset(preset_id).profile()


def content_profile_preset_id_for_mapping(profile: dict[str, object]) -> str | None:
    """Return the canonical preset whose complete axes exactly match ``profile``.

    Partial/manual axis overrides deliberately return ``None``.  Consumers must
    not guess a user-facing profile ID from an incomplete request.
    """

    try:
        normalized = {
            "format": str(profile.get("format") or ""),
            "editorial_mode": str(profile.get("editorial_mode") or ""),
            "domain": str(profile.get("domain") or ""),
            "traits": list(order_profile_ids("traits", [str(item) for item in profile.get("traits", [])])),
        }
    except (TypeError, ValueError):
        return None
    return next(
        (preset_id for preset_id, preset in CONTENT_PROFILE_PRESETS.items() if preset.profile() == normalized),
        None,
    )


def taxonomy_axis(axis_id: ProfileAxisId) -> ProfileTaxonomyAxis:
    return PROFILE_TAXONOMY[axis_id]


def profile_value_ids(axis_id: ProfileAxisId) -> tuple[str, ...]:
    """Return canonical stored IDs in their contract order."""

    return tuple(item.id for item in taxonomy_axis(axis_id).values)


def user_overridable_values(axis_id: ProfileAxisId) -> tuple[ProfileTaxonomyValue, ...]:
    """Return values that may be explicitly selected by a user."""

    return tuple(item for item in taxonomy_axis(axis_id).values if item.user_overridable)


def user_override_ids(axis_id: ProfileAxisId) -> tuple[str, ...]:
    return tuple(item.id for item in user_overridable_values(axis_id))


def order_profile_ids(axis_id: ProfileAxisId, value_ids: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    """Return unique IDs in canonical taxonomy order.

    Validation remains the consumer's responsibility so invalid input is never
    silently discarded.
    """

    requested = set(value_ids)
    ordered = tuple(value_id for value_id in profile_value_ids(axis_id) if value_id in requested)
    unknown = requested - set(ordered)
    if unknown:
        raise ValueError(f"Unsupported {axis_id} profile IDs: {sorted(unknown)!r}")
    return ordered


def profile_input_ids(axis_id: ProfileAxisId) -> tuple[str, ...]:
    """Return accepted single-choice input IDs, including the input-only sentinel."""

    return (AUTO_PROFILE_INPUT, *user_override_ids(axis_id))


def unknown_fallback(axis_id: ProfileAxisId) -> str | tuple[()]:
    return taxonomy_axis(axis_id).unknown_fallback
