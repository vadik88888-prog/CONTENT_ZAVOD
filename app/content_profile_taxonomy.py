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


CONTENT_PROFILE_SCHEMA_VERSION = "5A.2"
LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS = ("5A.1",)
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
