from __future__ import annotations

"""Phase 7C semantic caption planning and deterministic Tier 1 ASS output.

The planner is intentionally downstream of :class:`CreativeIntent`.  It only
turns verified intent and transcript evidence into bounded immutable caption
primitives; it never accepts raw ASS or FFmpeg expressions from the Brain.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Literal, Sequence

from app.config import ProductionRenderConfig
from app.creative_contracts import (
    BeatRole,
    CaptionCollisionDecision,
    CaptionCuePlan,
    CaptionEmphasisPlan,
    CaptionFeasibilityDecision,
    CaptionFeasibilityEvidence,
    CaptionFontManifest,
    CaptionPlan,
    CaptionQualityFinding,
    CaptionQualityMetrics,
    CaptionQualityProvenance,
    CaptionQualityReport,
    CaptionTypographyToken,
    CaptionWordPlan,
    CAPTION_PLAN_SCHEMA_VERSION,
    CompositionPlan,
    CreativeIntent,
    Intensity,
    MotionDomain,
    NormalizedRect,
    OutputInterval,
    ResolvedBeat,
    ResolvedEmphasis,
    ResolvedMotionEvent,
    SemanticClass,
    SourceInterval,
    SourceOutputTimeMap,
    canonical_hash,
)
from app.production_subtitles import resolve_subtitle_style
from app.utils import write_bytes_atomic
from app.video_models import SubtitleStyle


CAPTION_PLANNER_VERSION = "7J.2A-3.caption-temporal-feasibility.1"
CAPTION_FEASIBILITY_VERSION = "7J.2A-3.caption-feasibility.1"
CAPTION_HARD_CPS_CEILING = 20.0
LIBASS_BACKEND_VERSION = "tier1-libass-7B"
CAPTION_TARGET_CPS_MIN = 13.0
CAPTION_TARGET_CPS_MAX = 17.0
SEMANTIC_PRESENTATION_MIN_CONFIDENCE = 0.70
MAX_SEMANTIC_EMPHASIS_WORDS = 2

_SAFE_INSETS: dict[str, tuple[float, float, float, float]] = {
    "universal": (0.03, 0.03, 0.03, 0.03),
    "tiktok": (0.06, 0.05, 0.12, 0.08),
    "reels": (0.06, 0.05, 0.08, 0.08),
    "shorts": (0.05, 0.05, 0.05, 0.07),
}
_LANES: tuple[Literal["lower", "lower_mid", "upper_mid", "upper"], ...] = (
    "lower", "lower_mid", "upper_mid", "upper",
)
_LANE_CENTERS = {"lower": 0.82, "lower_mid": 0.63, "upper_mid": 0.38, "upper": 0.18}
_NO_BREAK_AFTER = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with",
    "в", "во", "и", "к", "ко", "на", "о", "об", "от", "по", "с", "со", "у", "за", "из", "не",
})
_STYLE_BY_FAMILY = {"clean": "clean", "emphasis": "dynamic", "minimal": "minimal", "editorial": "documentary"}
_FALLBACK_FONTS = ("Arial", "DejaVu Sans", "Noto Sans")
_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
_NORMALISE_RE = re.compile(r"[^\w]+", re.UNICODE)
_QT_APPLICATION: Any | None = None

CaptionLane = Literal["lower", "lower_mid", "upper_mid", "upper"]
CaptionFallbackReason = Literal[
    "weak_timing", "missing_font", "metrics_unavailable", "readability",
    "collision", "unsupported_primitive",
]
CollisionReason = Literal[
    "preferred_lane", "protected_region_avoidance", "platform_safe_zone",
    "stable_lane", "least_overlap_fallback",
]


@dataclass(frozen=True, slots=True)
class CaptionWordInput:
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float = 1.0
    timing_source: Literal["verified", "aligned", "phrase", "estimated"] = "verified"


@dataclass(frozen=True, slots=True)
class CaptionProtectedRegion:
    region_id: str
    output: OutputInterval
    bounds: NormalizedRect
    kind: Literal["face", "object", "screen", "text", "overlay"] = "object"
    importance: float = 1.0
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class _MappedWord:
    word_id: str
    text: str
    output: OutputInterval
    timing_source: Literal["verified", "aligned", "phrase", "estimated"]
    confidence: float
    source: SourceInterval
    map_ids: tuple[str, ...]

    @property
    def map_id(self) -> str:
        return self.map_ids[0]


@dataclass(frozen=True, slots=True)
class _Layout:
    words: tuple[_MappedWord, ...]
    lines: tuple[str, ...]
    font_size: int
    fallback: bool


@dataclass(frozen=True, slots=True)
class _SemanticEvent:
    kind: Literal["emphasis", "beat", "motion"]
    output: OutputInterval
    confidence: float
    importance: float
    evidence_refs: tuple[str, ...]
    emphasis: ResolvedEmphasis | None = None
    beat: ResolvedBeat | None = None
    motion: ResolvedMotionEvent | None = None


class _FontMeasurer:
    def __init__(self, manifest: CaptionFontManifest, font_path: Path | None, style: SubtitleStyle) -> None:
        self.manifest = manifest
        self.font_path = font_path
        self.style = style
        self.used_heuristic = False
        if os.name == "nt" and self.font_path is not None:
            try:
                import ctypes

                ctypes.windll.gdi32.AddFontResourceExW(str(self.font_path), 0x10, 0)
            except (AttributeError, OSError):
                pass

    def width(self, text: str, pixel_size: int) -> float:
        global _QT_APPLICATION
        try:
            if os.name == "nt":
                return _gdi_text_width(
                    text, self.manifest.resolved_family, pixel_size,
                    bold=self.style.font_weight == "bold",
                )
            from PySide6.QtCore import QCoreApplication
            from PySide6.QtGui import QFont, QFontDatabase, QFontMetricsF, QGuiApplication

            application = QGuiApplication.instance()
            if application is None:
                if QCoreApplication.instance() is not None:
                    raise RuntimeError("non-GUI Qt application is active")
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
                _QT_APPLICATION = QGuiApplication([])
            if self.font_path is not None:
                QFontDatabase.addApplicationFont(str(self.font_path))
            font = QFont(self.manifest.resolved_family)
            font.setPixelSize(pixel_size)
            font.setBold(self.style.font_weight == "bold")
            return float(QFontMetricsF(font).horizontalAdvance(text))
        except Exception:
            self.used_heuristic = True
            return sum(
                (1.02 if character.isupper() else 0.91 if ord(character) > 127 else 0.72) * pixel_size
                for character in text
            )


class CaptionPlanner:
    """Compile immutable semantic captions for the qualified libass backend."""

    def __init__(self, config: ProductionRenderConfig) -> None:
        self.config = config

    def plan(
        self,
        intent: CreativeIntent,
        transcript: dict[str, Any] | Sequence[CaptionWordInput | dict[str, Any]],
        *,
        composition_plan: CompositionPlan | None = None,
        protected_regions: Iterable[CaptionProtectedRegion] = (),
        font_manifest: CaptionFontManifest | None = None,
    ) -> CaptionPlan:
        platform = intent.policy.platform
        manifest, font_path = _resolve_font_manifest(
            self.config.subtitle_font_family,
            weight="bold" if intent.policy.caption_style_family != "minimal" else "normal",
            supplied=font_manifest,
        )
        style = _caption_style(intent, self.config, manifest)
        measurer = _FontMeasurer(manifest, font_path, style)
        typography = _typography(intent, self.config, style)
        words, input_diagnostics = _mapped_words(intent, transcript)
        regions = (*_composition_regions(composition_plan), *tuple(protected_regions))
        diagnostics = list(input_diagnostics)
        if not words:
            feasibility = _caption_feasibility_not_applicable(intent)
            findings: tuple[CaptionQualityFinding, ...] = (
                CaptionQualityFinding(
                    code="CAPTION_TIMING_WEAK", severity="warning", measured_value="no_mapped_words",
                    threshold="mapped phrase timing", message="No transcript words map into the approved output timeline.",
                ),
            )
            report = _quality_report((), findings, manifest, measurer, intent)
            return CaptionPlan(
                schema_version=CAPTION_PLAN_SCHEMA_VERSION,
                intent_id=intent.intent_id, backend_id="libass", intensity=intent.policy.intensity,
                font_manifest=manifest, typography=typography,
                feasibility_decision=feasibility, quality_report=report,
                diagnostics=tuple((*diagnostics, "NO_MAPPED_CAPTION_WORDS")),
            )

        base_size = max(12, round(style.font_size * min(self.config.output_width / 1080, self.config.output_height / 1920)))
        minimum_size = max(8, round(base_size * self.config.subtitle_min_font_scale))
        max_width = _safe_caption_width(self.config, platform, typography)
        layouts: list[_Layout] = []
        protected_phrases = tuple(
            tuple(_normalise(part) for part in emphasis.text_span.split() if _normalise(part))
            for emphasis in intent.semantic_emphasis
        )
        for run in _word_runs(words, intent):
            for group in _cue_groups(run, self.config, intent):
                layouts.extend(_fit_layout(
                    group, measurer, base_size, minimum_size, max_width,
                    protected_phrases,
                ))

        layouts, readability_coalesced = _coalesce_layouts_for_readability(
            layouts,
            intent,
            self.config,
            measurer,
            base_size,
            minimum_size,
            max_width,
            protected_phrases,
        )
        cue_outputs, readability_extended = _cue_outputs(layouts, intent, self.config)
        cues: list[CaptionCuePlan] = []
        previous_lane: CaptionLane | None = None
        last_motion_frame = -100_000
        intensity_degraded: set[str] = set()
        for index, (layout, output) in enumerate(zip(layouts, cue_outputs), start=1):
            timing_mode, timing_confidence = _timing_mode(layout.words)
            semantic_output = _words_output(layout.words)
            event = _semantic_event(intent, layout.words, semantic_output, last_motion_frame)
            emphasis_event = _semantic_emphasis_event(intent, layout.words, semantic_output)
            primitive = _primitive(intent, timing_mode, event)
            if event is not None and primitive != "static":
                last_motion_frame = output.start_frame
            elif event is not None and timing_mode != "word":
                intensity_degraded.add(f"caption-{index:03d}")
            emphasis = _emphasis_plan(
                emphasis_event, layout.words, output, primitive, timing_mode,
            )
            bounds_by_lane: dict[CaptionLane, NormalizedRect] = {
                lane: _caption_bounds(layout, lane, measurer, self.config, typography, platform)
                for lane in _LANES
            }
            lane, bounds, collision = _resolve_lane(
                output, bounds_by_lane, regions, previous_lane, platform,
            )
            cue_id = f"caption-{index:03d}"
            words_plan = tuple(CaptionWordPlan(
                word_id=word.word_id, text=word.text, output=word.output,
                timing_source=word.timing_source, confidence=word.confidence,
                source=word.source, mapping_segment_ids=word.map_ids,
            ) for word in layout.words)
            fallback_reason: CaptionFallbackReason | None = None
            if timing_mode != "word":
                fallback_reason = "weak_timing"
            if manifest.fallback_used:
                fallback_reason = fallback_reason or "missing_font"
            if measurer.used_heuristic:
                fallback_reason = fallback_reason or "metrics_unavailable"
            if layout.fallback:
                fallback_reason = fallback_reason or "readability"
            if collision.overlap_ratio > 0:
                fallback_reason = fallback_reason or "collision"
            scale_percent = 96 if primitive == "scale" else 100
            slide_distance = 0.025 if primitive == "slide" else 0.0
            cues.append(CaptionCuePlan(
                cue_id=cue_id,
                output=output,
                resolved_lines=layout.lines,
                lane=lane,
                typography_token_id=typography.token_id,
                semantic_class=emphasis.semantic_class if emphasis is not None else None,
                evidence_refs=tuple(dict.fromkeys((
                    *(event.evidence_refs if event is not None else ()),
                    *(emphasis_event.evidence_refs if emphasis_event is not None else ()),
                ))),
                primitive_id=primitive,
                easing_id="ease_in_out" if primitive in {"fade", "scale", "slide"} else "linear" if primitive == "karaoke" else "none",
                normalized_bounds=bounds,
                words=words_plan,
                timing_mode=timing_mode,
                timing_confidence=timing_confidence,
                emphasis=emphasis,
                beat_role=event.beat.role if event is not None and event.beat is not None else None,
                collision=collision,
                resolved_font_size_ratio=layout.font_size / self.config.output_height,
                motion_duration_frames=min(8, max(3, round((output.end_frame - output.start_frame) * 0.16))) if primitive in {"fade", "scale", "slide"} else 0,
                scale_percent=scale_percent,
                slide_distance_ratio=slide_distance,
                fallback_reason=fallback_reason,
            ))
            previous_lane = lane

        if measurer.used_heuristic and manifest.metrics_backend != "heuristic":
            manifest = manifest.model_copy(update={"metrics_backend": "heuristic"})
        feasibility = _caption_feasibility_decision(
            words, cues, intent, self.config, measurer, minimum_size, max_width,
            protected_phrases,
        )
        findings = _assess_plan(
            cues, manifest, measurer, intent, self.config, intensity_degraded,
            feasibility,
        )
        report = _quality_report(cues, findings, manifest, measurer, intent)
        diagnostics.extend(_diagnostics(cues, manifest, measurer))
        if readability_coalesced:
            diagnostics.append("CAPTION_READABILITY_COALESCED")
        if readability_extended:
            diagnostics.append("CAPTION_PRESENTATION_WINDOW_EXTENDED")
        return CaptionPlan(
            schema_version=CAPTION_PLAN_SCHEMA_VERSION,
            intent_id=intent.intent_id,
            cues=tuple(cues),
            backend_id="libass",
            intensity=intent.policy.intensity,
            font_manifest=manifest,
            typography=typography,
            feasibility_decision=feasibility,
            quality_report=report,
            diagnostics=tuple(dict.fromkeys(diagnostics)),
        )


def build_caption_plan(
    intent: CreativeIntent,
    transcript: dict[str, Any] | Sequence[CaptionWordInput | dict[str, Any]],
    config: ProductionRenderConfig,
    *,
    composition_plan: CompositionPlan | None = None,
    protected_regions: Iterable[CaptionProtectedRegion] = (),
    font_manifest: CaptionFontManifest | None = None,
) -> CaptionPlan:
    return CaptionPlanner(config).plan(
        intent, transcript, composition_plan=composition_plan,
        protected_regions=protected_regions, font_manifest=font_manifest,
    )


def resolve_caption_font_manifest(
    requested_family: str, *, weight: Literal["normal", "bold"] = "bold",
) -> CaptionFontManifest:
    """Resolve a family to an exact, checksummed system font when possible."""

    manifest, _path = _resolve_font_manifest(requested_family, weight=weight, supplied=None)
    return manifest


def _resolve_font_manifest(
    requested_family: str,
    *,
    weight: Literal["normal", "bold"],
    supplied: CaptionFontManifest | None,
) -> tuple[CaptionFontManifest, Path | None]:
    if supplied is not None:
        path = _find_manifest_file(supplied)
        if path is not None:
            return supplied, path
        # A persisted manifest from another host is not silently trusted by
        # family name. Re-resolve its request into a new deterministic plan.
        requested_family = supplied.requested_family
    requested = requested_family.strip() or "Arial"
    unsafe = any(character in requested for character in "\\/\x00")
    families = (*(() if unsafe else (requested,)), *tuple(item for item in _FALLBACK_FONTS if item.casefold() != requested.casefold()))
    for family in families:
        path = _find_font_file(family, weight)
        if path is None:
            continue
        checksum = _file_sha256(path)
        actual_family, scripts = _font_details(path, family)
        fallback = unsafe or actual_family.casefold() != requested.casefold()
        return CaptionFontManifest(
            font_id=f"font-{checksum[:24]}", requested_family=requested,
            resolved_family=actual_family, weight=weight, file_name=path.name,
            file_sha256=checksum, supported_scripts=scripts,
            metrics_backend="gdi_file_metrics" if os.name == "nt" else "qt_file_metrics",
            shaping_backend="libass-harfbuzz",
            fallback_chain=_FALLBACK_FONTS, deployment_status="system", fallback_used=fallback,
        ), path
    identity = canonical_hash({"family": "Arial", "weight": weight, "status": "unverified"})
    return CaptionFontManifest(
        font_id=f"font-{identity[:24]}", requested_family=requested, resolved_family="Arial",
        weight=weight, supported_scripts=("unknown",), metrics_backend="heuristic",
        shaping_backend="libass-harfbuzz", fallback_chain=_FALLBACK_FONTS,
        deployment_status="unverified", fallback_used=True,
    ), None


def _find_font_file(family: str, weight: str) -> Path | None:
    matcher = shutil.which("fc-match")
    if matcher:
        try:
            pattern = f"{family}:style={'Bold' if weight == 'bold' else 'Regular'}"
            value = subprocess.run(
                [matcher, "-f", "%{file}", pattern], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=5, check=True,
            ).stdout.strip()
            if value and Path(value).is_file():
                return Path(value).resolve()
        except (OSError, subprocess.SubprocessError):
            pass
    if os.name == "nt":
        try:
            import winreg

            key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                entries: list[tuple[str, str]] = []
                index = 0
                while True:
                    try:
                        name, value, _kind = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    entries.append((str(name), str(value)))
                    index += 1
            needle = family.casefold()
            ranked = sorted(entries, key=lambda item: _font_registry_rank(item[0], needle, weight))
            for display_name, value in ranked:
                if needle not in display_name.casefold():
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / value
                if path.is_file():
                    return path.resolve()
        except (OSError, ImportError):
            pass
    compact = re.sub(r"[^a-z0-9]", "", family.casefold())
    roots = _font_roots()
    candidates = sorted(
        (path for root in roots if root.is_dir() for path in root.iterdir() if path.suffix.casefold() in {".ttf", ".otf", ".ttc"}),
        key=lambda path: path.name.casefold(),
    )
    for path in candidates:
        filename = re.sub(r"[^a-z0-9]", "", path.stem.casefold())
        if compact and (filename.startswith(compact) or compact.startswith(filename)):
            return path.resolve()
    return None


def _font_registry_rank(name: str, needle: str, weight: str) -> tuple[int, int, str]:
    value = name.casefold()
    exact = 0 if value.startswith(needle + " (") else 1
    wants_bold = weight == "bold"
    weight_mismatch = 0 if ("bold" in value) == wants_bold else 1
    italic = 1 if "italic" in value else 0
    return exact + weight_mismatch, italic, value


def _font_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    if os.name == "nt":
        roots.append(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    else:
        roots.extend((Path("/usr/share/fonts"), Path("/usr/local/share/fonts")))
    return tuple(roots)


def _font_details(path: Path, fallback_family: str) -> tuple[str, tuple[Literal["latin", "cyrillic", "unknown"], ...]]:
    # QRawFont(file) can access-violate on Windows after a QApplication used by
    # another test/window has been torn down.  File identity is established by
    # SHA-256; the exact file is registered later for QFontMetricsF.  Script
    # coverage remains conservative unless the approved launch family is known.
    del path
    known_multiscript = {"arial", "dejavu sans", "noto sans"}
    scripts: tuple[Literal["latin", "cyrillic", "unknown"], ...] = (
        ("latin", "cyrillic") if fallback_family.casefold() in known_multiscript else ("unknown",)
    )
    return fallback_family, scripts


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gdi_text_width(text: str, family: str, pixel_size: int, *, bold: bool) -> float:
    """Measure shaped Windows text without relying on QApplication lifetime."""

    import ctypes
    from ctypes import wintypes

    class Size(ctypes.Structure):
        _fields_ = (("cx", wintypes.LONG), ("cy", wintypes.LONG))

    gdi = ctypes.windll.gdi32
    gdi.CreateCompatibleDC.restype = ctypes.c_void_p
    gdi.CreateFontW.restype = ctypes.c_void_p
    gdi.SelectObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    gdi.SelectObject.restype = ctypes.c_void_p
    gdi.GetTextExtentPoint32W.argtypes = (
        ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(Size),
    )
    gdi.GetTextExtentPoint32W.restype = wintypes.BOOL
    gdi.DeleteObject.argtypes = (ctypes.c_void_p,)
    gdi.DeleteDC.argtypes = (ctypes.c_void_p,)
    hdc = gdi.CreateCompatibleDC(0)
    if not hdc:
        raise OSError("CreateCompatibleDC failed")
    font = gdi.CreateFontW(
        -int(pixel_size), 0, 0, 0, 700 if bold else 400,
        0, 0, 0, 1, 0, 0, 5, 0, family,
    )
    if not font:
        gdi.DeleteDC(hdc)
        raise OSError("CreateFontW failed")
    previous = gdi.SelectObject(hdc, font)
    size = Size()
    try:
        if not gdi.GetTextExtentPoint32W(hdc, text, len(text), ctypes.byref(size)):
            raise OSError("GetTextExtentPoint32W failed")
        return float(size.cx)
    finally:
        gdi.SelectObject(hdc, previous)
        gdi.DeleteObject(font)
        gdi.DeleteDC(hdc)


def _find_manifest_file(manifest: CaptionFontManifest) -> Path | None:
    if manifest.file_name is None or manifest.file_sha256 is None:
        return None
    for root in _font_roots():
        candidate = root / manifest.file_name
        if candidate.is_file() and _file_sha256(candidate) == manifest.file_sha256:
            return candidate.resolve()
        if root.is_dir():
            for nested in root.rglob(manifest.file_name):
                if nested.is_file() and _file_sha256(nested) == manifest.file_sha256:
                    return nested.resolve()
    return None


def _caption_style(intent: CreativeIntent, config: ProductionRenderConfig, manifest: CaptionFontManifest) -> SubtitleStyle:
    style_id = _STYLE_BY_FAMILY[intent.policy.caption_style_family]
    resolved = replace(config, subtitle_style=style_id, subtitle_font_family=manifest.resolved_family)
    style, _fallback, _warning = resolve_subtitle_style(resolved)
    return style.model_copy(update={"uppercase": False})


def _typography(intent: CreativeIntent, config: ProductionRenderConfig, style: SubtitleStyle) -> CaptionTypographyToken:
    return CaptionTypographyToken(
        token_id=f"caption-{intent.policy.caption_style_family}-{intent.policy.intensity.value}-v1",
        font_size_ratio=style.font_size / 1920,
        minimum_font_size_ratio=(style.font_size / 1920) * config.subtitle_min_font_scale,
        font_weight=style.font_weight,
        text_color=style.text_color,
        highlight_color=style.highlight_color,
        outline_color=style.outline_color,
        outline_width_ratio=style.outline_width / 1920,
        shadow_ratio=style.shadow / 1920,
        max_width_ratio=config.subtitle_max_rendered_width_ratio,
        alignment=style.alignment,
        uppercase_emphasis=intent.policy.caption_style_family == "emphasis",
    )


def _mapped_words(
    intent: CreativeIntent,
    transcript: dict[str, Any] | Sequence[CaptionWordInput | dict[str, Any]],
) -> tuple[list[_MappedWord], list[str]]:
    raw_words, diagnostics = _word_inputs(transcript, intent)
    result: list[_MappedWord] = []
    for index, word in enumerate(raw_words, start=1):
        if not word.text.strip() or not math.isfinite(word.start_seconds) or not math.isfinite(word.end_seconds):
            diagnostics.append("INVALID_WORD_TIMING_DROPPED")
            continue
        if word.end_seconds <= word.start_seconds:
            diagnostics.append("INVALID_WORD_TIMING_DROPPED")
            continue
        source = SourceInterval.from_seconds(max(0.0, word.start_seconds), max(0.0, word.end_seconds))
        mapped = intent.source_output_mapping.map_continuous_interval(source)
        if mapped is None:
            diagnostics.append("UNMAPPED_WORD_DROPPED")
            continue
        output, map_ids = mapped
        result.append(_MappedWord(
            word_id=f"word-{index:05d}", text=word.text.strip(), output=output,
            timing_source=word.timing_source, confidence=max(0.0, min(1.0, word.confidence)),
            source=source, map_ids=map_ids,
        ))
    result.sort(key=lambda item: (item.output.start_frame, item.output.end_frame, item.word_id))
    return result, list(dict.fromkeys(diagnostics))


def _word_inputs(
    transcript: dict[str, Any] | Sequence[CaptionWordInput | dict[str, Any]],
    intent: CreativeIntent,
) -> tuple[list[CaptionWordInput], list[str]]:
    diagnostics: list[str] = []
    if not isinstance(transcript, dict):
        words = [_coerce_word(item) for item in transcript]
        return [item for item in words if item is not None], diagnostics
    top_words = transcript.get("words")
    if isinstance(top_words, list) and top_words:
        words = [_coerce_word(item) for item in top_words]
        return [item for item in words if item is not None], diagnostics
    segments = transcript.get("segments")
    phrase_words: list[CaptionWordInput] = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            nested = segment.get("words")
            if isinstance(nested, list) and nested:
                phrase_words.extend(item for item in (_coerce_word(value) for value in nested) if item is not None)
                continue
            text = str(segment.get("text") or "").strip()
            start = _float_value(segment, "start", "start_seconds")
            end = _float_value(segment, "end", "end_seconds")
            if text and start is not None and end is not None and end > start:
                phrase_words.extend(_distributed_phrase(text, start, end, "phrase", 0.60))
    if phrase_words:
        diagnostics.append("PHRASE_TIMING_FALLBACK")
        return phrase_words, diagnostics
    text = str(transcript.get("text") or "").strip()
    if text:
        first = intent.source_output_mapping.segments[0].source
        start = first.start_tick / 1_000_000
        end = first.end_tick / 1_000_000
        diagnostics.append("ESTIMATED_TIMING_FALLBACK")
        return _distributed_phrase(text, start, end, "estimated", 0.25), diagnostics
    return [], diagnostics


def _coerce_word(value: CaptionWordInput | dict[str, Any]) -> CaptionWordInput | None:
    if isinstance(value, CaptionWordInput):
        return value
    if not isinstance(value, dict):
        return None
    text = str(value.get("text") or value.get("word") or "").strip()
    start = _float_value(value, "start", "start_seconds")
    end = _float_value(value, "end", "end_seconds")
    if not text or start is None or end is None:
        return None
    source = str(value.get("timing_source") or "verified")
    if source not in {"verified", "aligned", "phrase", "estimated"}:
        source = "verified"
    raw_confidence = value.get("confidence")
    if raw_confidence is None:
        raw_confidence = value.get("probability", 1.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return CaptionWordInput(text, start, end, max(0.0, min(1.0, confidence)), source)  # type: ignore[arg-type]


def _float_value(value: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in value:
            try:
                return float(value[key])
            except (TypeError, ValueError):
                return None
    return None


def _distributed_phrase(
    text: str, start: float, end: float,
    timing_source: Literal["phrase", "estimated"], confidence: float,
) -> list[CaptionWordInput]:
    tokens = _TOKEN_RE.findall(text)
    if not tokens or end <= start:
        return []
    weights = [max(1.0, len(_normalise(token)) ** 0.58) for token in tokens]
    total = sum(weights)
    cursor = start
    result: list[CaptionWordInput] = []
    for index, (token, weight) in enumerate(zip(tokens, weights)):
        token_end = end if index == len(tokens) - 1 else cursor + (end - start) * weight / total
        result.append(CaptionWordInput(token, cursor, max(cursor + 0.001, token_end), confidence, timing_source))
        cursor = token_end
    return result


def _word_runs(words: list[_MappedWord], intent: CreativeIntent) -> list[tuple[_MappedWord, ...]]:
    runs: list[list[_MappedWord]] = []
    current: list[_MappedWord] = []
    for word in words:
        # Consecutive ProductionPlan dialogue mappings frequently divide one
        # sentence into evidence-sized fragments.  Their output clock is still
        # continuous, so treating every map_id change as a caption hard stop
        # can make an otherwise feasible CPS partition impossible.  A real
        # presentation gap remains the run boundary.
        if current and (
            word.output.start_frame - current[-1].output.end_frame > 18
            or not _map_ids_are_continuous(
                intent.source_output_mapping,
                current[-1].map_ids[-1],
                word.map_ids[0],
            )
        ):
            runs.append(current)
            current = []
        current.append(word)
    if current:
        runs.append(current)
    return [tuple(run) for run in runs]


def _cue_groups(
    words: tuple[_MappedWord, ...], config: ProductionRenderConfig, intent: CreativeIntent,
) -> list[tuple[_MappedWord, ...]]:
    # Partition the whole word run, rather than committing greedily at the
    # first punctuation mark.  A locally attractive split can leave the next
    # cue physically unable to meet CPS even though a neighbouring 3/9/6/6
    # partition is fully readable with the same frozen word timings.
    maximum_frames = round(config.subtitle_max_duration * 30)
    protected_phrases = tuple(
        tuple(_normalise(part) for part in emphasis.text_span.split() if _normalise(part))
        for emphasis in intent.semantic_emphasis
    )
    # end -> (hard-ceiling failures, crossed semantic boundaries, overlapping
    #         cue boundaries, target-CPS shortfall, cue count, negative
    #         boundary score, ranges)
    best: dict[int, tuple[int, int, int, int, int, float, tuple[tuple[int, int], ...]]] = {
        0: (0, 0, 0, 0, 0, 0.0, ()),
    }
    for end in range(1, len(words) + 1):
        if end < len(words) and _splits_protected_phrase(words, end, protected_phrases):
            continue
        for start in range(max(0, end - config.subtitle_max_words_per_cue), end):
            previous = best.get(start)
            if previous is None:
                continue
            size = end - start
            if size < config.subtitle_min_words_per_cue and end != len(words):
                continue
            raw = _words_output(words[start:end])
            if raw.end_frame - raw.start_frame > maximum_frames:
                continue
            slot_start = (
                _mapping_chain_bounds(intent, words[start].map_id)[0] if start == 0
                else words[start - 1].output.end_frame - 2
            )
            slot_end = (
                max(raw.end_frame, words[end].output.start_frame)
                if end < len(words)
                else _mapping_chain_bounds(intent, words[end - 1].map_id)[1]
            )
            readable = (
                _timing_mode(words[start:end])[0] != "word"
                or _minimum_caption_frames(
                    words[start:end], _maximum_caption_cps(words[start:end]),
                ) <= slot_end - slot_start
            )
            target_shortfall = max(
                0,
                _minimum_caption_frames(words[start:end], _target_caption_cps(config))
                - (slot_end - slot_start),
            )
            candidate = (
                previous[0] + (0 if readable else 1),
                previous[1] + _crossed_semantic_boundaries(words, start, end, intent),
                previous[2] + (
                    1 if end < len(words) and words[end].output.start_frame < words[end - 1].output.end_frame
                    else 0
                ),
                previous[3] + target_shortfall,
                previous[4] + 1,
                previous[5] - _cue_boundary_score(words, start, end, intent),
                (*previous[6], (start, end)),
            )
            current = best.get(end)
            if current is None or candidate[:6] < current[:6]:
                best[end] = candidate
    partition = best.get(len(words))
    if partition is None:
        return [words]
    return [words[start:end] for start, end in partition[6]]


def _crossed_semantic_boundaries(
    words: tuple[_MappedWord, ...], start: int, end: int, intent: CreativeIntent,
) -> int:
    if end - start < 2:
        return 0
    internal_splits = tuple(
        (words[index - 1].output.end_frame + words[index].output.start_frame) / 2
        for index in range(start + 1, end)
    )
    boundaries = {
        frame
        for event in (*intent.beats, *intent.semantic_emphasis)
        for frame in (event.output.start_frame, event.output.end_frame)
        if words[start].output.start_frame < frame < words[end - 1].output.end_frame
    }
    return sum(
        min(abs(split - boundary) for split in internal_splits) <= 3
        for boundary in boundaries
    )


def _cue_boundary_score(words: tuple[_MappedWord, ...], start: int, end: int, intent: CreativeIntent) -> float:
    if end >= len(words):
        return 1000.0
    previous, following = words[end - 1], words[end]
    score = 0.0
    if previous.text.rstrip().endswith((".", "!", "?")):
        score += 80
    elif previous.text.rstrip().endswith((",", ";", ":", "—", "-")):
        score += 45
    score += min(20, max(0, following.output.start_frame - previous.output.end_frame)) * 2
    score -= abs((end - start) - 5) * 0.7
    boundary = previous.output.end_frame
    for emphasis in intent.semantic_emphasis:
        if emphasis.output.start_frame < boundary < emphasis.output.end_frame:
            score -= 100 * emphasis.importance
    return score


def _fit_layout(
    words: tuple[_MappedWord, ...], measurer: _FontMeasurer,
    base_size: int, minimum_size: int, maximum_width: float,
    protected_phrases: tuple[tuple[str, ...], ...],
) -> list[_Layout]:
    sizes = list(dict.fromkeys((base_size, round(base_size * 0.95), round(base_size * 0.90), minimum_size)))
    sizes = [max(minimum_size, value) for value in sizes]
    for size in sizes:
        lines = _best_lines(words, measurer, size, maximum_width, protected_phrases)
        if lines is not None:
            return [_Layout(words, lines, size, size != base_size)]
    if len(words) > 1:
        split = _best_layout_split(words, protected_phrases)
        return [
            *_fit_layout(words[:split], measurer, base_size, minimum_size, maximum_width, protected_phrases),
            *_fit_layout(words[split:], measurer, base_size, minimum_size, maximum_width, protected_phrases),
        ]
    return [_Layout(words, (words[0].text,), minimum_size, True)]


def _best_lines(
    words: tuple[_MappedWord, ...], measurer: _FontMeasurer,
    size: int, maximum_width: float, protected_phrases: tuple[tuple[str, ...], ...],
) -> tuple[str, ...] | None:
    candidates: list[tuple[float, tuple[str, ...]]] = []
    full = " ".join(word.text for word in words)
    full_width = measurer.width(full, size)
    if full_width <= maximum_width:
        candidates.append((full_width / maximum_width * 0.15, (full,)))
    for split in range(1, len(words)):
        if _splits_protected_phrase(words, split, protected_phrases):
            continue
        left = " ".join(word.text for word in words[:split])
        right = " ".join(word.text for word in words[split:])
        left_width, right_width = measurer.width(left, size), measurer.width(right, size)
        if max(left_width, right_width) > maximum_width:
            continue
        penalty = abs(left_width - right_width) / maximum_width
        if _normalise(words[split - 1].text) in _NO_BREAK_AFTER:
            penalty += 4
        if split == 1 or len(words) - split == 1:
            penalty += 0.45
        candidates.append((penalty, (left, right)))
    return min(candidates, key=lambda item: (item[0], item[1]))[1] if candidates else None


def _best_layout_split(
    words: tuple[_MappedWord, ...], protected_phrases: tuple[tuple[str, ...], ...],
) -> int:
    center = len(words) / 2
    safe_choices = [
        index for index in range(1, len(words))
        if not _splits_protected_phrase(words, index, protected_phrases)
    ]
    choices: Iterable[int] = safe_choices or range(1, len(words))
    return max(
        choices,
        key=lambda index: (
            40 if words[index - 1].text.rstrip().endswith((".", "!", "?")) else
            20 if words[index - 1].text.rstrip().endswith((",", ";", ":")) else 0
        ) - abs(index - center),
    )


def _splits_protected_phrase(
    words: tuple[_MappedWord, ...], split: int, protected_phrases: tuple[tuple[str, ...], ...],
) -> bool:
    normalised = [_normalise(word.text) for word in words]
    for phrase in protected_phrases:
        start = _find_phrase(normalised, list(phrase))
        if start is not None and start < split < start + len(phrase):
            return True
    return False


def _safe_caption_width(
    config: ProductionRenderConfig, platform: str, typography: CaptionTypographyToken,
) -> float:
    left, _top, right, _bottom = _SAFE_INSETS.get(platform, _SAFE_INSETS["universal"])
    return config.output_width * min(typography.max_width_ratio, 1 - left - right)


def _coalesce_layouts_for_readability(
    layouts: list[_Layout],
    intent: CreativeIntent,
    config: ProductionRenderConfig,
    measurer: _FontMeasurer,
    base_size: int,
    minimum_size: int,
    maximum_width: float,
    protected_phrases: tuple[tuple[str, ...], ...],
) -> tuple[list[_Layout], set[int]]:
    """Find a readable cue partition without changing mapped word frames.

    A source-preserving edit can contain several very short adjacent dialogue
    segments.  Treating every segment as a standalone caption can exceed the
    hard CPS ceiling even when a neighbouring, semantically neutral phrase
    makes a readable two-line cue.  The dynamic partition below only combines
    truly contiguous output-map runs, respects the configured word/duration
    limits, and never combines two protected hook/payoff/emphasis/motion cues.
    One protected cue may absorb a neutral neighbour: its frozen word timing
    and semantic event remain intact while the combined reading window fixes
    an otherwise artificial local CPS spike. Adjacent Whisper word intervals
    commonly overlap by one frame; the two-frame boundary allowance below
    prevents that quantisation artifact from stealing readable cue time.
    """

    if not layouts:
        return layouts, set()
    result: list[_Layout] = []
    coalesced: set[int] = set()
    for run_start, run_end in _layout_run_ranges(layouts, intent):
        chain_start, chain_end = _mapping_chain_bounds(
            intent, layouts[run_start].words[0].map_id,
        )
        # Each entry is the highest-fidelity readable partition from this
        # source index to the end of the contiguous run.  More cues win so the
        # repair combines only what the hard ceiling actually requires.
        best: dict[int, tuple[tuple[_Layout, bool], ...]] = {run_end: ()}
        for start in range(run_end - 1, run_start - 1, -1):
            selected: tuple[tuple[_Layout, bool], ...] | None = None
            words: tuple[_MappedWord, ...] = ()
            for end in range(start + 1, run_end + 1):
                words = (*words, *layouts[end - 1].words)
                if len(words) > config.subtitle_max_words_per_cue:
                    break
                if end - start > 1 and sum(
                    _layout_has_protected_semantic_event(layouts[index], intent)
                    for index in range(start, end)
                ) > 1:
                    break
                raw = _words_output(words)
                if raw.end_frame - raw.start_frame > round(config.subtitle_max_duration * 30):
                    break
                if end - start == 1:
                    candidate = layouts[start]
                else:
                    fitted = _fit_layout(
                        words,
                        measurer,
                        base_size,
                        minimum_size,
                        maximum_width,
                        protected_phrases,
                    )
                    if len(fitted) != 1:
                        break
                    candidate = fitted[0]
                slot_start = (
                    chain_start if start == run_start
                    else max(
                        chain_start,
                        _words_output(layouts[start - 1].words).end_frame - 2,
                    )
                )
                slot_end = (
                    max(raw.end_frame, _words_output(layouts[end].words).start_frame)
                    if end < run_end else chain_end
                )
                timing_mode, _confidence = _timing_mode(words)
                readable = (
                    timing_mode != "word"
                    or _minimum_caption_frames(words, _maximum_caption_cps(words))
                    <= slot_end - slot_start
                )
                tail = best.get(end)
                if not readable or tail is None:
                    continue
                option = ((candidate, end - start > 1), *tail)
                if selected is None or len(option) > len(selected):
                    selected = option
            if selected is not None:
                best[start] = selected
        partition = best.get(run_start)
        if partition is None:
            partition = tuple((layout, False) for layout in layouts[run_start:run_end])
        for layout, was_coalesced in partition:
            output_index = len(result)
            result.append(layout)
            if was_coalesced:
                coalesced.add(output_index)
    return result, coalesced


def _cue_outputs(
    layouts: list[_Layout], intent: CreativeIntent, config: ProductionRenderConfig,
) -> tuple[list[OutputInterval], set[int]]:
    raw = [_words_output(layout.words) for layout in layouts]
    result = list(raw)
    extended: set[int] = set()
    for run_start, run_end in _layout_run_ranges(layouts, intent):
        chain_start, chain_end = _mapping_chain_bounds(
            intent, layouts[run_start].words[0].map_id,
        )
        for index in range(run_start, run_end):
            output = raw[index]
            slot_start = (
                chain_start if index == run_start
                else max(chain_start, raw[index - 1].end_frame - 2)
            )
            slot_end = (
                max(output.end_frame, raw[index + 1].start_frame) if index + 1 < run_end
                else chain_end
            )
            target_cps = _target_caption_cps(config)
            target_frames = _minimum_caption_frames(layouts[index].words, target_cps)
            maximum_frames = _minimum_caption_frames(
                layouts[index].words, _maximum_caption_cps(layouts[index].words),
            )
            available = min(
                slot_end - slot_start,
                round(config.subtitle_max_duration * 30),
            )
            requested = (
                target_frames if target_frames <= available
                else maximum_frames if maximum_frames <= available
                else output.end_frame - output.start_frame
            )
            duration = max(output.end_frame - output.start_frame, requested)
            end = min(slot_end, max(output.end_frame, output.start_frame + duration))
            start = max(slot_start, min(output.start_frame, end - duration))
            resolved = OutputInterval(start_frame=start, end_frame=end)
            # The presentation window may grow into otherwise unused frames,
            # but the frozen word activation frames remain byte-for-byte exact.
            if not resolved.contains(output):
                resolved = output
            result[index] = resolved
            if resolved != output:
                extended.add(index)
    return result, extended


def _layout_run_ranges(
    layouts: list[_Layout], intent: CreativeIntent,
) -> list[tuple[int, int]]:
    if not layouts:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(layouts)):
        previous_id = layouts[index - 1].words[-1].map_ids[-1]
        current_id = layouts[index].words[0].map_ids[0]
        contiguous = _map_ids_are_continuous(
            intent.source_output_mapping, previous_id, current_id,
        )
        if not contiguous:
            ranges.append((start, index))
            start = index
    ranges.append((start, len(layouts)))
    return ranges


def _mapping_chain_bounds(intent: CreativeIntent, map_id: str) -> tuple[int, int]:
    mapping = intent.source_output_mapping.segments
    index = next(position for position, segment in enumerate(mapping) if segment.map_id == map_id)
    start = index
    end = index
    while start > 0 and intent.source_output_mapping.segments_are_continuous(
        mapping[start - 1], mapping[start],
    ):
        start -= 1
    while end + 1 < len(mapping) and intent.source_output_mapping.segments_are_continuous(
        mapping[end], mapping[end + 1],
    ):
        end += 1
    return mapping[start].output.start_frame, mapping[end].output.end_frame


def _map_ids_are_continuous(
    mapping: SourceOutputTimeMap, left_id: str, right_id: str,
) -> bool:
    positions = {segment.map_id: index for index, segment in enumerate(mapping.segments)}
    left = positions[left_id]
    right = positions[right_id]
    if right < left:
        return False
    return all(
        mapping.segments_are_continuous(
            mapping.segments[index], mapping.segments[index + 1],
        )
        for index in range(left, right)
    )


def _layout_has_protected_semantic_event(layout: _Layout, intent: CreativeIntent) -> bool:
    output = _words_output(layout.words)
    return (
        any(_overlaps(output, emphasis.output) for emphasis in intent.semantic_emphasis)
        or any(
            beat.role in {BeatRole.HOOK, BeatRole.PAYOFF} and _overlaps(output, beat.output)
            for beat in intent.beats
        )
        or any(
            motion.domain == MotionDomain.CAPTION and _overlaps(output, motion.output)
            for motion in intent.motion_events
        )
    )


def _words_output(words: tuple[_MappedWord, ...]) -> OutputInterval:
    return OutputInterval(
        start_frame=min(word.output.start_frame for word in words),
        end_frame=max(word.output.end_frame for word in words),
    )


def _maximum_caption_cps(words: tuple[_MappedWord, ...]) -> float:
    return CAPTION_HARD_CPS_CEILING


def _target_caption_cps(config: ProductionRenderConfig) -> float:
    return max(
        CAPTION_TARGET_CPS_MIN,
        min(CAPTION_TARGET_CPS_MAX, float(config.subtitle_reading_speed_cps)),
    )


def _minimum_caption_frames(words: tuple[_MappedWord, ...], cps: float) -> int:
    text = " ".join(word.text for word in words)
    return math.ceil(len(text.replace(" ", "")) / cps * 30)


def _caption_feasibility_not_applicable(intent: CreativeIntent) -> CaptionFeasibilityDecision:
    payload = {
        "intent_id": intent.intent_id,
        "version": CAPTION_FEASIBILITY_VERSION,
        "status": "NOT_APPLICABLE",
    }
    return CaptionFeasibilityDecision(
        decision_id=f"caption-feasibility-{canonical_hash(payload)[:24]}",
        status="NOT_APPLICABLE",
        reason_code="NO_MAPPED_WORDS",
    )


def _caption_feasibility_decision(
    words: list[_MappedWord],
    cues: list[CaptionCuePlan],
    intent: CreativeIntent,
    config: ProductionRenderConfig,
    measurer: _FontMeasurer,
    minimum_size: int,
    maximum_width: float,
    protected_phrases: tuple[tuple[str, ...], ...],
) -> CaptionFeasibilityDecision:
    """Exhaustively decide whether legal cue partitions can satisfy hard CPS.

    The search changes only cue partition, presentation window, and line break.
    Word identity and activation frames stay frozen; speech retiming and text
    rewriting are deliberately absent from the state space.
    """

    feasible = True
    evaluated = 0
    failed_windows: list[tuple[tuple[_MappedWord, ...], int, int]] = []
    for run in _word_runs(words, intent):
        run_feasible, run_evaluated, run_failed_windows = _caption_run_is_feasible(
            run,
            intent,
            config,
            measurer,
            minimum_size,
            maximum_width,
            protected_phrases,
        )
        feasible = feasible and run_feasible
        evaluated += run_evaluated
        if not run_feasible and run_failed_windows:
            failed_windows.extend(run_failed_windows)

    evidence: list[CaptionFeasibilityEvidence] = []
    if not feasible:
        for mapped, available, required in failed_windows:
            hard_cps = _maximum_caption_cps(mapped)
            characters = len("".join(word.text for word in mapped).replace(" ", ""))
            measured = characters / max(1 / 30, available / 30)
            if required <= available or measured <= hard_cps + 1e-9:
                continue
            source = SourceInterval(
                start_tick=min(word.source.start_tick for word in mapped),
                end_tick=max(word.source.end_tick for word in mapped),
            )
            map_ids = tuple(dict.fromkeys(
                map_id for word in mapped for map_id in word.map_ids
            ))
            evidence.append(CaptionFeasibilityEvidence(
                evidence_id=f"caption-cps-evidence-{len(evidence) + 1:03d}",
                word_ids=tuple(word.word_id for word in mapped),
                text=" ".join(word.text for word in mapped),
                source=source,
                immutable_word_output=_words_output(mapped),
                mapping_segment_ids=map_ids,
                character_count=characters,
                available_frames=available,
                required_frames=required,
                measured_cps=round(measured, 6),
                hard_cps_ceiling=hard_cps,
                reason="minimum_presentation_window_exceeds_available",
            ))
    if not feasible and not evidence:
        by_id = {word.word_id: word for word in words}
        for cue in cues:
            mapped = tuple(by_id[word.word_id] for word in cue.words if word.word_id in by_id)
            if not mapped or cue.timing_mode != "word":
                continue
            hard_cps = _maximum_caption_cps(mapped)
            characters = len("".join(cue.resolved_lines).replace(" ", ""))
            available = cue.output.end_frame - cue.output.start_frame
            required = _minimum_caption_frames(mapped, hard_cps)
            measured = characters / max(1 / 30, available / 30)
            if required <= available or measured <= hard_cps + 1e-9:
                continue
            source = SourceInterval(
                start_tick=min(word.source.start_tick for word in mapped),
                end_tick=max(word.source.end_tick for word in mapped),
            )
            map_ids = tuple(dict.fromkeys(
                map_id for word in mapped for map_id in word.map_ids
            ))
            evidence.append(CaptionFeasibilityEvidence(
                evidence_id=f"caption-cps-evidence-{len(evidence) + 1:03d}",
                word_ids=tuple(word.word_id for word in mapped),
                text=" ".join(word.text for word in mapped),
                source=source,
                immutable_word_output=_words_output(mapped),
                mapping_segment_ids=map_ids,
                character_count=characters,
                available_frames=available,
                required_frames=required,
                measured_cps=round(measured, 6),
                hard_cps_ceiling=hard_cps,
                reason="minimum_presentation_window_exceeds_available",
            ))
    # A failed search must have an auditable local witness. If the current
    # partition has no hard-CPS witness, it is not safe to claim physical
    # impossibility; retain the ordinary quality finding instead.
    infeasible = not feasible and bool(evidence)
    status = "INFEASIBLE" if infeasible else "FEASIBLE"
    reason = "CAPTION_CPS_INFEASIBLE" if infeasible else "CAPTION_TEMPORALLY_FEASIBLE"
    identity = canonical_hash({
        "version": CAPTION_FEASIBILITY_VERSION,
        "intent_id": intent.intent_id,
        "mapping": intent.source_output_mapping.fingerprint,
        "status": status,
        "word_ids": [word.word_id for word in words],
        "word_outputs": [word.output.model_dump(mode="json") for word in words],
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "evaluated_partition_count": evaluated,
    })
    return CaptionFeasibilityDecision(
        decision_id=f"caption-feasibility-{identity[:24]}",
        status=status,
        reason_code=reason,
        evaluated_word_count=len(words),
        evaluated_partition_count=evaluated,
        evidence=tuple(evidence) if infeasible else (),
    )


def _caption_run_is_feasible(
    words: tuple[_MappedWord, ...],
    intent: CreativeIntent,
    config: ProductionRenderConfig,
    measurer: _FontMeasurer,
    minimum_size: int,
    maximum_width: float,
    protected_phrases: tuple[tuple[str, ...], ...],
) -> tuple[bool, int, list[tuple[tuple[_MappedWord, ...], int, int]]]:
    chain_start, chain_end = _mapping_chain_bounds(intent, words[0].map_ids[0])
    maximum_frames = round(config.subtitle_max_duration * 30)
    # index -> earliest possible exclusive end frame. An earlier end dominates
    # every later state because it leaves at least as much time for the tail.
    best: dict[int, int] = {0: chain_start}
    evaluated = 0
    failed_windows: list[tuple[tuple[_MappedWord, ...], int, int]] = []
    for end in range(1, len(words) + 1):
        if end < len(words) and _splits_protected_phrase(words, end, protected_phrases):
            continue
        for start in range(max(0, end - config.subtitle_max_words_per_cue), end):
            previous_end = best.get(start)
            if previous_end is None:
                continue
            if end - start < config.subtitle_min_words_per_cue and end != len(words):
                continue
            selected = words[start:end]
            raw = _words_output(selected)
            if raw.end_frame - raw.start_frame > maximum_frames:
                continue
            if _best_lines(
                selected, measurer, minimum_size, maximum_width, protected_phrases,
            ) is None:
                continue
            evaluated += 1
            required = max(
                raw.end_frame - raw.start_frame,
                _minimum_caption_frames(selected, _maximum_caption_cps(selected)),
            )
            if required > maximum_frames:
                continue
            earliest_start = max(chain_start, previous_end - 2)
            if earliest_start > raw.start_frame:
                continue
            earliest_end = max(raw.end_frame, earliest_start + required)
            if earliest_end > chain_end:
                if end == len(words):
                    failed_windows.append((selected, chain_end - earliest_start, required))
                continue
            current = best.get(end)
            if current is None or earliest_end < current:
                best[end] = earliest_end
    return len(words) in best, evaluated, failed_windows


def _timing_mode(words: tuple[_MappedWord, ...]) -> tuple[Literal["word", "phrase", "static"], float]:
    confidence = min((word.confidence for word in words), default=0.0)
    sources = {word.timing_source for word in words}
    monotonic = all(
        right.output.start_frame >= left.output.start_frame
        and right.output.start_frame >= left.output.end_frame - 1
        for left, right in zip(words, words[1:])
    )
    plausible = all(1 <= word.output.end_frame - word.output.start_frame <= 45 for word in words)
    if sources <= {"verified", "aligned"} and confidence >= 0.70 and monotonic and plausible:
        return "word", round(confidence, 3)
    if "estimated" not in sources and confidence >= 0.45:
        return "phrase", round(confidence, 3)
    return "static", round(confidence, 3)


def _semantic_event(
    intent: CreativeIntent, words: tuple[_MappedWord, ...], output: OutputInterval, last_motion_frame: int,
) -> _SemanticEvent | None:
    candidates: list[_SemanticEvent] = []
    cue_text = [_normalise(word.text) for word in words]
    for emphasis in intent.semantic_emphasis:
        if not _overlaps(output, emphasis.output):
            continue
        phrase = [_normalise(value) for value in emphasis.text_span.split() if _normalise(value)]
        text_match = _find_phrase(cue_text, phrase) is not None
        word_overlap = any(_overlaps(word.output, emphasis.output) for word in words)
        if text_match or word_overlap:
            candidates.append(_SemanticEvent(
                "emphasis", emphasis.output, emphasis.confidence, emphasis.importance,
                emphasis.evidence_refs, emphasis=emphasis,
            ))
    for beat in intent.beats:
        if beat.role in {BeatRole.HOOK, BeatRole.PAYOFF} and _overlaps(output, beat.output):
            candidates.append(_SemanticEvent(
                "beat", beat.output, beat.confidence, beat.importance, beat.evidence_refs, beat=beat,
            ))
    for motion in intent.motion_events:
        if motion.domain.value == "caption" and _overlaps(output, motion.output):
            candidates.append(_SemanticEvent(
                "motion", motion.output, motion.confidence, 0.75, motion.evidence_refs, motion=motion,
            ))
    if not candidates:
        return None
    threshold = {Intensity.LOW: 0.78, Intensity.BALANCED: 0.58, Intensity.HIGH: 0.42}[intent.policy.intensity]
    candidates = [
        item for item in candidates
        if item.confidence >= SEMANTIC_PRESENTATION_MIN_CONFIDENCE
        and item.confidence * item.importance >= threshold
    ]
    if not candidates:
        return None
    chosen = max(
        candidates,
        key=lambda item: (
            3 if item.beat is not None and item.beat.role == BeatRole.PAYOFF else
            2 if item.kind == "emphasis" else 1,
            item.importance * item.confidence,
            -item.output.start_frame,
        ),
    )
    cooldown = {Intensity.LOW: 90, Intensity.BALANCED: 45, Intensity.HIGH: 30}[intent.policy.intensity]
    if output.start_frame - last_motion_frame < cooldown and not (
        chosen.beat is not None and chosen.beat.role == BeatRole.PAYOFF
    ):
        return None
    return chosen


def _semantic_emphasis_event(
    intent: CreativeIntent, words: tuple[_MappedWord, ...], output: OutputInterval,
) -> _SemanticEvent | None:
    """Resolve caption emphasis independently from beat/motion presentation.

    A payoff may own the cue's entrance primitive while an evidence-backed
    phrase still owns its color/karaoke treatment.  Selecting a single event
    for both concerns silently discarded the emphasis whenever both occupied
    the same cue.
    """

    cue_text = [_normalise(word.text) for word in words]
    threshold = {
        Intensity.LOW: 0.78,
        Intensity.BALANCED: 0.58,
        Intensity.HIGH: 0.42,
    }[intent.policy.intensity]
    candidates: list[_SemanticEvent] = []
    for emphasis in intent.semantic_emphasis:
        if not _overlaps(output, emphasis.output):
            continue
        phrase = [_normalise(value) for value in emphasis.text_span.split() if _normalise(value)]
        if _find_phrase(cue_text, phrase) is None and not any(
            _overlaps(word.output, emphasis.output) for word in words
        ):
            continue
        if emphasis.confidence < SEMANTIC_PRESENTATION_MIN_CONFIDENCE:
            continue
        if emphasis.confidence * emphasis.importance < threshold:
            continue
        candidates.append(_SemanticEvent(
            "emphasis",
            emphasis.output,
            emphasis.confidence,
            emphasis.importance,
            emphasis.evidence_refs,
            emphasis=emphasis,
        ))
    return max(
        candidates,
        key=lambda item: (item.importance * item.confidence, -item.output.start_frame),
        default=None,
    )


def _primitive(
    intent: CreativeIntent, timing_mode: str, event: _SemanticEvent | None,
) -> Literal["static", "fade", "scale", "slide", "karaoke"]:
    if event is None:
        return "static"
    if timing_mode != "word":
        return "static"
    if intent.policy.reduced_motion:
        return "fade"
    if intent.policy.intensity == Intensity.LOW:
        return "fade" if event.beat is not None else "static"
    if event.emphasis is not None and timing_mode == "word":
        return "karaoke"
    if event.beat is not None and event.beat.role == BeatRole.PAYOFF:
        return "scale"
    if event.beat is not None and event.beat.role == BeatRole.HOOK:
        return "slide" if intent.policy.intensity == Intensity.HIGH else "fade"
    if event.motion is not None:
        return "scale" if intent.policy.intensity == Intensity.HIGH else "fade"
    return "fade"


def _emphasis_plan(
    event: _SemanticEvent | None, words: tuple[_MappedWord, ...], cue: OutputInterval,
    primitive: str, timing_mode: str,
) -> CaptionEmphasisPlan | None:
    if (
        event is None
        or event.emphasis is None
        or timing_mode != "word"
        or event.confidence < SEMANTIC_PRESENTATION_MIN_CONFIDENCE
    ):
        return None
    normalised = [_normalise(word.text) for word in words]
    phrase = [_normalise(value) for value in event.emphasis.text_span.split() if _normalise(value)]
    found = _find_phrase(normalised, phrase)
    if found is not None:
        indexes = tuple(range(found, found + len(phrase)))
    else:
        indexes = tuple(index for index, word in enumerate(words) if _overlaps(word.output, event.output))
    indexes = indexes[:MAX_SEMANTIC_EMPHASIS_WORDS]
    if not indexes:
        return None
    start = max(cue.start_frame, min(words[index].output.start_frame for index in indexes))
    end = min(cue.end_frame, max(words[index].output.end_frame for index in indexes))
    treatment: Literal["color", "phrase_color", "karaoke", "bounded_scale"]
    if primitive == "karaoke" and timing_mode == "word":
        treatment = "karaoke"
    elif primitive == "scale":
        treatment = "bounded_scale"
    elif timing_mode == "word":
        treatment = "color"
    else:
        treatment = "phrase_color"
    return CaptionEmphasisPlan(
        emphasis_id=event.emphasis.decision_id,
        output=OutputInterval(start_frame=start, end_frame=max(start + 1, end)),
        word_indexes=indexes,
        semantic_class=event.emphasis.semantic_class,
        importance=event.emphasis.importance,
        confidence=event.emphasis.confidence,
        evidence_refs=event.emphasis.evidence_refs,
        treatment=treatment,
    )


def _find_phrase(words: list[str], phrase: list[str]) -> int | None:
    if not phrase:
        return None
    for index in range(len(words) - len(phrase) + 1):
        if words[index:index + len(phrase)] == phrase:
            return index
    return None


def _caption_bounds(
    layout: _Layout, lane: str, measurer: _FontMeasurer, config: ProductionRenderConfig,
    typography: CaptionTypographyToken, platform: str,
) -> NormalizedRect:
    outline = typography.outline_width_ratio * config.output_height
    shadow = typography.shadow_ratio * config.output_height
    width = max(measurer.width(line, layout.font_size) for line in layout.lines) + 2 * (outline + shadow)
    height = layout.font_size * typography.line_height * len(layout.lines) + 2 * (outline + shadow)
    left, top, right, bottom = _SAFE_INSETS.get(platform, _SAFE_INSETS["universal"])
    width_ratio = min(width / config.output_width, 1 - left - right)
    height_ratio = min(height / config.output_height, 1 - top - bottom)
    x = max(left, min(1 - right - width_ratio, 0.5 - width_ratio / 2))
    y = max(top, min(1 - bottom - height_ratio, _LANE_CENTERS[lane] - height_ratio / 2))
    return NormalizedRect(
        x=round(x, 7), y=round(y, 7), width=round(width_ratio, 7), height=round(height_ratio, 7),
    )


def _composition_regions(composition: CompositionPlan | None) -> tuple[CaptionProtectedRegion, ...]:
    if composition is None:
        return ()
    result: list[CaptionProtectedRegion] = []
    for segment in composition.segments:
        for index, bounds in enumerate(segment.protected_regions, start=1):
            kind: Literal["face", "object", "screen", "text", "overlay"] = (
                "screen" if segment.target.value == "screen" else
                "face" if segment.target.value in {"speaker", "subject", "reaction"} else "object"
            )
            result.append(CaptionProtectedRegion(
                region_id=f"{segment.segment_id}-protected-{index}", output=segment.output,
                bounds=bounds, kind=kind, importance=1.0, confidence=1.0,
            ))
    return tuple(result)


def _resolve_lane(
    output: OutputInterval,
    bounds_by_lane: dict[CaptionLane, NormalizedRect],
    regions: Sequence[CaptionProtectedRegion],
    previous_lane: CaptionLane | None,
    platform: str,
) -> tuple[CaptionLane, NormalizedRect, CaptionCollisionDecision]:
    relevant = [region for region in regions if _overlaps(output, region.output)]
    costs: list[tuple[CaptionLane, float, float, tuple[str, ...], bool]] = []
    for index, lane in enumerate(_LANES):
        bounds = bounds_by_lane[lane]
        safe = _inside_safe_zone(bounds, platform)
        overlaps = [(_rect_overlap_ratio(bounds, region.bounds), region) for region in relevant]
        ratio = max((value for value, _region in overlaps), default=0.0)
        ids = tuple(region.region_id for value, region in overlaps if value > 0)
        overlap_cost = sum(value * region.importance * region.confidence * 10_000 for value, region in overlaps)
        stability = 0 if previous_lane is None or lane == previous_lane else 18
        preference = index * 0.8
        cost = overlap_cost + stability + preference + (10_000 if not safe else 0)
        costs.append((lane, round(cost, 6), ratio, ids, safe))
    lane, _cost, ratio, ids, safe = min(costs, key=lambda item: (item[1], _LANES.index(item[0])))
    switched = previous_lane is not None and lane != previous_lane
    reason: CollisionReason
    if not safe:
        reason = "platform_safe_zone"
    elif ratio > 0:
        reason = "least_overlap_fallback"
    elif switched:
        reason = "protected_region_avoidance"
    elif previous_lane == lane:
        reason = "stable_lane"
    else:
        reason = "preferred_lane"
    decision = CaptionCollisionDecision(
        lane=lane, candidate_costs=tuple((item[0], item[1]) for item in costs),
        overlap_ratio=round(ratio, 6), protected_region_ids=ids,
        safe_zone_valid=safe, switched_lane=switched, reason=reason,
    )
    return lane, bounds_by_lane[lane], decision


def _inside_safe_zone(bounds: NormalizedRect, platform: str) -> bool:
    left, top, right, bottom = _SAFE_INSETS.get(platform, _SAFE_INSETS["universal"])
    return (
        bounds.x >= left - 1e-7 and bounds.y >= top - 1e-7
        and bounds.x + bounds.width <= 1 - right + 1e-7
        and bounds.y + bounds.height <= 1 - bottom + 1e-7
    )


def _rect_overlap_ratio(left: NormalizedRect, right: NormalizedRect) -> float:
    width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    return (width * height) / max(1e-9, left.width * left.height)


def _assess_plan(
    cues: list[CaptionCuePlan], manifest: CaptionFontManifest, measurer: _FontMeasurer,
    intent: CreativeIntent, config: ProductionRenderConfig, intensity_degraded: set[str],
    feasibility: CaptionFeasibilityDecision,
) -> tuple[CaptionQualityFinding, ...]:
    findings: list[CaptionQualityFinding] = []
    maximum_cps = CAPTION_HARD_CPS_CEILING
    left, _top, right, _bottom = _SAFE_INSETS.get(intent.policy.platform, _SAFE_INSETS["universal"])
    maximum_width = config.output_width * min(config.subtitle_max_rendered_width_ratio, 1 - left - right)
    for cue in cues:
        duration = (cue.output.end_frame - cue.output.start_frame) / 30
        cps = len("".join(cue.resolved_lines).replace(" ", "")) / max(1 / 30, duration)
        if cps > maximum_cps:
            severity: Literal["warning", "blocker"] = "blocker" if cue.timing_mode == "word" else "warning"
            code: Literal["CAPTION_CPS_HIGH", "CAPTION_CPS_INFEASIBLE"] = (
                "CAPTION_CPS_INFEASIBLE"
                if severity == "blocker" and feasibility.status == "INFEASIBLE"
                else "CAPTION_CPS_HIGH"
            )
            findings.append(CaptionQualityFinding(
                code=code, severity=severity, cue_id=cue.cue_id,
                measured_value=round(cps, 3), threshold=maximum_cps,
                message=(
                    "Frozen word timing and semantic constraints make the hard CPS ceiling physically infeasible."
                    if code == "CAPTION_CPS_INFEASIBLE"
                    else "Caption reading speed exceeds the language-profile ceiling."
                ),
            ))
        if cue.timing_mode != "word":
            findings.append(CaptionQualityFinding(
                code="CAPTION_TIMING_WEAK", severity="warning", cue_id=cue.cue_id,
                measured_value=cue.timing_mode, threshold="trusted word timing",
                message="Weak word timing was safely degraded to phrase/static caption timing.",
            ))
        if cue.cue_id in intensity_degraded:
            findings.append(CaptionQualityFinding(
                code="CAPTION_INTENSITY_DEGRADED", severity="warning", cue_id=cue.cue_id,
                measured_value=intent.policy.intensity.value, threshold="timing-safe primitive",
                message="Requested intensity was reduced because the semantic event has weak timing.",
            ))
        resolved_size = max(8, round((cue.resolved_font_size_ratio or 0.03) * config.output_height))
        measured_width = max((measurer.width(line, resolved_size) for line in cue.resolved_lines), default=0.0)
        if measured_width > maximum_width + 0.5:
            findings.append(CaptionQualityFinding(
                code="CAPTION_LINE_OVERFLOW", severity="blocker", cue_id=cue.cue_id,
                measured_value=round(measured_width, 3), threshold=round(maximum_width, 3),
                message="Resolved font-metric line width exceeds the approved caption width.",
            ))
        elif cue.fallback_reason == "readability":
            findings.append(CaptionQualityFinding(
                code="CAPTION_READABILITY_FALLBACK", severity="warning", cue_id=cue.cue_id,
                measured_value=resolved_size, threshold="base font size and semantic fit",
                message="Caption fitting used bounded font reduction or a safe cue split.",
            ))
        if cue.collision is not None and not cue.collision.safe_zone_valid:
            findings.append(CaptionQualityFinding(
                code="CAPTION_SAFE_ZONE_VIOLATION", severity="blocker", cue_id=cue.cue_id,
                measured_value=False, threshold=True,
                message="No caption lane keeps the resolved block inside the platform safe zone.",
            ))
        if cue.collision is not None and cue.collision.overlap_ratio > 0.01:
            findings.append(CaptionQualityFinding(
                code="CAPTION_PROTECTED_REGION_OVERLAP", severity="blocker", cue_id=cue.cue_id,
                measured_value=cue.collision.overlap_ratio, threshold=0.01,
                message="No caption lane avoids an important face/object/screen region.",
            ))
    if manifest.fallback_used:
        findings.append(CaptionQualityFinding(
            code="CAPTION_FONT_FALLBACK", severity="warning", measured_value=manifest.resolved_family,
            threshold=manifest.requested_family,
            message="The requested font was unavailable; an approved deterministic fallback was selected.",
        ))
    if measurer.used_heuristic or manifest.metrics_backend == "heuristic":
        findings.append(CaptionQualityFinding(
            code="CAPTION_METRICS_FALLBACK", severity="warning", measured_value=manifest.metrics_backend,
            threshold="qt_file_metrics", message="Exact font metrics were unavailable; conservative metrics were used.",
        ))
    duration_minutes = max((cue.output.end_frame for cue in cues), default=0) / 30 / 60
    switches = sum(
        left.lane != right.lane for left, right in zip(cues, cues[1:])
    )
    rate = switches / duration_minutes if duration_minutes > 0 else 0
    if rate > 8:
        findings.append(CaptionQualityFinding(
            code="CAPTION_LANE_SWITCH_RATE_HIGH", severity="warning",
            measured_value=round(rate, 3), threshold=8,
            message="Caption lane switches exceed the stable-lane readability budget.",
        ))
    return tuple(findings)


def _quality_report(
    cues: Sequence[CaptionCuePlan], findings: tuple[CaptionQualityFinding, ...],
    manifest: CaptionFontManifest, measurer: _FontMeasurer, intent: CreativeIntent,
) -> CaptionQualityReport:
    status: Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED"] = (
        "BLOCKED" if any(item.severity == "blocker" for item in findings)
        else "PASS_WITH_WARNINGS" if findings else "PASS"
    )
    durations = [(cue.output.end_frame - cue.output.start_frame) / 30 for cue in cues]
    cps = [len("".join(cue.resolved_lines).replace(" ", "")) / max(1 / 30, duration) for cue, duration in zip(cues, durations)]
    switches = sum(left.lane != right.lane for left, right in zip(cues, cues[1:]))
    return CaptionQualityReport(
        status=status,
        findings=findings,
        metrics=CaptionQualityMetrics(
            cue_count=len(cues),
            word_timed_cue_count=sum(cue.timing_mode == "word" for cue in cues),
            weak_timing_cue_count=sum(cue.timing_mode != "word" for cue in cues),
            semantic_emphasis_count=sum(cue.emphasis is not None for cue in cues),
            motion_cue_count=sum(cue.primitive_id != "static" for cue in cues),
            lane_switch_count=switches,
            protected_overlap_count=sum(bool(cue.collision and cue.collision.overlap_ratio > 0.01) for cue in cues),
            safe_zone_violation_count=sum(bool(cue.collision and not cue.collision.safe_zone_valid) for cue in cues),
            max_cps=round(max(cps, default=0.0), 3),
            font_exact=manifest.file_sha256 is not None,
            metrics_exact=not measurer.used_heuristic and manifest.metrics_backend in {"gdi_file_metrics", "qt_file_metrics"},
        ),
        provenance=CaptionQualityProvenance(
            producer="app.caption_planning.CaptionPlanner",
            planner_version=CAPTION_PLANNER_VERSION,
            backend=LIBASS_BACKEND_VERSION,
            intent_id=intent.intent_id,
        ),
    )


def _diagnostics(
    cues: Sequence[CaptionCuePlan], manifest: CaptionFontManifest, measurer: _FontMeasurer,
) -> list[str]:
    result: list[str] = []
    if any(cue.timing_mode != "word" for cue in cues):
        result.append("WEAK_TIMING_DEGRADED_TO_PHRASE_STATIC")
    if manifest.fallback_used:
        result.append("APPROVED_FONT_FALLBACK_USED")
    if measurer.used_heuristic:
        result.append("HEURISTIC_FONT_METRICS_USED")
    if any(cue.collision and cue.collision.switched_lane for cue in cues):
        result.append("CAPTION_LANE_SWITCH_FOR_PROTECTED_REGION")
    return result


def write_caption_plan_ass(
    plan: CaptionPlan, path: Path, width: int, height: int, *, verify_font: bool = True,
) -> Path:
    """Render the exact plan lines/timing/geometry as safe Tier 1 ASS.

    Preview and Final may pass different resolutions; all semantic decisions
    stay frozen and only normalized geometry/typography is scaled.
    """

    if plan.backend_id != "libass" or plan.typography is None or plan.font_manifest is None:
        raise ValueError("CAPTION_PLAN_NOT_RENDERABLE_BY_LIBASS")
    if width <= 0 or height <= 0:
        raise ValueError("caption canvas dimensions must be positive")
    if verify_font and plan.font_manifest.file_sha256 is not None and _find_manifest_file(plan.font_manifest) is None:
        raise ValueError("CAPTION_FONT_CHECKSUM_MISMATCH")
    typography = plan.typography
    manifest = plan.font_manifest
    font_size = max(8, round(typography.font_size_ratio * height))
    outline = max(0, round(typography.outline_width_ratio * height))
    shadow = max(0, round(typography.shadow_ratio * height))
    header = f"""[Script Info]
; CaptionPlan: {plan.schema_version}
; FontSHA256: {manifest.file_sha256 or 'unverified'}
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: CaptionPlan,{_escape_style(manifest.resolved_family)},{font_size},{_ass_color(typography.text_color)},{_ass_color(typography.highlight_color)},{_ass_color(typography.outline_color)},&H00000000,{-1 if typography.font_weight == 'bold' else 0},0,0,0,100,100,0,0,1,{outline},{shadow},8,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events: list[str] = []
    for cue in plan.cues:
        if cue.normalized_bounds is None:
            raise ValueError(f"CAPTION_GEOMETRY_MISSING: {cue.cue_id}")
        payload = _format_plan_cue(cue, typography, width, height)
        events.append(
            f"Dialogue: 0,{_ass_time(cue.output.start_frame / 30)},{_ass_time(cue.output.end_frame / 30)},"
            f"CaptionPlan,{_escape_style(cue.cue_id)},0,0,0,,{payload}"
        )
    write_bytes_atomic(path, (header + "\n".join(events) + "\n").encode("utf-8-sig"))
    return path


def _format_plan_cue(
    cue: CaptionCuePlan, typography: CaptionTypographyToken, width: int, height: int,
) -> str:
    assert cue.normalized_bounds is not None
    x = round((cue.normalized_bounds.x + cue.normalized_bounds.width / 2) * width)
    y = round(cue.normalized_bounds.y * height)
    size = max(8, round((cue.resolved_font_size_ratio or typography.font_size_ratio) * height))
    duration_ms = round(cue.motion_duration_frames / 30 * 1000)
    tags = [r"\an8", fr"\fs{size}"]
    if cue.primitive_id == "slide":
        start_y = y + round(cue.slide_distance_ratio * height)
        tags.append(fr"\move({x},{start_y},{x},{y},0,{duration_ms})")
    else:
        tags.append(fr"\pos({x},{y})")
    if cue.primitive_id == "fade":
        tags.append(fr"\fad({duration_ms},100)")
    elif cue.primitive_id == "scale":
        tags.extend((fr"\fscx{cue.scale_percent}\fscy{cue.scale_percent}", fr"\t(0,{duration_ms},\fscx100\fscy100)"))
    text = _format_plan_text(cue, typography)
    return "{" + "".join(tags) + "}" + text


def _format_plan_text(cue: CaptionCuePlan, typography: CaptionTypographyToken) -> str:
    if not cue.words:
        return r"\N".join(_escape_text(line) for line in cue.resolved_lines)
    line_counts = [len(line.split()) for line in cue.resolved_lines]
    boundaries: set[int] = set()
    cursor = 0
    for count in line_counts[:-1]:
        cursor += count
        boundaries.add(cursor)
    emphasis = set(cue.emphasis.word_indexes if cue.emphasis is not None else ())
    primary = _ass_color(typography.text_color)
    highlight = _ass_color(typography.highlight_color)
    pieces: list[str] = []
    for index, word in enumerate(cue.words):
        rendered = word.text.upper() if typography.uppercase_emphasis and index in emphasis else word.text
        escaped = _escape_text(rendered)
        if cue.primitive_id == "karaoke" and cue.timing_mode == "word":
            next_start = cue.words[index + 1].output.start_frame if index + 1 < len(cue.words) else cue.output.end_frame
            # ASS karaoke advances cumulatively.  Using activation-start deltas
            # preserves the frozen word activation frames, including pauses.
            duration = max(1, round((next_start - word.output.start_frame) / 30 * 100))
            if index in emphasis:
                piece = f"{{\\1c{highlight}\\2c{primary}\\kf{duration}}}{escaped}{{\\1c{primary}}}"
            else:
                piece = f"{{\\1c{primary}\\2c{primary}\\k{duration}}}{escaped}"
        elif index in emphasis:
            piece = f"{{\\1c{highlight}}}{escaped}{{\\1c{primary}}}"
        else:
            piece = escaped
        pieces.append(piece)
        if index + 1 in boundaries:
            pieces.append(r"\N")
        elif index + 1 < len(cue.words):
            pieces.append(" ")
    return "".join(pieces)


def _overlaps(left: OutputInterval, right: OutputInterval) -> bool:
    return left.start_frame < right.end_frame and right.start_frame < left.end_frame


def _normalise(value: str) -> str:
    return _NORMALISE_RE.sub("", value.casefold())


def _contains_cyrillic(value: str) -> bool:
    return any("а" <= character.casefold() <= "я" or character.casefold() == "ё" for character in value)


def _escape_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _escape_style(value: str) -> str:
    return value.replace(",", " ").replace("\n", " ").replace("\r", " ")


def _ass_color(value: str) -> str:
    red, green, blue = value[1:3], value[3:5], value[5:7]
    return f"&H00{blue}{green}{red}&"


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"
