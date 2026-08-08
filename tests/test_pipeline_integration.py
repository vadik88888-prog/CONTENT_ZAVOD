from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.pipeline import Pipeline
from app.sources import Source
from app.utils import read_json, write_json


def test_pipeline_creates_artifacts_and_reuses_completed_stages(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = {"media": 0, "transcription": 0, "render": 0}

    def fake_prepare_media(source_path: Path, work_directory: Path) -> dict:
        calls["media"] += 1
        audio = work_directory / "audio.wav"
        audio.write_bytes(b"wav")
        metadata = {
            "duration": 45,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "audio_streams": 1,
            "audio_path": str(audio),
        }
        write_json(work_directory / "metadata.json", metadata)
        return metadata

    def fake_transcribe(audio_path, source_id, source_duration, config, destination):
        calls["transcription"] += 1
        words = [
            {"start": 1 + index, "end": 1.7 + index, "text": f"слово{index}"}
            for index in range(36)
        ]
        transcript = {
            "source_id": source_id,
            "language": "ru",
            "duration": source_duration,
            "segments": [{"start": 1, "end": 37, "text": " ".join(word["text"] for word in words)}],
            "words": words,
            "model": "fake",
            "runtime": {"device": "cpu"},
            "processing_duration_seconds": 0.01,
        }
        write_json(destination, transcript)
        destination.with_suffix(".txt").write_text(transcript["segments"][0]["text"], encoding="utf-8")
        return transcript

    def fake_render(source_path, item, ass_path, destination, config):
        calls["render"] += 1
        assert ass_path is not None and ass_path.is_file()
        destination.write_bytes(b"mp4")
        return destination, False, None

    monkeypatch.setattr("app.pipeline.prepare_media", fake_prepare_media)
    monkeypatch.setattr("app.pipeline.transcribe", fake_transcribe)
    monkeypatch.setattr("app.pipeline.render_clip", fake_render)

    config = AppConfig(score_threshold=0)
    first = Pipeline(tmp_path, config, mock_ai=True).run(input_path=str(source))

    assert len(first.output_files) == 1
    assert first.report_path.is_file()
    assert (first.work_directory / "transcript.json").is_file()
    assert (first.work_directory / "multimodal_timeline.json").is_file()
    assert (first.work_directory / "candidates.raw.json").is_file()
    assert (first.work_directory / "candidates.scored.json").is_file()
    assert calls == {"media": 1, "transcription": 1, "render": 1}

    second = Pipeline(tmp_path, config, mock_ai=True).run(input_path=str(source))

    # Analysis artifacts are source-cached, while every run gets its own
    # canonical output directory and therefore performs its own final render.
    assert second.output_files != first.output_files
    assert second.output_files[0].is_relative_to(second.output_directory)
    assert first.output_files[0].is_relative_to(first.output_directory)
    assert calls == {"media": 1, "transcription": 1, "render": 2}
    timeline = read_json(second.work_directory / "multimodal_timeline.json", {})
    story_units = read_json(second.work_directory / "story_units.json", {})["story_units"]
    second_report = read_json(second.report_path, {})
    assert timeline["source_id"]
    assert timeline["diagnostics"]["external_vision_api_calls"] == 0
    assert story_units and story_units[0]["multimodal_evidence"]["analysis_run_id"] == timeline["analysis_run_id"]
    assert second_report["stages"]["multimodal_timeline"]["cache_hit"] is True
    assert second_report["content_understanding"]["multimodal_timeline_ref"].endswith("multimodal_timeline.json")


def test_pipeline_reports_video_without_audio_without_crashing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"silent")

    def fake_prepare_media(source_path: Path, work_directory: Path) -> dict:
        metadata = {
            "duration": 12,
            "width": 1080,
            "height": 1920,
            "audio_streams": 0,
            "audio_path": None,
            "warning": "В видео нет аудиодорожки.",
        }
        write_json(work_directory / "metadata.json", metadata)
        return metadata

    monkeypatch.setattr("app.pipeline.prepare_media", fake_prepare_media)

    result = Pipeline(tmp_path, AppConfig(), mock_ai=True).run(input_path=str(source))

    report = result.report_path.read_text(encoding="utf-8")
    assert result.output_files == []
    assert "В видео нет аудиодорожки." in report
    timeline = read_json(result.work_directory / "multimodal_timeline.json", {})
    assert timeline["diagnostics"]["evidence"]["audio"]["status"] == "missing"
    assert timeline["diagnostics"]["evidence"]["visual"]["status"] == "missing"


def test_changed_config_invalidates_transcript_but_keeps_metadata(tmp_path, monkeypatch) -> None:
    source = tmp_path / "config-change.mp4"
    source.write_bytes(b"source")
    calls = {"media": 0, "transcription": 0}

    def fake_prepare_media(source_path: Path, work_directory: Path) -> dict:
        calls["media"] += 1
        audio = work_directory / "audio.wav"
        audio.write_bytes(b"wav")
        metadata = {"duration": 20, "width": 1080, "height": 1920, "audio_streams": 1, "audio_path": str(audio)}
        write_json(work_directory / "metadata.json", metadata)
        return metadata

    def fake_transcribe(audio_path, source_id, source_duration, config, destination):
        calls["transcription"] += 1
        words = [{"start": number, "end": number + 0.5, "text": f"word{number}"} for number in range(1, 19)]
        transcript = {
            "source_id": source_id, "language": "en", "duration": source_duration,
            "segments": [{"start": 1, "end": 19, "text": " ".join(word["text"] for word in words)}],
            "words": words, "model": config.whisper_model, "runtime": {}, "processing_duration_seconds": 0,
        }
        write_json(destination, transcript)
        destination.with_suffix(".txt").write_text("text", encoding="utf-8")
        return transcript

    def fake_render(source_path, item, ass_path, destination, config):
        destination.write_bytes(b"mp4")
        return destination, False, None

    monkeypatch.setattr("app.pipeline.prepare_media", fake_prepare_media)
    monkeypatch.setattr("app.pipeline.transcribe", fake_transcribe)
    monkeypatch.setattr("app.pipeline.render_clip", fake_render)

    Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True).run(input_path=str(source))
    Pipeline(tmp_path, AppConfig(score_threshold=0, whisper_model="base"), mock_ai=True).run(input_path=str(source))

    assert calls == {"media": 1, "transcription": 2}


def test_downloaded_source_is_deleted_only_after_a_successful_render(tmp_path, monkeypatch) -> None:
    downloaded = tmp_path / "downloaded.mp4"
    downloaded.write_bytes(b"source")

    def fake_url_source(url, destination):
        return Source("remote", downloaded, "downloaded", url, downloaded=True)

    def fake_prepare_media(source_path: Path, work_directory: Path) -> dict:
        audio = work_directory / "audio.wav"
        audio.write_bytes(b"wav")
        metadata = {"duration": 20, "width": 1080, "height": 1920, "audio_streams": 1, "audio_path": str(audio)}
        write_json(work_directory / "metadata.json", metadata)
        return metadata

    def fake_transcribe(audio_path, source_id, source_duration, config, destination):
        words = [{"start": number, "end": number + 0.5, "text": f"word{number}"} for number in range(1, 19)]
        transcript = {
            "source_id": source_id, "language": "en", "duration": source_duration,
            "segments": [{"start": 1, "end": 19, "text": " ".join(word["text"] for word in words)}],
            "words": words, "model": "fake", "runtime": {"device": "cpu"}, "processing_duration_seconds": 0,
        }
        write_json(destination, transcript)
        destination.with_suffix(".txt").write_text("text", encoding="utf-8")
        return transcript

    def fake_render(source_path, item, ass_path, destination, config):
        destination.write_bytes(b"mp4")
        return destination, False, None

    monkeypatch.setattr("app.pipeline.url_source", fake_url_source)
    monkeypatch.setattr("app.pipeline.prepare_media", fake_prepare_media)
    monkeypatch.setattr("app.pipeline.transcribe", fake_transcribe)
    monkeypatch.setattr("app.pipeline.render_clip", fake_render)

    result = Pipeline(
        tmp_path, AppConfig(score_threshold=0, delete_downloaded_source=True), mock_ai=True
    ).run(url="https://example.test/video")

    assert result.output_files
    assert not downloaded.exists()
    assert "Загруженный исходник удалён" in result.report_path.read_text(encoding="utf-8")
