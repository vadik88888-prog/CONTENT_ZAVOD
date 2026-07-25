from app.models import Candidate
from app.subtitles import create_ass


def test_ass_uses_relative_clip_timestamps_and_two_lines(tmp_path) -> None:
    transcript = {
        "words": [
            {"start": 10.0, "end": 10.3, "text": "Привет"},
            {"start": 10.3, "end": 10.6, "text": "это"},
            {"start": 10.6, "end": 11.0, "text": "тест"},
            {"start": 11.0, "end": 11.3, "text": "субтитров"},
        ]
    }
    path = create_ass(transcript, Candidate("one", 10, 15, "текст"), tmp_path / "clip.ass")
    content = path.read_text(encoding="utf-8-sig")

    assert "PlayResX: 1080" in content
    assert "Dialogue: 0,0:00:00.00" in content
    assert r"\N" in content
