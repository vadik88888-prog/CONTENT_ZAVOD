from app.ai import build_gemini_payload
from app.models import Candidate


def test_gemini_payload_contains_timestamped_transcript_not_video_path() -> None:
    payload = build_gemini_payload(
        [Candidate("one", 1, 16, "Самостоятельный фрагмент.")],
        {
            "language": "ru",
            "source_path": r"C:\private\source.mp4",
            "segments": [{"start": 1, "end": 16, "text": "Самостоятельный фрагмент."}],
        },
    )

    assert payload["transcript"] == [
        {"start": 1.0, "end": 16.0, "text": "Самостоятельный фрагмент."}
    ]
    assert "source_path" not in payload
    assert payload["candidates"][0]["id"] == "one"
