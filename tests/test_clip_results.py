from app.clip_results import ClipResult, primary_clip_results, unique_primary_results


def test_registry_rejects_duplicate_candidate_plan_range_and_content() -> None:
    results = primary_clip_results({"items": [
        {
            "clip_result_id": "one", "candidate_id": "candidate-one", "production_plan_id": "plan-one",
            "output_file": "one.mp4", "source_start_seconds": 10, "source_end_seconds": 30,
            "content_fingerprint": "media-one", "status": "completed",
        },
        {
            "clip_result_id": "two", "candidate_id": "candidate-two", "production_plan_id": "plan-two",
            "output_file": "two.mp4", "source_start_seconds": 10.1, "source_end_seconds": 30.1,
            "content_fingerprint": "media-two", "status": "completed",
        },
        {
            "clip_result_id": "three", "candidate_id": "candidate-three", "production_plan_id": "plan-three",
            "output_file": "three.mp4", "source_start_seconds": 50, "source_end_seconds": 70,
            "content_fingerprint": "media-one", "status": "completed",
        },
    ]})

    assert [item.candidate_id for item in results] == ["candidate-one"]


def test_legacy_registry_keeps_distinct_paths_and_drops_same_candidate() -> None:
    results = unique_primary_results([
        ClipResult("one", "one.mp4"),
        ClipResult("one", "two.mp4"),
        ClipResult("two", "two.mp4"),
    ])

    assert [item.output_file for item in results] == ["one.mp4", "two.mp4"]
