from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from system1.scenes.gemini_judge import StructuredSceneBoundaryJudge
from system1.scenes.grouping import group_scenes, plan_focus_windows, vote_weight


def shots(count: int) -> list[dict]:
    return [
        {
            "shot_id": f"v_SH{index:05d}",
            "video_id": "v",
            "shot_index": index,
            "start_frame": index * 10,
            "end_frame": (index + 1) * 10,
            "start_sec": index * 0.4,
            "end_sec": (index + 1) * 0.4,
        }
        for index in range(count)
    ]


def config() -> dict:
    return {
        "focus_gap_count": 8,
        "context_shots_each_side": 4,
        "stride": 6,
        "boundary_threshold": 0.67,
        "non_boundary_threshold": 0.33,
        "dense_boundary_count": 2,
        "dense_boundary_gap_window": 3,
        "strong_disagreement_min_votes": 2,
        "strong_disagreement_requires_both_labels": True,
        "max_consistency_review_rounds": 1,
        "consistency_review_max_gaps": 2,
        "min_failed_gaps_before_abort": 3,
        "max_failed_gap_ratio": 0.25,
        "scene_confidence_aggregation": "null_v1",
    }


class Judge:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def judge(self, *, request_kind, focus_gap_ids, context):
        self.calls.append((request_kind, focus_gap_ids, tuple(item["shot_id"] for item in context)))
        return self.handler(request_kind, focus_gap_ids)


def test_window_planning_covers_every_gap_and_overlaps() -> None:
    windows = plan_focus_windows(20, focus_gap_count=8, context_shots_each_side=4, stride=6)
    covered = {gap for window in windows for gap in window.focus_gap_indices}
    assert covered == set(range(19))
    assert set(windows[0].focus_gap_indices) & set(windows[1].focus_gap_indices) == {6, 7}
    assert vote_weight(0, 8) == 1.0
    assert vote_weight(3, 8) == 2.0


def test_all_false_result_is_valid_one_scene() -> None:
    judge = Judge(lambda _kind, ids: {gap_id: False for gap_id in ids})
    rows, decisions = group_scenes(
        video_id="v", shots=shots(5), evidence=shots(5), judge=judge, config=config()
    )
    assert len(rows) == 1
    assert rows[0]["confidence"] is None
    assert all(not decision.is_boundary for decision in decisions)


def test_boundary_partition_has_deterministic_ids_and_ranges() -> None:
    judge = Judge(lambda _kind, ids: {gap_id: gap_id == "v_SH00001" for gap_id in ids})
    rows, _ = group_scenes(
        video_id="v", shots=shots(4), evidence=shots(4), judge=judge, config=config()
    )
    assert [row["scene_id"] for row in rows] == ["v_SC00000", "v_SC00001"]
    assert [(row["start_frame"], row["end_frame"]) for row in rows] == [(0, 20), (20, 40)]


def test_ambiguous_overlap_triggers_focused_review() -> None:
    def handler(kind, ids):
        if kind == "primary":
            # Gap 6 occurs in both windows and receives conflicting votes.
            return {gap_id: (gap_id == "v_SH00006" and len(ids) == 8 and ids[0] == "v_SH00000") for gap_id in ids}
        return {gap_id: True for gap_id in ids}

    judge = Judge(handler)
    _rows, decisions = group_scenes(
        video_id="v", shots=shots(10), evidence=shots(10), judge=judge, config=config()
    )
    decision = next(item for item in decisions if item.after_shot_id == "v_SH00006")
    assert decision.review_route in {"focused_review", "consistency_review"}
    assert any(call[0] == "focused_review" for call in judge.calls)


def test_dense_boundaries_trigger_bounded_consistency_regions() -> None:
    def handler(kind, ids):
        if kind == "primary":
            return {gap_id: gap_id in {"v_SH00001", "v_SH00002"} for gap_id in ids}
        return {gap_id: False for gap_id in ids}

    judge = Judge(handler)
    _rows, decisions = group_scenes(
        video_id="v", shots=shots(7), evidence=shots(7), judge=judge, config=config()
    )
    consistency_calls = [call for call in judge.calls if call[0] == "consistency_review"]
    assert consistency_calls
    cap = config()["consistency_review_max_gaps"]
    assert all(len(call[1]) <= cap for call in consistency_calls)
    # Both halves of the contract: no request larger than the cap, and no more
    # requests than the cap requires. Dropping the merge step would keep every
    # request small while doubling how many of them get sent.
    reviewed = {gap_id for call in consistency_calls for gap_id in call[1]}
    assert len(consistency_calls) == -(-len(reviewed) // cap)
    assert any(decision.consistency_review_triggered for decision in decisions)


def test_judge_must_return_exact_boolean_gap_set() -> None:
    judge = Judge(lambda _kind, _ids: {"unknown": True})
    with pytest.raises(ValueError, match="gap set mismatch"):
        group_scenes(video_id="v", shots=shots(3), evidence=shots(3), judge=judge, config=config())


def test_generic_qwen_boundary_judge_receives_existing_multimodal_evidence(
    tmp_path: Path,
) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.last_request = None

        def request(self, request):
            self.last_request = request
            return {
                "boundaries": [{
                    "gap_index": 0,
                    "is_scene_boundary": True,
                    "reason": "location changed",
                    "confidence": 0.8,
                    "evidence_used": ["caption", "images"],
                }]
            }

    context = []
    for index in range(2):
        paths = {}
        for role in ("representative", "early", "late"):
            path = tmp_path / f"{index}_{role}.jpg"
            Image.new("RGB", (32, 32), color=(index * 50, 20, 20)).save(path)
            paths[role] = path
        context.append({
            "shot_id": f"v_SH{index:05d}",
            "start_sec": float(index),
            "end_sec": float(index + 1),
            "representative_path": paths["representative"],
            "early_path": paths["early"],
            "late_path": paths["late"],
            "caption_vi": "Một cảnh",
            "caption_en": "A scene",
            "ocr_text": ["text"],
            "transcript": "speech",
        })
    client = RecordingClient()
    prompt_dir = Path(__file__).resolve().parents[1] / "prompts"
    judge = StructuredSceneBoundaryJudge(
        client,
        video_id="v",
        prompt_dir=prompt_dir,
        diagnostics_dir=tmp_path / "diagnostics",
        model_config={
            "prompt_version": "scene_boundary_primary_v1",
            "focused_prompt_version": "scene_boundary_focused_v1",
            "consistency_prompt_version": "scene_boundary_consistency_v1",
            "response_schema_version": "scene_boundary_response_v1",
        },
    )

    result = judge.judge(
        request_kind="focused_review",
        focus_gap_ids=("v_SH00000",),
        context=context,
    )

    assert result == {"v_SH00000": True}
    assert client.last_request.request_kind == "scene_boundary_focused_review"
    assert len(client.last_request.image_paths) == 1
    assert "ORDERED SHOT EVIDENCE" in client.last_request.prompt
