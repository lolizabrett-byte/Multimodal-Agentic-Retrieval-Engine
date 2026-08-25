from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from system1.config import load_configs
from system1.phase01.production import (
    _build_captions,
    _build_scene_evidence,
)
from system1.phase01.validation import validate_rows
from system1.scenes import group_scenes
from system1.vlm.client import BatchRequestError

SYSTEM1_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = SYSTEM1_ROOT / "configs"


def _caption_response(index: int) -> dict[str, Any]:
    return {
        "caption_vi": f"Cảnh {index}",
        "caption_en": f"Scene {index}",
        "objects_vi": [],
        "objects_en": [],
        "actions_vi": [],
        "actions_en": [],
        "visible_text_summary_vi": "",
        "visible_text_summary_en": "",
        "scene_type": "unknown",
        "__provider": "qwen_local",
        "__model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "__model_revision": "revision",
    }


def _shots_and_keyframes(count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shots = [
        {
            "shot_id": f"shot_{index:03d}",
            "start_sec": float(index),
            "end_sec": float(index) + 1.0,
        }
        for index in range(count)
    ]
    keyframes = [
        {
            "shot_id": shot["shot_id"],
            "keyframe_id": f"keyframe_{index:03d}:{index}",
            "keyframe_ref": f"media://keyframes/frame_{index:03d}.jpg",
            "timestamp_sec": float(index),
            "is_representative": True,
            "keyframe_role": "middle",
        }
        for index, shot in enumerate(shots)
    ]
    return shots, keyframes


class _PartialFailureClient:
    """Mirrors the OCR stage's BatchRequestError contract: N requests in,
    a mix of completed results and per-index errors out."""

    def __init__(self, failing_indexes: set[int]) -> None:
        self.failing_indexes = failing_indexes

    def request_many(self, requests):
        results: list[dict[str, Any] | None] = []
        errors: dict[int, Exception] = {}
        for index, _request in enumerate(requests):
            if index in self.failing_indexes:
                results.append(None)
                errors[index] = RuntimeError(f"provider error on request {index}")
            else:
                results.append(_caption_response(index))
        if errors:
            raise BatchRequestError(results=results, errors=errors)
        return results


def _model_config() -> dict[str, Any]:
    return dict(load_configs(CONFIG_DIR)["models"]["phase01"]["shot_caption"])


def _build(tmp_path: Path, shot_count: int, failing_indexes: set[int], caption_config=None):
    shots, keyframes = _shots_and_keyframes(shot_count)
    client = _PartialFailureClient(failing_indexes)
    rows = _build_captions(
        video_id="L21_V001",
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        stage_dir=tmp_path,
        client=client,
        model_config=_model_config(),
        max_concurrency=2,
        caption_config=caption_config,
    )
    return shots, keyframes, rows


# --- Unit: schema ---------------------------------------------------------


def test_failed_caption_row_is_valid_against_schema() -> None:
    row = {
        "shot_caption_id": "shot_000_caption", "video_id": "L21_V001",
        "shot_id": "shot_000", "representative_keyframe_id": "keyframe_000:0",
        "representative_timestamp_sec": 0.0, "caption_vi": "", "caption_en": "",
        "objects_vi": [], "objects_en": [], "actions_vi": [], "actions_en": [],
        "visible_text_summary_vi": "", "visible_text_summary_en": "",
        "scene_type": "", "provider": "qwen_local",
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "model_version": "revision",
        "prompt_version": "shot_caption_v2", "schema_version": "shot_caption_response_v2",
        "confidence": None, "status": "failed",
    }
    validate_rows("shot_captions", [row])


def test_pass_caption_row_missing_caption_vi_is_still_rejected() -> None:
    row = {
        "shot_caption_id": "shot_000_caption", "video_id": "L21_V001",
        "shot_id": "shot_000", "representative_keyframe_id": "keyframe_000:0",
        "representative_timestamp_sec": 0.0, "caption_vi": "", "caption_en": "A scene",
        "objects_vi": [], "objects_en": [], "actions_vi": [], "actions_en": [],
        "visible_text_summary_vi": "", "visible_text_summary_en": "",
        "scene_type": "unknown", "provider": "qwen_local",
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "model_version": "revision",
        "prompt_version": "shot_caption_v2", "schema_version": "shot_caption_response_v2",
        "confidence": None, "status": "pass",
    }
    with pytest.raises(ValueError, match="violates canonical schema"):
        validate_rows("shot_captions", [row])


# --- Integration ------------------------------------------------------------


def test_partial_batch_failure_keeps_successful_shots_and_marks_the_rest_failed(
    tmp_path: Path,
) -> None:
    shots, keyframes, rows = _build(tmp_path, shot_count=12, failing_indexes=set(range(5)))

    assert len(rows) == 12
    statuses = [row["status"] for row in rows]
    assert statuses.count("pass") == 7
    assert statuses.count("failed") == 5
    validate_rows("shot_captions", rows)


def test_scene_evidence_keeps_one_row_per_shot_when_captions_fail(
    tmp_path: Path,
) -> None:
    # group_scenes indexes evidence by shot position. Dropping a row would
    # misalign every later shot, so failed captions become placeholders.
    shots, keyframes, rows = _build(tmp_path, shot_count=12, failing_indexes=set(range(5)))

    evidence = _build_scene_evidence(
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        captions=rows,
        asr_rows=[],
        links=[],
        stage_dir=tmp_path,
    )

    assert len(evidence) == len(shots)
    assert [row["shot_id"] for row in evidence] == [
        str(shot["shot_id"]) for shot in shots
    ]


def test_scene_evidence_marks_failed_shots_with_empty_text_fields(
    tmp_path: Path,
) -> None:
    shots, keyframes, rows = _build(tmp_path, shot_count=12, failing_indexes={0, 3, 7})

    evidence = _build_scene_evidence(
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        captions=rows,
        asr_rows=[],
        links=[],
        stage_dir=tmp_path,
    )

    failed = [row for row in evidence if row["caption_missing"]]
    assert {row["shot_id"] for row in failed} == {"shot_000", "shot_003", "shot_007"}
    for row in failed:
        assert row["caption_vi"] == ""
        assert row["caption_en"] == ""
        assert row["objects_vi"] == []
        assert row["actions_en"] == []
        # Keyframe evidence survives — the judge can still see the image.
        assert row["representative_path"] is not None

    for row in evidence:
        if not row["caption_missing"]:
            assert row["caption_vi"] != ""


def test_scene_grouping_runs_end_to_end_with_failed_captions(
    tmp_path: Path,
) -> None:
    # Regression for the blocking ValueError that killed the L23_V011 run:
    # 28 shots, 9 failed captions -> 19 evidence rows -> crash at stage `scenes`.
    shots, keyframes, rows = _build(tmp_path, shot_count=12, failing_indexes={0, 3, 7})
    for index, shot in enumerate(shots):
        shot.update(
            video_id="L23_V011",
            shot_index=index,
            start_frame=index * 25,
            end_frame=(index + 1) * 25,
        )

    evidence = _build_scene_evidence(
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        captions=rows,
        asr_rows=[],
        links=[],
        stage_dir=tmp_path,
    )

    class _NoBoundaryJudge:
        def judge(self, *, request_kind, focus_gap_ids, context):
            return {gap_id: False for gap_id in focus_gap_ids}

    config = yaml.safe_load(
        (CONFIG_DIR / "phase01.yaml").read_text(encoding="utf-8")
    )["scene_grouping"]

    scenes, _decisions = group_scenes(
        video_id="L23_V011",
        shots=shots,
        evidence=evidence,
        judge=_NoBoundaryJudge(),
        config=config,
    )

    assert sum(int(scene["shot_count"]) for scene in scenes) == len(shots)


def test_all_shots_failing_raises_instead_of_shipping_an_empty_table(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="max_failed_ratio"):
        _build(tmp_path, shot_count=12, failing_indexes=set(range(12)))


def test_failure_ratio_above_configured_threshold_fails_even_with_some_success(
    tmp_path: Path,
) -> None:
    # 4/12 fail (~33%) is within the default 0.5 threshold, but a stricter
    # 0.2 threshold configured for this video must still reject it.
    with pytest.raises(RuntimeError, match="max_failed_ratio"):
        _build(
            tmp_path,
            shot_count=12,
            failing_indexes=set(range(4)),
            caption_config={"max_failed_ratio": 0.2},
        )


# --- Edge cases ---------------------------------------------------------


def test_zero_shots_returns_empty_rows_without_crashing(tmp_path: Path) -> None:
    _shots, _keyframes, rows = _build(tmp_path, shot_count=0, failing_indexes=set())
    assert rows == []


def test_single_failing_shot_exceeds_ratio_and_fails_the_stage(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="max_failed_ratio"):
        _build(tmp_path, shot_count=1, failing_indexes={0})
