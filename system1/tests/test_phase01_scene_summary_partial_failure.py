from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from system1.phase01.production import _build_scene_summaries
from system1.phase01.qa import write_manual_review_report
from system1.phase01.validation import validate_rows

VIDEO_ID = "L21_V001"


def _model_config() -> dict[str, Any]:
    return {
        "provider": "vintern_local",
        "model_id": "5CD-AI/Vintern-3B-R-beta",
        "model_revision": "4fd34d713dfca446cdecc00d921f5038909e3efb",
        "prompt_version": "scene_summary_v2",
        "response_schema_version": "scene_summary_response_v1",
    }


def _summary_response(index: int) -> dict[str, Any]:
    return {
        "summary_vi": f"Tom tat canh {index}",
        "summary_en": f"Scene summary {index}",
        "__provider": "gemini",
        "__model_id": "gemini-3.6-flash",
        "__model_revision": "gemini-3.6-flash",
    }


def _scenes_shots_keyframes(tmp_path: Path, scene_count: int) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    scenes = [
        {
            "scene_id": f"scene_{index:03d}",
            "start_sec": float(index),
            "end_sec": float(index) + 1.0,
        }
        for index in range(scene_count)
    ]
    shots = [
        {
            "shot_id": f"shot_{index:03d}",
            "start_sec": float(index),
            "end_sec": float(index) + 1.0,
        }
        for index in range(scene_count)
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
    keyframes_dir = tmp_path / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    for index in range(scene_count):
        Image.new("RGB", (64, 48), color=(10, 20, 30)).save(
            keyframes_dir / f"frame_{index:03d}.jpg"
        )
    return scenes, shots, keyframes


class _PartialFailureClient:
    """Mirrors the caption stage's per-item failure contract but for the
    scene_summary stage's single `client.request()` call (no batching)."""

    def __init__(self, failing_scene_ids: set[str], error_factory=None) -> None:
        self.failing_scene_ids = failing_scene_ids
        self._error_factory = error_factory or (
            lambda scene_id: ValueError(f"model refused scene {scene_id}")
        )
        self.calls = 0

    def request(self, request):
        self.calls += 1
        scene_id = str(request.identity["scene_id"])
        if scene_id in self.failing_scene_ids:
            raise self._error_factory(scene_id)
        index = int(scene_id.rsplit("_", 1)[-1])
        return _summary_response(index)


def _build(
    tmp_path: Path,
    scene_count: int,
    failing_scene_ids: set[str],
    summary_config: dict[str, Any] | None = None,
    error_factory=None,
):
    scenes, shots, keyframes = _scenes_shots_keyframes(tmp_path, scene_count)
    client = _PartialFailureClient(failing_scene_ids, error_factory=error_factory)
    rows = _build_scene_summaries(
        video_id=VIDEO_ID,
        scenes=scenes,
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        captions=[],
        asr_rows=[],
        scene_links=[],
        stage_dir=tmp_path,
        client=client,
        model_config=_model_config(),
        summary_config=summary_config or {"max_representative_images": 12, "max_failed_ratio": 0.5},
    )
    return scenes, shots, rows


# --- Unit -------------------------------------------------------------------


def test_khong_hong_thi_khong_doi(tmp_path: Path) -> None:
    _scenes, _shots, rows = _build(tmp_path, scene_count=3, failing_scene_ids=set())

    assert len(rows) == 3
    assert all(row["status"] == "pass" for row in rows)
    assert all(row["summary_vi"] and row["summary_en"] for row in rows)


def test_mot_scene_hong_van_du_dong(tmp_path: Path) -> None:
    _scenes, _shots, rows = _build(
        tmp_path, scene_count=9, failing_scene_ids={"scene_004"}
    )

    assert len(rows) == 9
    statuses = [row["status"] for row in rows]
    assert statuses.count("failed") == 1
    assert statuses.count("pass") == 8
    failed_row = next(row for row in rows if row["status"] == "failed")
    assert failed_row["summary_vi"] == ""
    assert failed_row["summary_en"] == ""


def test_scene_id_khop_tuyet_doi(tmp_path: Path) -> None:
    scenes, _shots, rows = _build(
        tmp_path, scene_count=9, failing_scene_ids={"scene_002", "scene_007"}
    )

    assert {row["scene_id"] for row in rows} == {
        scene["scene_id"] for scene in scenes
    }
    failed_ids = {row["scene_id"] for row in rows if row["status"] == "failed"}
    assert failed_ids == {"scene_002", "scene_007"}


def test_qua_nguong_thi_nem_loi(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="max_failed_ratio"):
        _build(
            tmp_path,
            scene_count=4,
            failing_scene_ids={"scene_000", "scene_001", "scene_002"},
        )


def test_duoi_nguong_thi_di_tiep(tmp_path: Path) -> None:
    _scenes, _shots, rows = _build(
        tmp_path, scene_count=4, failing_scene_ids={"scene_000"}
    )

    assert len(rows) == 4
    assert sum(1 for row in rows if row["status"] == "failed") == 1


def test_moi_scene_deu_hong(tmp_path: Path) -> None:
    all_scene_ids = {f"scene_{index:03d}" for index in range(3)}
    with pytest.raises(RuntimeError, match="max_failed_ratio"):
        _build(tmp_path, scene_count=3, failing_scene_ids=all_scene_ids)


def test_loi_ha_tang_nem_ra_ngoai_khong_thanh_failed(tmp_path: Path) -> None:
    def _oom_error(_scene_id: str) -> Exception:
        return RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _build(
            tmp_path,
            scene_count=3,
            failing_scene_ids={"scene_001"},
            error_factory=_oom_error,
        )


# --- Schema -------------------------------------------------------------


def test_schema_chap_nhan_dong_failed() -> None:
    row = {
        "scene_id": "scene_000", "video_id": VIDEO_ID,
        "summary_vi": "", "summary_en": "",
        "provider": "vintern_local", "model_name": "5CD-AI/Vintern-3B-R-beta",
        "model_version": "4fd34d713dfca446cdecc00d921f5038909e3efb",
        "prompt_version": "scene_summary_v2",
        "schema_version": "scene_summary_response_v1",
        "confidence": None, "status": "failed",
    }
    validate_rows("scene_summaries", [row])


def test_schema_van_chan_pass_rong() -> None:
    row = {
        "scene_id": "scene_000", "video_id": VIDEO_ID,
        "summary_vi": "", "summary_en": "Non-empty",
        "provider": "vintern_local", "model_name": "5CD-AI/Vintern-3B-R-beta",
        "model_version": "4fd34d713dfca446cdecc00d921f5038909e3efb",
        "prompt_version": "scene_summary_v2",
        "schema_version": "scene_summary_response_v1",
        "confidence": None, "status": "pass",
    }
    with pytest.raises(ValueError, match="violates canonical schema"):
        validate_rows("scene_summaries", [row])


# --- Integration ----------------------------------------------------------


def test_qa_bo_qua_dong_failed(tmp_path: Path) -> None:
    import zipfile

    import pandas as pd

    release_dir = tmp_path / "release"
    release_dir.mkdir()
    artifact_path = release_dir / f"{VIDEO_ID}.zip"
    scenes = pd.DataFrame([
        {"scene_id": "scene_000", "scene_index": 0, "start_sec": 0.0, "end_sec": 1.0},
        {"scene_id": "scene_001", "scene_index": 1, "start_sec": 1.0, "end_sec": 2.0},
    ])
    summaries = pd.DataFrame([
        {
            "scene_id": "scene_000", "video_id": VIDEO_ID,
            "summary_vi": "Tom tat", "summary_en": "Summary",
            "status": "pass",
        },
        {
            "scene_id": "scene_001", "video_id": VIDEO_ID,
            "summary_vi": "", "summary_en": "",
            "status": "failed",
        },
    ])
    shot_captions = pd.DataFrame([
        {
            "shot_id": "shot_000", "video_id": VIDEO_ID,
            "keyframe_ref": "media://keyframes/frame_000.jpg",
            "caption_vi": "Cap", "caption_en": "Cap",
            "objects_vi": [], "objects_en": [], "actions_vi": [], "actions_en": [],
            "visible_text_summary_vi": "", "visible_text_summary_en": "",
            "quality_score": 1.0,
        }
    ])
    keyframes = pd.DataFrame([
        {
            "keyframe_id": "keyframe_000:0",
            "shot_id": "shot_000",
            "keyframe_ref": "media://keyframes/frame_000.jpg",
            "is_representative": True,
            "quality_score": 1.0,
        }
    ])
    ocr = pd.DataFrame([{"keyframe_id": "keyframe_000:0", "text": ""}])

    with zipfile.ZipFile(artifact_path, "w") as archive:
        archive.writestr(
            f"{VIDEO_ID}/scenes.parquet", scenes.to_parquet(index=False)
        )
        archive.writestr(
            f"{VIDEO_ID}/scene_summaries.parquet", summaries.to_parquet(index=False)
        )
        archive.writestr(
            f"{VIDEO_ID}/shot_captions.parquet", shot_captions.to_parquet(index=False)
        )
        archive.writestr(
            f"{VIDEO_ID}/keyframes.parquet", keyframes.to_parquet(index=False)
        )
        archive.writestr(f"{VIDEO_ID}/ocr.parquet", ocr.to_parquet(index=False))

    report_path = write_manual_review_report(
        release_dir=release_dir,
        batch_id="batch_001",
        worker_id="worker_001",
        video_results=[{"video_id": VIDEO_ID, "status": "complete", "artifact": str(artifact_path)}],
        sample_size=12,
    )
    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    scene_summary_ids = {
        item["entity_id"]
        for item in payload["samples"]
        if item["review_kind"] == "scene_summary"
    }
    assert scene_summary_ids == {"scene_000"}
    assert "scene_001" not in scene_summary_ids


def test_validation_qua_duoc(tmp_path: Path) -> None:
    scenes, _shots, rows = _build(
        tmp_path, scene_count=9, failing_scene_ids={"scene_003", "scene_006"}
    )
    validate_rows("scene_summaries", rows)


def test_merge_khong_sinh_text_source_rong_tu_dong_failed() -> None:
    import pandas as pd

    from system1.release.merge import _build_structure_text_sources

    scene_summaries = pd.DataFrame([
        {
            "scene_id": "scene_000", "video_id": VIDEO_ID,
            "summary_vi": "Tom tat", "summary_en": "Summary",
            "provider": "vintern_local", "model_name": "Vintern",
            "status": "pass",
        },
        {
            "scene_id": "scene_001", "video_id": VIDEO_ID,
            "summary_vi": "", "summary_en": "",
            "provider": "vintern_local", "model_name": "Vintern",
            "status": "failed",
        },
    ])
    text_sources = _build_structure_text_sources(
        asr=pd.DataFrame(columns=["video_id", "text"]),
        ocr=pd.DataFrame(columns=["video_id", "keyframe_id", "text"]),
        shot_captions=pd.DataFrame(
            columns=["video_id", "shot_id", "caption_vi", "caption_en"]
        ),
        scene_summaries=scene_summaries,
    )
    scene_summary_rows = text_sources[text_sources["source_type"] == "scene_summary"]
    assert set(scene_summary_rows["entity_id"]) == {"scene_000"}


# --- Edge cases -------------------------------------------------------------


def test_video_mot_scene_hong_nem_loi(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="max_failed_ratio"):
        _build(tmp_path, scene_count=1, failing_scene_ids={"scene_000"})
