from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from system1.media.contact_sheet import COLUMNS, LABEL_HEIGHT, TILE_HEIGHT, TILE_WIDTH, write_contact_sheet
from system1.phase01.production import _build_scene_summaries
from system1.scenes.gemini_judge import StructuredSceneBoundaryJudge


def _make_image(path: Path) -> Path:
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(path)
    return path


class RaisingIfNotSingleImage:
    """Bắt chước đúng câu chặn thật của _call_vintern_many: khác 1 ảnh -> lỗi."""

    def __init__(self) -> None:
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        if len(request.image_paths) != 1:
            raise RuntimeError("vintern_local expects exactly one image per request")
        return {"summary_vi": "tom tat", "summary_en": "summary"}


def test_contact_sheet_ghep_du_o(tmp_path: Path) -> None:
    tiles = [(_make_image(tmp_path / f"img_{i}.jpg"), f"shot{i}") for i in range(12)]
    output = tmp_path / "sheet.jpg"

    result = write_contact_sheet(tiles, output)

    assert result == output
    assert output.exists()
    with Image.open(output) as sheet:
        rows = (12 + COLUMNS - 1) // COLUMNS
        assert sheet.size == (COLUMNS * TILE_WIDTH, rows * (TILE_HEIGHT + LABEL_HEIGHT))


def test_contact_sheet_mot_anh(tmp_path: Path) -> None:
    tiles = [(_make_image(tmp_path / "img_0.jpg"), "shot0")]
    output = tmp_path / "sheet.jpg"

    result = write_contact_sheet(tiles, output)

    assert result == output
    assert output.exists()
    with Image.open(output) as sheet:
        assert sheet.size == (COLUMNS * TILE_WIDTH, TILE_HEIGHT + LABEL_HEIGHT)


def test_contact_sheet_danh_sach_rong(tmp_path: Path) -> None:
    output = tmp_path / "sheet.jpg"

    result = write_contact_sheet([], output)

    assert result is None
    assert not output.exists()


def test_scene_12_shot_khong_nem_loi(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    keyframes_dir = stage_dir / "keyframes"
    keyframes_dir.mkdir(parents=True)

    shots = []
    keyframes = []
    captions = []
    for index in range(12):
        shot_id = f"v_SH{index:05d}"
        keyframe_name = f"{shot_id}_kf.jpg"
        _make_image(keyframes_dir / keyframe_name)
        shots.append({
            "shot_id": shot_id,
            "start_sec": float(index),
            "end_sec": float(index + 1),
        })
        keyframes.append({
            "shot_id": shot_id,
            "keyframe_id": f"{shot_id}:0",
            "keyframe_ref": keyframe_name,
            "is_representative": True,
        })
        captions.append({
            "shot_id": shot_id,
            "status": "pass",
            "caption_vi": "mo ta",
            "caption_en": "caption",
        })

    scene = {
        "scene_id": "v_SC00000",
        "video_id": "v",
        "start_sec": 0.0,
        "end_sec": 12.0,
    }
    client = RaisingIfNotSingleImage()

    rows = _build_scene_summaries(
        video_id="v",
        scenes=[scene],
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        captions=captions,
        asr_rows=[],
        scene_links=[],
        stage_dir=stage_dir,
        client=client,
        model_config={
            "prompt_version": "scene_summary_v2",
            "response_schema_version": "scene_summary_response_v1",
            "provider": "vintern_local",
            "model_id": "vintern",
            "model_revision": "1",
        },
        summary_config={
            "max_representative_images": 12,
            "image_sampling": "evenly_spaced_shots",
        },
    )

    assert len(rows) == 1
    assert len(client.requests) == 1
    assert len(client.requests[0].image_paths) == 1


def test_judge_vong_review_mot_anh(tmp_path: Path) -> None:
    class RecordingClient:
        def request(self, request):
            self.last_request = request
            return {
                "boundaries": [{
                    "gap_index": 0,
                    "is_scene_boundary": True,
                }]
            }

    context = []
    for index in range(2):
        paths = {}
        for role in ("representative", "early", "late"):
            path = tmp_path / f"{index}_{role}.jpg"
            _make_image(path)
            paths[role] = path
        context.append({
            "shot_id": f"v_SH{index:05d}",
            "start_sec": float(index),
            "end_sec": float(index + 1),
            "representative_path": paths["representative"],
            "early_path": paths["early"],
            "late_path": paths["late"],
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

    judge.judge(
        request_kind="focused_review",
        focus_gap_ids=("v_SH00000",),
        context=context,
    )

    assert len(client.last_request.image_paths) == 1


def test_scenes_hanh_vi_khong_doi(tmp_path: Path) -> None:
    # Xác nhận stage `scenes` (grouping) không đổi hành vi sau khi tách hàm
    # vẽ lưới ra dùng chung -- gọi lại chính test đã có ở
    # test_phase01_scene_grouping.py thay vì chép logic.
    from test_phase01_scene_grouping import (
        test_generic_qwen_boundary_judge_receives_existing_multimodal_evidence,
    )

    test_generic_qwen_boundary_judge_receives_existing_multimodal_evidence(tmp_path)
