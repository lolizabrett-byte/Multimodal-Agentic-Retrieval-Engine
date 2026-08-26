"""Judge trả vị trí, không trả shot id.

Lượt chạy Kaggle 26/08: model bịa id của video khác (`L20_V019` trong video
`L21_V019`), `_validate_judgement` từ chối, cả 3 video chết ở stage `scenes`.

Bắt model chép lại id vốn là việc thừa — judge đã biết `focus_gap_ids` và thứ
tự của chúng. Nay model trả `gap_index`, một số nguyên trong `[0, N)`. Nó
không thể bịa một con số ngoài khoảng, và sai thì bắt được bằng phép toán.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from system1.scenes.gemini_judge import StructuredSceneBoundaryJudge

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"

MODEL_CONFIG = {
    "prompt_version": "scene_boundary_primary_v3",
    "focused_prompt_version": "scene_boundary_focused_v3",
    "consistency_prompt_version": "scene_boundary_consistency_v3",
    "response_schema_version": "scene_boundary_response_v1",
}


class _ScriptedClient:
    """Trả sẵn một phản hồi, và giữ lại schema để test soi."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[Any] = []

    def request(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return self.payload


def _judge(payload: dict[str, Any], tmp_path: Path) -> StructuredSceneBoundaryJudge:
    return StructuredSceneBoundaryJudge(
        _ScriptedClient(payload),
        video_id="L21_V019",
        prompt_dir=PROMPT_DIR,
        diagnostics_dir=tmp_path / "diag",
        model_config=MODEL_CONFIG,
    )


def _context(count: int, tmp_path: Path) -> list[dict[str, Any]]:
    from PIL import Image

    rows = []
    for index in range(count):
        path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (32, 18), (index * 20 % 255, 100, 150)).save(path)
        rows.append(
            {
                "shot_id": f"L21_V019_SH{index:05d}",
                "start_sec": float(index),
                "end_sec": float(index) + 1.0,
                "representative_path": path,
            }
        )
    return rows


def test_gap_index_anh_xa_theo_vi_tri(tmp_path: Path) -> None:
    gaps = ("L21_V019_SH00000", "L21_V019_SH00001", "L21_V019_SH00002")
    payload = {
        "boundaries": [
            {"gap_index": 0, "is_scene_boundary": True},
            {"gap_index": 1, "is_scene_boundary": False},
            {"gap_index": 2, "is_scene_boundary": True},
        ]
    }

    result = _judge(payload, tmp_path).judge(
        request_kind="primary", focus_gap_ids=gaps, context=_context(4, tmp_path)
    )

    assert result == {
        "L21_V019_SH00000": True,
        "L21_V019_SH00001": False,
        "L21_V019_SH00002": True,
    }


def test_thu_tu_tra_ve_khong_quan_trong(tmp_path: Path) -> None:
    """gap_index mang thông tin vị trí, nên model trả lộn xộn vẫn khớp đúng."""
    gaps = ("L21_V019_SH00000", "L21_V019_SH00001")
    payload = {
        "boundaries": [
            {"gap_index": 1, "is_scene_boundary": True},
            {"gap_index": 0, "is_scene_boundary": False},
        ]
    }

    result = _judge(payload, tmp_path).judge(
        request_kind="primary", focus_gap_ids=gaps, context=_context(3, tmp_path)
    )

    assert result == {"L21_V019_SH00000": False, "L21_V019_SH00001": True}


def test_index_ngoai_khoang_bi_tu_choi(tmp_path: Path) -> None:
    gaps = ("L21_V019_SH00000", "L21_V019_SH00001")
    payload = {"boundaries": [{"gap_index": 0, "is_scene_boundary": True},
                              {"gap_index": 7, "is_scene_boundary": False}]}

    with pytest.raises(ValueError, match="out-of-range"):
        _judge(payload, tmp_path).judge(
            request_kind="primary", focus_gap_ids=gaps, context=_context(3, tmp_path)
        )


def test_index_trung_bi_tu_choi(tmp_path: Path) -> None:
    gaps = ("L21_V019_SH00000", "L21_V019_SH00001")
    payload = {"boundaries": [{"gap_index": 0, "is_scene_boundary": True},
                              {"gap_index": 0, "is_scene_boundary": False}]}

    with pytest.raises(ValueError, match="duplicated"):
        _judge(payload, tmp_path).judge(
            request_kind="primary", focus_gap_ids=gaps, context=_context(3, tmp_path)
        )


def test_schema_gui_di_chi_cho_phep_so_trong_khoang(tmp_path: Path) -> None:
    """Model không còn cửa để bịa chuỗi như `L20_V019_SH00004`."""
    gaps = ("L21_V019_SH00000", "L21_V019_SH00001")
    client = _ScriptedClient(
        {"boundaries": [{"gap_index": 0, "is_scene_boundary": True},
                        {"gap_index": 1, "is_scene_boundary": False}]}
    )
    judge = StructuredSceneBoundaryJudge(
        client, video_id="L21_V019", prompt_dir=PROMPT_DIR,
        diagnostics_dir=tmp_path / "diag", model_config=MODEL_CONFIG,
    )

    judge.judge(request_kind="primary", focus_gap_ids=gaps, context=_context(3, tmp_path))

    item_schema = client.requests[0].response_schema["properties"]["boundaries"]["items"]
    assert "after_shot_id" not in item_schema["properties"]
    assert item_schema["properties"]["gap_index"] == {
        "type": "integer", "minimum": 0, "maximum": 1
    }
    assert item_schema["required"] == ["gap_index", "is_scene_boundary"]


def test_prompt_v3_khong_con_nhac_shot_id() -> None:
    for name in ("primary", "focused", "consistency"):
        text = (PROMPT_DIR / f"scene_boundary_{name}_v3.txt").read_text(encoding="utf-8")
        assert "after_shot_id" not in text
        assert "gap_index" in text
