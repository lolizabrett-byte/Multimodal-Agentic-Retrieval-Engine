"""Judge trả mảng trần thay vì object — lỗi chặn đo được trên Kaggle 26/08.

Vintern-3B trả `[{...}, {...}]` trong khi prompt đòi `{"boundaries": [...]}`.
Parser cũ thấy chuỗi không bắt đầu bằng `{` nên cắt từ `{` đầu tới `}` cuối,
làm nát mảng. Cả 3 video trong lượt chạy thử đều chết vì việc này.
"""

from __future__ import annotations

import pytest

from system1.vlm.client import _parse_json_object

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "boundaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "after_shot_id": {"type": "string"},
                    "is_scene_boundary": {"type": "boolean"},
                },
                "required": ["after_shot_id", "is_scene_boundary"],
            },
        }
    },
    "required": ["boundaries"],
}

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {"caption_vi": {"type": "string"}, "caption_en": {"type": "string"}},
    "required": ["caption_vi", "caption_en"],
}


def test_mang_tran_duoc_boc_thanh_object():
    # Nguyên văn hình dạng model trả trên Kaggle (video L22_V001).
    raw = (
        '[{"after_shot_id": "L22_V001_SH00008", "is_scene_boundary": true}, '
        '{"after_shot_id": "L22_V001_SH00009", "is_scene_boundary": false}]'
    )

    result = _parse_json_object(raw, BOUNDARY_SCHEMA)

    assert [item["after_shot_id"] for item in result["boundaries"]] == [
        "L22_V001_SH00008",
        "L22_V001_SH00009",
    ]


def test_object_dung_van_giu_nguyen():
    raw = '{"boundaries": [{"after_shot_id": "A", "is_scene_boundary": true}]}'

    assert len(_parse_json_object(raw, BOUNDARY_SCHEMA)["boundaries"]) == 1


def test_mang_rong_van_hop_le():
    assert _parse_json_object("[]", BOUNDARY_SCHEMA) == {"boundaries": []}


def test_khong_boc_khi_schema_co_nhieu_truong_bat_buoc():
    """Schema hai trường thì không đoán được bọc vào đâu — phải từ chối."""
    with pytest.raises(Exception):
        _parse_json_object('[{"x": 1}]', CAPTION_SCHEMA)


def test_object_lan_trong_text_thua_van_cat_duoc():
    raw = 'Here you go: {"caption_vi": "a", "caption_en": "b"} hope that helps'

    assert _parse_json_object(raw, CAPTION_SCHEMA)["caption_vi"] == "a"


def test_mang_tran_kem_fence_markdown():
    raw = '```json\n[{"after_shot_id": "A", "is_scene_boundary": true}]\n```'

    assert len(_parse_json_object(raw, BOUNDARY_SCHEMA)["boundaries"]) == 1
