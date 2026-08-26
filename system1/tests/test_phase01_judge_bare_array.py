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


STRICT_SCHEMA = {
    "type": "object",
    "properties": {
        "boundaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "after_shot_id": {"type": "string"},
                    "is_scene_boundary": {"type": "boolean"},
                    "evidence_used": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["after_shot_id", "is_scene_boundary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["boundaries"],
}


def test_bo_key_la_model_go_sai_ten():
    """Model gõ `evidences_used`; key này tuỳ chọn nên bỏ đi là an toàn."""
    raw = '[{"after_shot_id": "A", "is_scene_boundary": true, "evidences_used": []}]'

    result = _parse_json_object(raw, STRICT_SCHEMA)

    assert result["boundaries"] == [{"after_shot_id": "A", "is_scene_boundary": True}]


def test_hai_mang_noi_nhau_duoc_gop():
    raw = (
        '[{"after_shot_id": "A", "is_scene_boundary": true}], '
        '[{"after_shot_id": "B", "is_scene_boundary": false}]'
    )

    result = _parse_json_object(raw, STRICT_SCHEMA)

    assert [item["after_shot_id"] for item in result["boundaries"]] == ["A", "B"]


def test_thieu_truong_bat_buoc_van_phai_hong():
    """Bỏ key thừa được, nhưng thiếu key bắt buộc thì không được bịa ra."""
    with pytest.raises(Exception):
        _parse_json_object('[{"after_shot_id": "A"}]', STRICT_SCHEMA)


GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "boundaries": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "gap_index": {"type": "integer", "minimum": 0, "maximum": 1},
                    "is_scene_boundary": {"type": "boolean"},
                },
                "required": ["gap_index", "is_scene_boundary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["boundaries"],
}


def test_wrapper_lap_lai_moi_phan_tu():
    """Model lặp lại lớp bọc cho từng entry — gom lại thay vì bỏ cả câu trả lời."""
    raw = (
        '{"boundaries": [{"gap_index": 0, "is_scene_boundary": true}], '
        '{"gap_index": 1, "is_scene_boundary": false}}'
    )

    result = _parse_json_object(raw, GAP_SCHEMA)

    assert [item["gap_index"] for item in result["boundaries"]] == [0, 1]


def test_object_dung_khong_bi_dong_vao():
    raw = (
        '{"boundaries": [{"gap_index": 0, "is_scene_boundary": true}, '
        '{"gap_index": 1, "is_scene_boundary": false}]}'
    )

    assert len(_parse_json_object(raw, GAP_SCHEMA)["boundaries"]) == 2


def test_thieu_entry_van_phai_hong():
    """Gom được không có nghĩa là bịa thêm cho đủ."""
    with pytest.raises(Exception):
        _parse_json_object('[{"gap_index": 0, "is_scene_boundary": true}]', GAP_SCHEMA)
