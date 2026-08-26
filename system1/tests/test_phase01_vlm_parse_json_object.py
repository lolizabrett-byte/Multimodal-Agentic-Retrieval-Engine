"""Parser tests built from the three replies that failed the 26/08 Kaggle run.

The samples in fixtures/scene_boundary_failure_samples.json are verbatim from
kg9/worker2-phase04-smoke.log. Two of them carry every requested boundary and
fail only on a stray closing bracket; the third is genuinely truncated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from system1.vlm.client import _parse_json_object

FIXTURE = (
    Path(__file__).parent / "fixtures" / "scene_boundary_failure_samples.json"
)
SAMPLES = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _schema(item_count: int = 2) -> dict:
    """The judge's schema, trimmed to what the parser actually consults."""
    return {
        "type": "object",
        "properties": {
            "boundaries": {
                "type": "array",
                "minItems": item_count,
                "maxItems": item_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "gap_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": item_count - 1,
                        },
                        "is_scene_boundary": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "confidence": {"type": ["number", "null"]},
                        "evidence_used": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["gap_index", "is_scene_boundary"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["boundaries"],
        "additionalProperties": False,
    }


def test_well_formed_object_is_unchanged():
    raw = json.dumps(
        {
            "boundaries": [
                {"gap_index": 0, "is_scene_boundary": True},
                {"gap_index": 1, "is_scene_boundary": False},
            ]
        }
    )
    assert len(_parse_json_object(raw, _schema())["boundaries"]) == 2


def test_bare_array_without_trailing_junk_is_unchanged():
    raw = json.dumps(
        [
            {"gap_index": 0, "is_scene_boundary": True},
            {"gap_index": 1, "is_scene_boundary": False},
        ]
    )
    assert len(_parse_json_object(raw, _schema())["boundaries"]) == 2


@pytest.mark.parametrize("sample_key", ["len_477", "len_686"])
def test_trailing_bracket_keeps_the_boundaries_already_parsed(sample_key):
    """Both replies answered every gap; only a stray bracket follows them."""
    raw = SAMPLES[sample_key]["raw_text"]
    parsed = _parse_json_object(raw, _schema())
    assert [item["gap_index"] for item in parsed["boundaries"]] == [0, 1]


def test_truncated_reply_still_raises():
    """len_440 stops mid-key, so the second boundary never arrived."""
    with pytest.raises(Exception):
        _parse_json_object(SAMPLES["len_440"]["raw_text"], _schema())


@pytest.mark.parametrize("raw", ["[", "[[[", "", "not json at all"])
def test_unusable_text_still_raises(raw):
    with pytest.raises(Exception):
        _parse_json_object(raw, _schema())


def test_too_few_items_still_raises():
    raw = json.dumps([{"gap_index": 0, "is_scene_boundary": True}])
    with pytest.raises(Exception):
        _parse_json_object(raw, _schema())


def _schema_v4(item_count: int = 2) -> dict:
    """The v4 judge schema: the two fields grouping.py actually consumes."""
    schema = _schema(item_count)
    properties = schema["properties"]["boundaries"]["items"]["properties"]
    for dropped in ("reason", "confidence", "evidence_used"):
        del properties[dropped]
    return schema


def test_v4_schema_accepts_a_two_field_reply():
    raw = json.dumps(
        [
            {"gap_index": 0, "is_scene_boundary": True},
            {"gap_index": 1, "is_scene_boundary": False},
        ]
    )
    parsed = _parse_json_object(raw, _schema_v4())
    assert [item["is_scene_boundary"] for item in parsed["boundaries"]] == [
        True,
        False,
    ]


def test_v4_schema_drops_fields_the_model_volunteers():
    """The model may still emit a reason; it is trimmed, not rejected."""
    raw = json.dumps(
        [
            {"gap_index": 0, "is_scene_boundary": True, "reason": "setting change"},
            {"gap_index": 1, "is_scene_boundary": False, "reason": "same event"},
        ]
    )
    parsed = _parse_json_object(raw, _schema_v4())
    assert all("reason" not in item for item in parsed["boundaries"])


def test_v4_schema_still_requires_the_verdict():
    raw = json.dumps([{"gap_index": 0}, {"gap_index": 1}])
    with pytest.raises(Exception):
        _parse_json_object(raw, _schema_v4())
