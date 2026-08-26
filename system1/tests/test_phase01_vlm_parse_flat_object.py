"""Parser tests for the flat-object replies that failed the 27/08 Kaggle run.

fixtures/scene_summary_failure_samples.json is verbatim from
kg14/phase04-smoke.log. All three carry a complete Vietnamese and English
summary and fail only because the model closed the object with `]`, then
echoed the prompt's own evidence after it.

Scene summaries have no array property, so none of the wrapper-based salvage
paths ever ran on them — the reply was lost with every field intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from system1.vlm.client import _parse_json_object

FIXTURE = Path(__file__).parent / "fixtures" / "scene_summary_failure_samples.json"
SAMPLES = json.loads(FIXTURE.read_text(encoding="utf-8"))

SCHEMA = {
    "type": "object",
    "properties": {
        "summary_vi": {"type": "string", "minLength": 1},
        "summary_en": {"type": "string", "minLength": 1},
    },
    "required": ["summary_vi", "summary_en"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("raw", SAMPLES, ids=range(len(SAMPLES)))
def test_every_recorded_failure_is_recovered(raw: str) -> None:
    parsed = _parse_json_object(raw, SCHEMA)
    assert parsed["summary_vi"].strip()
    assert parsed["summary_en"].strip()


@pytest.mark.parametrize("raw", SAMPLES, ids=range(len(SAMPLES)))
def test_recovery_keeps_what_the_model_wrote(raw: str) -> None:
    """The salvage must not truncate — these summaries run 250-360 characters."""
    parsed = _parse_json_object(raw, SCHEMA)
    assert raw.startswith('{"summary_vi": "' + parsed["summary_vi"][:40])
    assert len(parsed["summary_vi"]) > 100
    assert len(parsed["summary_en"]) > 100


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary_vi": "A", "summary_en": "B"]}',
        '{"summary_vi": "A", "summary_en": "B"]',
        '{"summary_vi": "A", "summary_en": "B"}\n\nOCR Text:\ntrailing evidence',
        '{"summary_vi": "A", "summary_en": "B"} stray "quoted" tail',
    ],
)
def test_the_shapes_the_model_actually_emits(raw: str) -> None:
    assert _parse_json_object(raw, SCHEMA) == {"summary_vi": "A", "summary_en": "B"}


def test_a_well_formed_reply_still_parses() -> None:
    raw = '{"summary_vi": "A", "summary_en": "B"}'
    assert _parse_json_object(raw, SCHEMA) == {"summary_vi": "A", "summary_en": "B"}


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary_vi": "A"}',
        '{"summary_vi": "A", "summary_en": "',
        '{"summary_vi": "", "summary_en": "B"}',
        '{"summary_vi": 1, "summary_en": "B"}',
        '{"summary_vi": "A", "summary_en": "B", "invented": 1}',
    ],
)
def test_salvage_never_invents_a_missing_field(raw: str) -> None:
    """Repairing punctuation must not repair content — these stay rejected."""
    with pytest.raises((ValidationError, json.JSONDecodeError)):
        _parse_json_object(raw, SCHEMA)


def test_a_reply_that_is_not_json_at_all_still_fails() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_json_object("the model wrote prose instead", SCHEMA)
