from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from system1.config import load_configs
from system1.phase01.production import _build_captions
from system1.phase01.validation import validate_rows
from system1.release.merge import _build_structure_text_sources

SYSTEM1_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = SYSTEM1_ROOT / "configs"
PROMPTS_DIR = SYSTEM1_ROOT / "prompts"

DROPPED_FIELDS = {"visible_text_summary_vi", "visible_text_summary_en"}
KEPT_FIELDS = {
    "caption_vi",
    "caption_en",
    "objects_vi",
    "objects_en",
    "actions_vi",
    "actions_en",
    "scene_type",
}


def _prompt_json_keys(prompt_text: str) -> set[str]:
    match = re.search(r"\{.*\}", prompt_text)
    assert match, "prompt must contain a JSON example object"
    payload = json.loads(match.group(0))
    return set(payload.keys())


def test_prompt_v4_khong_con_2_truong() -> None:
    text = (PROMPTS_DIR / "shot_caption_v4.txt").read_text(encoding="utf-8")
    assert "visible_text_summary_vi" not in text
    assert "visible_text_summary_en" not in text


def test_prompt_v4_van_du_7_truong() -> None:
    text = (PROMPTS_DIR / "shot_caption_v4.txt").read_text(encoding="utf-8")
    keys = _prompt_json_keys(text)
    assert keys == KEPT_FIELDS


def test_schema_khop_prompt_v4() -> None:
    """Canh cái bẫy: schema trong mã phải khớp đúng prompt v4, không được
    sót 2 trường visible_text_summary_* trong required (nếu sót -> model
    không sinh -> validation trượt toàn bộ)."""
    model_config = dict(load_configs(CONFIG_DIR)["models"]["phase01"]["shot_caption"])
    assert model_config["prompt_version"] == "shot_caption_v4"

    prompt_text = (PROMPTS_DIR / f"{model_config['prompt_version']}.txt").read_text(
        encoding="utf-8"
    )
    prompt_keys = _prompt_json_keys(prompt_text)

    shots = [{"shot_id": "shot_000", "start_sec": 0.0, "end_sec": 1.0}]
    keyframes = [
        {
            "shot_id": "shot_000",
            "keyframe_id": "keyframe_000:0",
            "keyframe_ref": "media://keyframes/frame_000.jpg",
            "timestamp_sec": 0.0,
            "is_representative": True,
            "keyframe_role": "middle",
        }
    ]
    captured_schema: dict[str, Any] = {}

    class _CaptureClient:
        def request_many(self, requests):
            captured_schema.update(requests[0].response_schema)
            return [_v4_response(0)]

    _build_captions(
        video_id="L21_V001",
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        stage_dir=SYSTEM1_ROOT,
        client=_CaptureClient(),
        model_config=model_config,
        max_concurrency=2,
    )

    schema_required = set(captured_schema["required"])
    schema_properties = set(captured_schema["properties"].keys())

    assert schema_required == prompt_keys
    assert schema_properties == prompt_keys
    assert not (schema_required & DROPPED_FIELDS)
    assert captured_schema["additionalProperties"] is False


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


def _v4_response(index: int) -> dict[str, Any]:
    return {
        "caption_vi": f"Cảnh {index}",
        "caption_en": f"Scene {index}",
        "objects_vi": [],
        "objects_en": [],
        "actions_vi": [],
        "actions_en": [],
        "scene_type": "unknown",
        "__provider": "vintern_local",
        "__model_id": "5CD-AI/Vintern-3B-R-beta",
        "__model_revision": "revision",
    }


class _V4Client:
    def request_many(self, requests):
        return [_v4_response(index) for index in range(len(requests))]


def test_bang_caption_van_du_cot() -> None:
    shots, keyframes = _shots_and_keyframes(2)
    ocr_rows = [
        {"keyframe_id": "keyframe_000:0", "text": "Xin chao", "status": "pass"},
    ]
    model_config = dict(load_configs(CONFIG_DIR)["models"]["phase01"]["shot_caption"])
    rows = _build_captions(
        video_id="L21_V001",
        shots=shots,
        keyframes=keyframes,
        ocr_rows=ocr_rows,
        stage_dir=SYSTEM1_ROOT,
        client=_V4Client(),
        model_config=model_config,
        max_concurrency=2,
    )
    validate_rows("shot_captions", rows)
    assert rows[0]["visible_text_summary_vi"] == "Xin chao"
    assert rows[0]["visible_text_summary_en"] == "Xin chao"
    assert rows[1]["visible_text_summary_vi"] == ""


def test_text_source_van_co_visible_text() -> None:
    shots, keyframes = _shots_and_keyframes(1)
    ocr_rows = [
        {"keyframe_id": "keyframe_000:0", "text": "Bien so 51F-123.45", "status": "pass"},
    ]
    model_config = dict(load_configs(CONFIG_DIR)["models"]["phase01"]["shot_caption"])
    rows = _build_captions(
        video_id="L21_V001",
        shots=shots,
        keyframes=keyframes,
        ocr_rows=ocr_rows,
        stage_dir=SYSTEM1_ROOT,
        client=_V4Client(),
        model_config=model_config,
        max_concurrency=2,
    )
    shot_captions = pd.DataFrame(rows)
    empty = pd.DataFrame(
        columns=["video_id", "keyframe_id", "text", "provider", "status", "language"]
    )
    empty_scenes = pd.DataFrame(columns=["video_id", "scene_id", "summary_vi", "summary_en"])
    text_sources = _build_structure_text_sources(empty, empty, shot_captions, empty_scenes)

    visible_text_rows = text_sources[text_sources["source_type"] == "visible_text_summary"]
    assert len(visible_text_rows) == 2  # vi + en
    assert set(visible_text_rows["raw_text"]) == {"Bien so 51F-123.45"}


def test_models_yaml_tro_v4() -> None:
    models = load_configs(CONFIG_DIR)["models"]["phase01"]
    shot_caption = models["shot_caption"]
    assert shot_caption["prompt_version"] == "shot_caption_v4"
    fallback_versions = [f["prompt_version"] for f in shot_caption.get("fallbacks", [])]
    assert "shot_caption_v2" in fallback_versions
