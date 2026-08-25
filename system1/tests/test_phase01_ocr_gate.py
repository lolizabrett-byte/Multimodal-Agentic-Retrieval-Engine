from __future__ import annotations

from pathlib import Path

from PIL import Image

from system1.phase01 import production
from system1.phase01.validation import validate_rows

MODEL_CONFIG = {
    "provider": "vintern_local",
    "model_id": "5CD-AI/Vintern-1B-v3_5",
    "model_revision": "revision",
    "prompt_version": "keyframe_ocr_v1",
    "response_schema_version": "keyframe_ocr_response_v1",
}
OCR_CONFIG = {
    "run_on_keyframe_roles": ["middle"],
    "text_presence_filter": {
        "enabled": True,
        "policy": "opencv_conservative_v1",
        "max_long_side": 960,
        "canny_low": 50,
        "canny_high": 150,
        "max_no_text_edge_density": 0.0015,
        "max_no_text_gray_std": 12,
    },
}


class RecordingClient:
    def __init__(self) -> None:
        self.requests = []

    def request_many(self, requests):
        self.requests.extend(requests)
        return [
            {
                "full_text": "UY BAN NHAN DAN",
                "ocr_blocks": [],
                "language": "vi",
                "confidence": 0.9,
                "__provider": "vintern_local",
                "__model_id": MODEL_CONFIG["model_id"],
                "__model_revision": MODEL_CONFIG["model_revision"],
            }
            for _request in requests
        ]


def _keyframe(stage_dir: Path, *, color: str = "black") -> dict:
    keyframes = stage_dir / "keyframes"
    keyframes.mkdir(parents=True)
    image_path = keyframes / "frame.jpg"
    Image.new("RGB", (320, 180), color=color).save(image_path)
    return {
        "keyframe_id": "L21_V001:0",
        "video_id": "L21_V001",
        "shot_id": "L21_V001_SH00000",
        "frame_id": 0,
        "keyframe_role": "middle",
        "keyframe_ref": "media://keyframes/L21_V001/frame.jpg",
    }


def test_high_confidence_no_text_skips_vintern_and_emits_empty_ocr_v2(
    tmp_path: Path,
) -> None:
    client = RecordingClient()
    diagnostics = {}

    rows = production._build_ocr(
        video_id="L21_V001",
        keyframes=[_keyframe(tmp_path)],
        stage_dir=tmp_path,
        client=client,
        model_config=MODEL_CONFIG,
        ocr_config=OCR_CONFIG,
        diagnostics=diagnostics,
    )

    assert client.requests == []
    assert rows[0]["status"] == "empty"
    assert rows[0]["provider"] == "opencv_text_gate"
    assert rows[0]["model_name"] == "opencv_mser_canny"
    assert diagnostics == {
        "gate_checked": 1,
        "gate_no_text": 1,
        "gate_failures": 0,
        "vintern_processed": 0,
        "dedup_reused": 0,
    }
    validate_rows("ocr", rows)


def test_uncertain_gate_runs_vintern(tmp_path: Path, monkeypatch) -> None:
    client = RecordingClient()
    monkeypatch.setattr(production, "_text_presence_gate", lambda *_args: "uncertain")

    rows = production._build_ocr(
        video_id="L21_V001",
        keyframes=[_keyframe(tmp_path, color="white")],
        stage_dir=tmp_path,
        client=client,
        model_config=MODEL_CONFIG,
        ocr_config=OCR_CONFIG,
    )

    assert len(client.requests) == 1
    assert rows[0]["status"] == "pass"
    assert rows[0]["provider"] == "vintern_local"
    validate_rows("ocr", rows)


def test_gate_failure_runs_vintern_and_counts_failure(
    tmp_path: Path, monkeypatch
) -> None:
    client = RecordingClient()
    diagnostics = {}

    def fail_gate(*_args):
        raise RuntimeError("detector error")

    monkeypatch.setattr(production, "_text_presence_gate", fail_gate)
    rows = production._build_ocr(
        video_id="L21_V001",
        keyframes=[_keyframe(tmp_path, color="white")],
        stage_dir=tmp_path,
        client=client,
        model_config=MODEL_CONFIG,
        ocr_config=OCR_CONFIG,
        diagnostics=diagnostics,
    )

    assert len(client.requests) == 1
    assert diagnostics["gate_failures"] == 1
    assert diagnostics["vintern_processed"] == 1
    validate_rows("ocr", rows)


DEDUP_OCR_CONFIG = {
    "run_on_keyframe_roles": ["early", "middle", "late"],
    "text_presence_filter": {"enabled": False},
    "dedup": {"enabled": True, "hamming_threshold": 4},
}


def _shot_keyframes(stage_dir: Path, colors: list[str]) -> list[dict]:
    """One shot, three role keyframes painted with the given colors."""
    keyframes_dir = stage_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    roles = ["early", "middle", "late"]
    rows = []
    for index, (role, color) in enumerate(zip(roles, colors, strict=True)):
        name = f"frame_{index}.jpg"
        image = Image.new("RGB", (320, 180), color="black")
        for x in range(0, 320, 8):
            for y in range(0, 180, 8):
                if (x // 8 + y // 8) % 2:
                    image.putpixel((x, y), (255, 255, 255) if color == "a" else (0, 255, 0))
        if color == "b":
            for x in range(160):
                for y in range(90):
                    image.putpixel((x, y), (255, 0, 0))
        image.save(keyframes_dir / name)
        rows.append(
            {
                "keyframe_id": f"L21_V001:{index}",
                "video_id": "L21_V001",
                "shot_id": "L21_V001_SH00000",
                "frame_id": index,
                "keyframe_role": role,
                "keyframe_ref": f"media://keyframes/L21_V001/{name}",
            }
        )
    return rows


def test_dedup_disabled_keeps_one_request_per_keyframe(tmp_path: Path) -> None:
    client = RecordingClient()
    config = {**DEDUP_OCR_CONFIG, "dedup": {"enabled": False}}

    rows = production._build_ocr(
        video_id="L21_V001",
        keyframes=_shot_keyframes(tmp_path, ["a", "a", "a"]),
        stage_dir=tmp_path,
        client=client,
        model_config=MODEL_CONFIG,
        ocr_config=config,
        diagnostics={},
    )

    assert len(client.requests) == 3
    assert len(rows) == 3
    validate_rows("ocr", rows)


def test_dedup_collapses_identical_frames_but_still_emits_every_row(
    tmp_path: Path,
) -> None:
    client = RecordingClient()
    diagnostics = {}

    rows = production._build_ocr(
        video_id="L21_V001",
        keyframes=_shot_keyframes(tmp_path, ["a", "a", "a"]),
        stage_dir=tmp_path,
        client=client,
        model_config=MODEL_CONFIG,
        ocr_config=DEDUP_OCR_CONFIG,
        diagnostics=diagnostics,
    )

    assert len(client.requests) == 1
    assert len(rows) == 3
    assert {row["text"] for row in rows} == {"UY BAN NHAN DAN"}
    assert diagnostics["dedup_reused"] == 2
    validate_rows("ocr", rows)


def test_dedup_keeps_visually_different_frames_apart(tmp_path: Path) -> None:
    client = RecordingClient()

    rows = production._build_ocr(
        video_id="L21_V001",
        keyframes=_shot_keyframes(tmp_path, ["a", "b", "a"]),
        stage_dir=tmp_path,
        client=client,
        model_config=MODEL_CONFIG,
        ocr_config=DEDUP_OCR_CONFIG,
        diagnostics={},
    )

    assert len(client.requests) >= 2
    assert len(rows) == 3
    validate_rows("ocr", rows)


def test_dedup_never_groups_across_shots(tmp_path: Path) -> None:
    client = RecordingClient()
    keyframes = _shot_keyframes(tmp_path, ["a", "a", "a"])
    for index, keyframe in enumerate(keyframes):
        keyframe["shot_id"] = f"L21_V001_SH{index:05d}"

    rows = production._build_ocr(
        video_id="L21_V001",
        keyframes=keyframes,
        stage_dir=tmp_path,
        client=client,
        model_config=MODEL_CONFIG,
        ocr_config=DEDUP_OCR_CONFIG,
        diagnostics={},
    )

    assert len(client.requests) == 3
    assert len(rows) == 3
