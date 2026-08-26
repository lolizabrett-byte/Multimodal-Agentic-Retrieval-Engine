"""Record what each judge request looked like, before it is sent.

The 26/08 run OOMed on judge request 124 of a video; the 123 before it peaked at
7.3-7.5 GB and that one took 14.2. Nothing in the log said how the requests
differed, so the cause could not be traced afterwards.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from system1.scenes.gemini_judge import (
    StructuredSceneBoundaryJudge,
    _emit_request_shape,
    _image_shape,
)


class _Client:
    provider_name = "vintern_local"

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.lifecycle_callback = self.events.append

    def request(self, request):
        return {
            "boundaries": [
                {"gap_index": index, "is_scene_boundary": False}
                for index in range(len(request.identity["focus_gap_ids"]))
            ]
        }


def _context(tmp_path: Path, count: int) -> list[dict]:
    context = []
    for index in range(count):
        path = tmp_path / f"shot_{index}.jpg"
        Image.new("RGB", (320, 180), (index * 10 % 255, 40, 40)).save(path)
        context.append(
            {
                "shot_id": f"v_SH{index:05d}",
                "start_sec": float(index),
                "end_sec": float(index + 1),
                "representative_path": path,
                "caption_vi": "Một cảnh quay",
                "caption_en": "A shot",
                "transcript": "lời thoại",
            }
        )
    return context


def _judge(tmp_path: Path, client: _Client) -> StructuredSceneBoundaryJudge:
    return StructuredSceneBoundaryJudge(
        client,
        video_id="v",
        prompt_dir=Path(__file__).resolve().parents[1] / "prompts",
        diagnostics_dir=tmp_path / "diag",
        model_config={
            "prompt_version": "scene_boundary_primary_v4",
            "focused_prompt_version": "scene_boundary_focused_v4",
            "consistency_prompt_version": "scene_boundary_consistency_v4",
            "response_schema_version": "scene_boundary_response_v1",
        },
    )


def test_every_request_reports_its_shape(tmp_path):
    client = _Client()
    judge = _judge(tmp_path, client)
    judge.judge(
        request_kind="primary",
        focus_gap_ids=("v_SH00000", "v_SH00001"),
        context=_context(tmp_path, 6),
    )
    shapes = [e for e in client.events if e.get("event") == "scene_judge_request"]
    assert len(shapes) == 1
    shape = shapes[0]
    assert shape["request_kind"] == "primary"
    assert shape["context_shots"] == 6
    assert shape["focus_gaps"] == 2
    assert shape["prompt_chars"] > 0
    assert shape["sheet_width"] == 1280


def test_a_wider_context_reports_a_longer_prompt(tmp_path):
    """The number the OOM investigation needed and did not have."""
    client = _Client()
    judge = _judge(tmp_path, client)
    for count in (3, 11):
        judge.judge(
            request_kind="primary",
            focus_gap_ids=("v_SH00000",),
            context=_context(tmp_path, count),
        )
    small, large = [
        e for e in client.events if e.get("event") == "scene_judge_request"
    ]
    assert large["prompt_chars"] > small["prompt_chars"]
    assert large["sheet_height"] > small["sheet_height"]


def test_the_index_increments_so_a_failing_request_is_identifiable(tmp_path):
    client = _Client()
    judge = _judge(tmp_path, client)
    for _ in range(3):
        judge.judge(
            request_kind="primary",
            focus_gap_ids=("v_SH00000",),
            context=_context(tmp_path, 3),
        )
    indices = [
        e["request_index"]
        for e in client.events
        if e.get("event") == "scene_judge_request"
    ]
    assert indices == [0, 1, 2]


def test_shape_is_emitted_before_the_request_runs(tmp_path):
    """A request that dies mid-flight must still have left its shape behind."""

    class Exploding(_Client):
        def request(self, request):
            raise RuntimeError("CUDA out of memory")

    client = Exploding()
    judge = _judge(tmp_path, client)
    try:
        judge.judge(
            request_kind="primary",
            focus_gap_ids=("v_SH00000",),
            context=_context(tmp_path, 4),
        )
    except RuntimeError:
        pass
    assert [e for e in client.events if e.get("event") == "scene_judge_request"]


def test_a_wrapped_client_still_reaches_the_log():
    """Production hands the judge a fallback wrapping a metadata wrapping local."""
    inner = _Client()
    wrapped = SimpleNamespace(clients=[SimpleNamespace(client=inner)])
    _emit_request_shape(wrapped, {"event": "scene_judge_request"})
    assert inner.events


def test_diagnostics_never_break_a_request():
    _emit_request_shape(SimpleNamespace(), {"event": "x"})  # no callback anywhere
    assert _image_shape(Path("does-not-exist.jpg")) == {}
