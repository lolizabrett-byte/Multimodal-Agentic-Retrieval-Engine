"""A malformed reply must not be mistaken for a broken machine.

`_retryable_video_error` matched the substring "decode" for ffmpeg, and
"JSONDecodeError" contains it. One summary in thirteen came back with the wrong
closing bracket on 27/08, was classed as infrastructure, re-raised past the
per-scene degradation, and killed all three videos after two hours of work.
"""

from __future__ import annotations

import json

import pytest
from jsonschema.exceptions import ValidationError

from system1.phase01.production import _retryable_video_error
from system1.vlm import BatchRequestError, SystemicProviderError

DECODE_ERROR = json.JSONDecodeError("Expecting ',' delimiter", "{}", 734)


@pytest.mark.parametrize(
    "exc",
    [
        DECODE_ERROR,
        BatchRequestError(results={}, errors={0: DECODE_ERROR}),
        BatchRequestError(results={}, errors={0: ValidationError("schema")}),
        ValueError("summary_vi is empty"),
    ],
    ids=["bare", "wrapped", "schema", "content"],
)
def test_a_bad_reply_degrades_the_scene_not_the_video(exc) -> None:
    assert _retryable_video_error(exc) is False


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA out of memory"),
        RuntimeError("connection timeout"),
        RuntimeError("i/o error reading file"),
        MemoryError("out of host memory"),
        SystemicProviderError("provider gone"),
        BatchRequestError(results={}, errors={0: SystemicProviderError("CUDA OOM")}),
    ],
    ids=["oom", "timeout", "io", "memoryerror", "systemic", "systemic-wrapped"],
)
def test_a_broken_machine_still_stops_the_video(exc) -> None:
    assert _retryable_video_error(exc) is True


def test_ffmpeg_decode_failures_keep_their_meaning() -> None:
    """The substring was there for video decoding, and that must still work."""
    assert _retryable_video_error(RuntimeError("ffmpeg failed to decode frame"))


def test_a_systemic_fault_wins_over_a_bad_reply_in_the_same_batch() -> None:
    """Mixed causes resolve to the safer answer: stop, do not paper over it."""
    exc = BatchRequestError(
        results={},
        errors={0: DECODE_ERROR, 1: SystemicProviderError("CUDA OOM")},
    )
    assert _retryable_video_error(exc) is True
