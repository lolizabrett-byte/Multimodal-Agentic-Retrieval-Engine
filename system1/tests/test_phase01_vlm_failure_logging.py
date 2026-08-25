"""Khi model trả JSON hỏng, phải biết nó đã trả gì.

Hiện tại `raw_text` bị vứt cùng exception, và nhãn `invalid_structured_response`
gộp cả lỗi cú pháp JSON lẫn lỗi sai schema — hai thứ cần hai cách sửa khác nhau.
Đo trên Kaggle 25/08: caption fail 5/12 shot mà không ai biết model trả gì.
"""

from __future__ import annotations

import json

import pytest
from jsonschema.exceptions import ValidationError

from system1.gemini import GeminiRequest
from system1.vlm.client import (
    LocalVisionStructuredClient,
    SystemicProviderError,
    _fallback_reason,
)

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def _request() -> GeminiRequest:
    return GeminiRequest(
        request_kind="test",
        video_id="L21_V001",
        prompt="return json",
        prompt_version="test_prompt",
        response_schema_version="test_schema",
        response_schema=SCHEMA,
    )


def _client(lifecycle: list[dict[str, object]]) -> LocalVisionStructuredClient:
    return LocalVisionStructuredClient(
        model_config={
            "provider": "vintern_local",
            "model_id": "5CD-AI/Vintern-3B-R-beta",
            "model_revision": "test",
            "total_attempts": 1,
            "inference_batch_size": 2,
        },
        lifecycle_callback=lambda payload: lifecycle.append(dict(payload)),
    )


def _batch_complete(lifecycle: list[dict[str, object]]) -> dict[str, object]:
    return next(
        event for event in lifecycle if event.get("status") == "batch_complete"
    )


def _run_and_capture(monkeypatch, raw_texts: list[str]) -> dict[str, object]:
    """Chạy một lô hỏng, trả về sự kiện batch_complete."""
    lifecycle: list[dict[str, object]] = []
    client = _client(lifecycle)
    monkeypatch.setattr(client, "_call_models", lambda requests: list(raw_texts))

    with pytest.raises(Exception):
        client.request_many([_request() for _ in raw_texts])

    return _batch_complete(lifecycle)


def test_json_decode_error_gets_its_own_reason():
    reason = _fallback_reason(json.JSONDecodeError("bad", "doc", 0))

    assert reason == "json_decode_error"


def test_schema_validation_error_gets_its_own_reason():
    reason = _fallback_reason(ValidationError("missing field"))

    assert reason == "schema_validation_error"


def test_systemic_error_reason_is_unchanged():
    reason = _fallback_reason(SystemicProviderError("driver gone"))

    assert reason == "systemic_local_runtime"


def test_other_value_errors_keep_the_legacy_reason():
    """Nhãn cũ phải còn để mã đang đọc nó không vỡ."""
    reason = _fallback_reason(ValueError("something else"))

    assert reason == "invalid_structured_response"


def test_broken_json_is_recorded_with_what_the_model_returned(monkeypatch):
    event = _run_and_capture(monkeypatch, ["khong phai json"])

    samples = event["failure_samples"]
    assert len(samples) == 1
    assert samples[0]["reason"] == "json_decode_error"
    assert "khong phai json" in samples[0]["raw_text"]


def test_missing_field_is_recorded_as_schema_failure(monkeypatch):
    event = _run_and_capture(monkeypatch, ['{"wrong_field": "x"}'])

    samples = event["failure_samples"]
    assert samples[0]["reason"] == "schema_validation_error"
    assert "wrong_field" in samples[0]["raw_text"]


def test_extra_field_is_recorded_as_schema_failure(monkeypatch):
    event = _run_and_capture(
        monkeypatch, ['{"value": "ok", "surprise": "extra"}']
    )

    samples = event["failure_samples"]
    assert samples[0]["reason"] == "schema_validation_error"
    assert "surprise" in samples[0]["raw_text"]


def test_long_raw_text_is_truncated_but_true_length_is_kept(monkeypatch):
    long_text = "x" * 10_000
    event = _run_and_capture(monkeypatch, [long_text])

    sample = event["failure_samples"][0]
    assert len(sample["raw_text"]) <= 2_000
    assert sample["raw_text_length"] == 10_000


def test_only_a_few_samples_are_kept_when_many_requests_fail(monkeypatch):
    event = _run_and_capture(monkeypatch, ["not json"] * 10)

    assert event["failed_request_count"] == 10
    assert len(event["failure_samples"]) <= 3


def test_successful_batch_reports_no_failure_samples(monkeypatch):
    lifecycle: list[dict[str, object]] = []
    client = _client(lifecycle)
    monkeypatch.setattr(
        client, "_call_models", lambda requests: ['{"value": "ok"}'] * len(requests)
    )

    client.request_many([_request()])

    event = _batch_complete(lifecycle)
    assert event["failed_request_count"] == 0
    assert event["failure_samples"] == []
