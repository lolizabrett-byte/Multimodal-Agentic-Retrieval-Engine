"""Tham số sinh phải đi từ config xuống tới model, không dừng ở config.

Vintern-3B mặc định `repetition_penalty: 1.0` — tức tắt hẳn. Nghiên cứu trên 355
ảnh đặt 1,05 và vẫn ghi nhận 5/9 ca lỗi là model lặp cụm từ tới hết trần token.
Pipeline không đặt gì, nên không có phanh nào cả.
"""

from __future__ import annotations

import pytest

from system1.vlm.client import LocalVisionStructuredClient

BASE_CONFIG = {
    "provider": "vintern_local",
    "model_id": "5CD-AI/Vintern-3B-R-beta",
    "model_revision": "test",
    "total_attempts": 1,
    "inference_batch_size": 1,
    "max_new_tokens": 768,
}


def _generation_config(extra: dict | None = None) -> dict:
    client = LocalVisionStructuredClient(model_config={**BASE_CONFIG, **(extra or {})})
    return client.generation_config()


def test_repetition_penalty_reaches_the_model_when_configured():
    config = _generation_config({"repetition_penalty": 1.05})

    assert config["repetition_penalty"] == pytest.approx(1.05)


def test_repetition_penalty_is_absent_when_not_configured():
    """Không khai báo thì không truyền — giữ nguyên hành vi cũ."""
    config = _generation_config()

    assert "repetition_penalty" not in config


def test_changing_the_config_changes_what_the_model_receives():
    assert _generation_config({"repetition_penalty": 1.2})[
        "repetition_penalty"
    ] == pytest.approx(1.2)


def test_existing_generation_settings_are_unchanged():
    config = _generation_config()

    assert config["max_new_tokens"] == 768
    assert config["do_sample"] is False


def test_max_new_tokens_still_comes_from_config():
    config = _generation_config({"max_new_tokens": 320})

    assert config["max_new_tokens"] == 320


@pytest.mark.parametrize("bad", ["nhieu", -1.0, 0])
def test_invalid_repetition_penalty_is_rejected_loudly(bad):
    """Giá trị vô nghĩa phải báo lỗi, không được âm thầm bỏ qua."""
    with pytest.raises(ValueError):
        _generation_config({"repetition_penalty": bad})


def test_no_repeat_ngram_size_reaches_the_model_when_configured():
    """Lần chạy L23_V011 cho thấy penalty mềm 1,05 vẫn để model lặp tới hết trần."""
    config = _generation_config({"no_repeat_ngram_size": 12})

    assert config["no_repeat_ngram_size"] == 12


def test_no_repeat_ngram_size_is_absent_when_not_configured():
    config = _generation_config()

    assert "no_repeat_ngram_size" not in config


@pytest.mark.parametrize("bad", ["muoi hai", -1, 0])
def test_invalid_no_repeat_ngram_size_is_rejected_loudly(bad):
    with pytest.raises(ValueError):
        _generation_config({"no_repeat_ngram_size": bad})


def test_shipped_config_has_both_repetition_brakes_on_caption_model():
    """Cấu hình thật phải bật cả hai phanh, không chỉ một."""
    import yaml
    from pathlib import Path

    models = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "models.yaml").read_text(
            encoding="utf-8"
        )
    )
    caption = models["phase01"]["shot_caption"]

    assert caption["repetition_penalty"] >= 1.1
    assert caption["no_repeat_ngram_size"] >= 2
