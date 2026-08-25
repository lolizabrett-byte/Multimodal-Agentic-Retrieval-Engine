"""Chọn video đại diện để đo, không phải video ngắn nhất.

Lượt đo 25/08 chạy trên video 0,5 phút — nhóm đó chiếm 0,4% tổng nội dung, trong
khi 61,5% số video dài 5–10 phút. Chi phí nạp model (77s) đè lên video 30 giây
làm méo mọi đơn giá suy ra từ nó.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "02_do_toc_do_mot_video.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("do_toc_do", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def measure():
    return _load_module()


def test_picks_the_video_inside_the_target_range(measure):
    durations = {"short": 120.0, "good": 420.0, "long": 1500.0}

    chosen, seconds, warning = measure.pick_representative_video(
        durations, target_min_sec=300, target_max_sec=600
    )

    assert chosen == "good"
    assert seconds == pytest.approx(420.0)
    assert warning is None


def test_picks_the_closest_video_when_none_are_in_range(measure):
    """Giữa khoảng là 450s, nên 700s gần hơn 30s — và phải kèm cảnh báo."""
    durations = {"tiny": 30.0, "just_over": 700.0}

    chosen, _seconds, warning = measure.pick_representative_video(
        durations, target_min_sec=300, target_max_sec=600
    )

    assert chosen == "just_over"
    assert warning is not None


def test_prefers_the_middle_of_the_range_when_several_qualify(measure):
    durations = {"low": 310.0, "middle": 450.0, "high": 590.0}

    chosen, _seconds, warning = measure.pick_representative_video(
        durations, target_min_sec=300, target_max_sec=600
    )

    assert chosen == "middle"
    assert warning is None


def test_empty_durations_is_an_explicit_error(measure):
    with pytest.raises(ValueError):
        measure.pick_representative_video({}, target_min_sec=300, target_max_sec=600)


def test_valid_caption_ratio_counts_only_passing_rows(measure):
    counts = {"pass": 7, "failed": 5}

    assert measure.caption_valid_ratio(counts) == pytest.approx(7 / 12)


def test_caption_ratio_is_none_when_nothing_was_captioned(measure):
    assert measure.caption_valid_ratio({}) is None


def test_unit_cost_subtracts_model_load_time(measure):
    cost = measure.unit_cost_excluding_load(
        stage_seconds=75.9, load_seconds=31.8, units=35
    )

    assert cost == pytest.approx((75.9 - 31.8) / 35)


def test_unit_cost_falls_back_to_raw_when_load_is_unknown(measure):
    cost = measure.unit_cost_excluding_load(
        stage_seconds=75.9, load_seconds=None, units=35
    )

    assert cost == pytest.approx(75.9 / 35)


def test_unit_cost_is_none_without_units(measure):
    assert (
        measure.unit_cost_excluding_load(stage_seconds=10.0, load_seconds=1.0, units=0)
        is None
    )
