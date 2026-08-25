from __future__ import annotations

import unicodedata

import pytest

from system1.metrics import compute_batch, compute_cer, compute_wer, normalize_vi

jiwer = pytest.importorskip("jiwer")


def test_normalize_collapses_nfc_and_nfd_to_same_string():
    composed = unicodedata.normalize("NFC", "điều kiện")
    decomposed = unicodedata.normalize("NFD", "điều kiện")

    assert composed != decomposed
    assert normalize_vi(composed) == normalize_vi(decomposed)


def test_cer_is_zero_for_nfc_vs_nfd_of_same_text():
    composed = unicodedata.normalize("NFC", "Trường Đại học Khoa học Tự nhiên")
    decomposed = unicodedata.normalize("NFD", "Trường Đại học Khoa học Tự nhiên")

    assert compute_cer(composed, decomposed) == 0.0
    assert compute_wer(composed, decomposed) == 0.0


def test_cer_is_zero_for_identical_text():
    assert compute_cer("xin chào", "xin chào") == 0.0


def test_cer_counts_single_character_substitution():
    assert compute_cer("abc", "abd", lowercase=False) == pytest.approx(1 / 3)


def test_missing_diacritics_are_counted_as_errors():
    assert compute_wer("xin chào", "xin chao") > 0.0


def test_normalize_collapses_whitespace():
    assert normalize_vi("  xin   chào \n") == "xin chào"


def test_lowercase_flag_is_respected():
    assert compute_cer("ABC", "abc") == 0.0
    assert compute_cer("ABC", "abc", lowercase=False) > 0.0


def test_empty_reference_and_hypothesis_do_not_divide_by_zero():
    assert compute_cer("", "") == 0.0
    assert compute_wer("", "") == 0.0


def test_empty_reference_with_text_hypothesis_returns_one():
    assert compute_cer("", "abc") == 1.0


def test_whitespace_only_is_treated_as_empty():
    assert compute_cer("   ", "") == 0.0


def test_batch_averages_and_keeps_per_pair_detail():
    result = compute_batch([("abc", "abc"), ("abc", "abd")])

    assert result["count"] == 2
    assert result["cer"] == pytest.approx((0.0 + 1 / 3) / 2)
    assert [item["index"] for item in result["details"]] == [0, 1]


def test_batch_groups_by_difficulty():
    result = compute_batch(
        [("abc", "abc"), ("abc", "abd"), ("abc", "xyz")],
        group_by=["easy", "hard", "hard"],
    )

    assert result["groups"]["easy"]["cer"] == 0.0
    assert result["groups"]["hard"]["count"] == 2
    assert result["groups"]["hard"]["cer"] > result["groups"]["easy"]["cer"]


def test_batch_rejects_mismatched_group_length():
    with pytest.raises(ValueError):
        compute_batch([("a", "a")], group_by=["easy", "hard"])


def test_batch_handles_empty_input():
    assert compute_batch([])["count"] == 0


def test_long_text_does_not_error():
    reference = "xin chào " * 1200
    assert compute_cer(reference, reference) == 0.0
