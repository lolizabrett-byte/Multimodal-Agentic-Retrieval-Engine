"""Catch the NaN-driven token walk before it reaches the JSON parser.

When attention logits go NaN the model emits token 0 ('!'), then 1, 2, 3… as the
repeat penalty pushes argmax along. The reply parses as neither JSON nor prose,
and calling that a json_decode_error hides what actually happened.

Thresholds come from measurement, not taste — see the ratio test below.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from system1.vlm.client import _is_degenerate_output

SYSTEM1_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = SYSTEM1_ROOT / "tests" / "fixtures" / "scene_boundary_failure_samples.json"
BENCHMARK_RESULTS = SYSTEM1_ROOT / "research" / "vlm_prompting" / "results"

THRESHOLD = 0.6


def _degenerate_sample() -> str:
    samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return "!" * 1024 if "len_1024" not in samples else samples["len_1024"]["raw_text"]


def test_the_exclamation_walk_is_rejected():
    assert _is_degenerate_output(_degenerate_sample(), THRESHOLD) is True


def test_a_single_repeated_character_is_rejected():
    assert _is_degenerate_output("!" * 200, THRESHOLD) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "aaaa",  # 100% one character, but far too short to judge
        "a" * 63,
    ],
)
def test_short_text_is_never_called_degenerate(text):
    assert _is_degenerate_output(text, THRESHOLD) is False


def test_boundary_is_inclusive():
    at_threshold = "x" * 60 + "abcdefghij" * 4
    assert len(at_threshold) == 100
    assert _is_degenerate_output(at_threshold, THRESHOLD) is True


def test_just_under_the_boundary_passes():
    under = "x" * 59 + "abcdefghij" * 4 + "z"
    assert len(under) == 100
    assert _is_degenerate_output(under, THRESHOLD) is False


def test_fallback_reason_separates_it_from_a_parse_failure():
    from system1.vlm.client import DegenerateOutputError, _fallback_reason

    assert _fallback_reason(DegenerateOutputError("x")) == "degenerate_output"
    assert _fallback_reason(ValueError("x")) != "degenerate_output"


def test_degenerate_error_is_a_value_error():
    """The fallback path catches Exception, so Gemini still picks the shot up."""
    from system1.vlm.client import DegenerateOutputError

    assert issubclass(DegenerateOutputError, ValueError)


def _benchmark_captions() -> list[str]:
    """Every caption string the 355-image benchmark actually produced."""
    found: list[str] = []

    def walk(node):
        if isinstance(node, str):
            if len(node) > 60:
                found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in sorted(BENCHMARK_RESULTS.glob("checkpoint_*.json")):
        try:
            walk(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return found


def test_real_captions_are_never_rejected():
    """The guard must not fire on any caption the model actually produced.

    Measured over these same strings: real captions top out at 0.302 (the most
    common character being a space), while degenerate replies sit at 0.917+.
    """
    captions = _benchmark_captions()
    if len(captions) < 100:
        pytest.skip("benchmark results not available in this checkout")
    rejected = [
        text
        for text in captions
        if _is_degenerate_output(text, THRESHOLD)
        and Counter(text).most_common(1)[0][0] != "!"
    ]
    assert rejected == [], f"guard would drop {len(rejected)} real captions"
