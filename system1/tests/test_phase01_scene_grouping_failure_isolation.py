"""One unjudgeable gap must not discard a video the earlier stages finished."""

from __future__ import annotations

import pytest

from system1.scenes.grouping import group_scenes

CONFIG = {
    "focus_gap_count": 2,
    "context_shots_each_side": 4,
    "stride": 2,
    "boundary_threshold": 0.67,
    "non_boundary_threshold": 0.33,
    "dense_boundary_count": 2,
    "dense_boundary_gap_window": 3,
    "strong_disagreement_min_votes": 2,
    "strong_disagreement_requires_both_labels": True,
    "max_consistency_review_rounds": 1,
    "min_failed_gaps_before_abort": 3,
    "max_failed_gap_ratio": 0.25,
}


def _shots(count: int, video_id: str = "L01_V001") -> list[dict]:
    return [
        {
            "shot_id": f"{video_id}_S{index:05d}",
            "video_id": video_id,
            "shot_index": index,
            "start_frame": index * 10,
            "end_frame": (index + 1) * 10,
            "start_sec": float(index),
            "end_sec": float(index + 1),
        }
        for index in range(count)
    ]


def _evidence(shots: list[dict]) -> list[dict]:
    return [{"shot_id": shot["shot_id"]} for shot in shots]


class _Judge:
    """Answers False everywhere except windows told to blow up."""

    def __init__(self, failing_windows: set[int]) -> None:
        self.failing_windows = failing_windows
        self.calls = 0

    def judge(self, *, request_kind, focus_gap_ids, context):
        index = self.calls
        self.calls += 1
        if index in self.failing_windows:
            raise ValueError("judge could not parse the reply")
        return {gap_id: False for gap_id in focus_gap_ids}


def _run(shot_count: int, failing_windows: set[int], **overrides):
    shots = _shots(shot_count)
    config = {**CONFIG, **overrides}
    return group_scenes(
        video_id="L01_V001",
        shots=shots,
        evidence=_evidence(shots),
        judge=_Judge(failing_windows),
        config=config,
    )


def test_single_failed_window_keeps_the_video():
    scenes, decisions = _run(11, {0})
    assert scenes, "video must survive one unjudgeable window"
    defaulted = [d for d in decisions if d.review_route == "default_on_error"]
    assert defaulted, "the failed gaps must be marked, not hidden"
    assert all(d.is_boundary is False for d in defaulted)


def test_healthy_run_marks_nothing_as_defaulted():
    _, decisions = _run(11, set())
    assert all(d.review_route != "default_on_error" for d in decisions)


def test_systemic_failure_still_aborts():
    """3 of 10 windows fail — 6 of 20 gaps, over ratio and floor, not total."""
    with pytest.raises(ValueError, match="too many gaps"):
        _run(21, {0, 1, 2})


def test_short_video_is_not_punished_by_ratio_alone():
    """5 shots, 1 failed window = 2 of 4 gaps — over the ratio, under the floor."""
    scenes, decisions = _run(5, {0})
    assert scenes
    assert any(d.review_route == "default_on_error" for d in decisions)
    assert any(d.review_route != "default_on_error" for d in decisions)


@pytest.mark.parametrize("shot_count", [3, 4, 5, 9])
def test_a_video_nothing_judged_never_ships_as_clean(shot_count):
    """The floor must not excuse a short video where every gap failed.

    A 4-gap video can never reach a floor of 3 failed replies, so total
    collapse used to return one scene marked "pass".
    """

    class AllFail:
        def judge(self, *, request_kind, focus_gap_ids, context):
            raise ValueError("judge failed outright")

    shots = _shots(shot_count)
    with pytest.raises(ValueError, match="failed on every gap"):
        group_scenes(
            video_id="L01_V001",
            shots=shots,
            evidence=_evidence(shots),
            judge=AllFail(),
            config=CONFIG,
        )


def test_floor_alone_does_not_abort():
    """Enough failures to clear the floor, but a low share of a long video."""
    scenes, _ = _run(41, {0, 1, 2})
    assert scenes


def test_wrong_gap_set_is_not_treated_as_a_flaky_reply():
    """Answering about gaps nobody asked for is a wiring bug — it must stop."""

    class WrongGapJudge:
        def judge(self, *, request_kind, focus_gap_ids, context):
            return {"a gap nobody asked about": True}

    shots = _shots(11)
    with pytest.raises(ValueError, match="gap set mismatch"):
        group_scenes(
            video_id="L01_V001",
            shots=shots,
            evidence=_evidence(shots),
            judge=WrongGapJudge(),
            config=CONFIG,
        )


def test_systemic_provider_error_is_never_defaulted():
    """A dead provider must not checkpoint a video nobody judged."""
    from system1.vlm import SystemicProviderError

    class DeadProvider:
        def judge(self, *, request_kind, focus_gap_ids, context):
            raise SystemicProviderError("CUDA OOM at batch size 1")

    shots = _shots(3)
    with pytest.raises(SystemicProviderError):
        group_scenes(
            video_id="L01_V001",
            shots=shots,
            evidence=_evidence(shots),
            judge=DeadProvider(),
            config=CONFIG,
        )


def test_focused_review_failures_reach_the_abort_gate():
    """The gate runs after focused_review, not before it."""

    class PrimaryOkFocusedDead:
        def judge(self, *, request_kind, focus_gap_ids, context):
            if request_kind == "primary":
                # Overlapping windows disagree, pushing gaps to focused_review.
                return {
                    gap_id: context[0]["shot_id"].endswith("00000")
                    for gap_id in focus_gap_ids
                }
            raise ValueError("focused review failed")

    shots = _shots(13)
    config = {
        **CONFIG,
        "focus_gap_count": 4,
        "stride": 2,
        "min_failed_gaps_before_abort": 1,
        "max_failed_gap_ratio": 0.1,
    }
    with pytest.raises(ValueError, match="too many gaps"):
        group_scenes(
            video_id="L01_V001",
            shots=shots,
            evidence=_evidence(shots),
            judge=PrimaryOkFocusedDead(),
            config=config,
        )


def test_floor_counts_replies_not_gaps():
    """One bad reply costs focus_gap_count gaps; the floor must not scale with it."""
    shots = _shots(9)
    config = {**CONFIG, "focus_gap_count": 2, "stride": 2}
    # 2 failed replies = 4 of 8 gaps (50%, over the ratio) but under the floor.
    scenes, _ = _run_with(shots, {0, 1}, config)
    assert scenes
    with pytest.raises(ValueError, match="failed replies"):
        _run_with(shots, {0, 1, 2}, config)


def _run_with(shots: list[dict], failing_windows: set[int], config: dict):
    return group_scenes(
        video_id="L01_V001",
        shots=shots,
        evidence=_evidence(shots),
        judge=_Judge(failing_windows),
        config=config,
    )


def test_abort_message_names_the_underlying_failure():
    shots = _shots(9)
    config = {**CONFIG, "focus_gap_count": 2, "stride": 2}
    with pytest.raises(ValueError, match="judge could not parse"):
        _run_with(shots, {0, 1, 2}, config)


def test_partition_still_covers_every_shot_after_a_failure():
    shots = _shots(11)
    scenes, _ = group_scenes(
        video_id="L01_V001",
        shots=shots,
        evidence=_evidence(shots),
        judge=_Judge({0}),
        config=CONFIG,
    )
    assert sum(scene["shot_count"] for scene in scenes) == len(shots)
