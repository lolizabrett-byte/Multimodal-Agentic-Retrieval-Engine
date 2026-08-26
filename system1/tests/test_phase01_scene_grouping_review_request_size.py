"""A dense video must not collapse its consistency review into one giant request.

Measured on L21_V019 (27/08): every gap triggered, the regions merged into one
spanning all 247 shots, and the single request carried 97,023 tokens — 30x a
normal one. It died on a 16 GB card and took the whole video with it.
"""

from __future__ import annotations

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
    "consistency_review_max_gaps": 2,
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


class _EveryGapIsABoundary:
    """Answers True everywhere, which is what makes every gap trigger review."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def judge(self, *, request_kind, focus_gap_ids, context):
        self.requests.append(
            {
                "kind": request_kind,
                "gaps": len(focus_gap_ids),
                "shots": len(context),
            }
        )
        return {gap_id: True for gap_id in focus_gap_ids}


def _run(shot_count: int, **overrides):
    shots = _shots(shot_count)
    judge = _EveryGapIsABoundary()
    group_scenes(
        video_id="L01_V001",
        shots=shots,
        evidence=_evidence(shots),
        judge=judge,
        config={**CONFIG, **overrides},
    )
    return judge


def test_review_requests_stay_within_the_configured_gap_cap():
    judge = _run(60)
    reviews = [r for r in judge.requests if r["kind"] == "consistency_review"]
    assert reviews, "a video where every gap is a boundary must trigger review"
    assert max(r["gaps"] for r in reviews) <= CONFIG["consistency_review_max_gaps"]


def test_review_requests_stay_near_the_primary_round_size():
    """The primary round already runs at this size on a 16 GB card."""
    judge = _run(60)
    primary = [r for r in judge.requests if r["kind"] == "primary"]
    reviews = [r for r in judge.requests if r["kind"] == "consistency_review"]
    assert max(r["shots"] for r in reviews) <= max(r["shots"] for r in primary)


def test_a_dense_video_never_sends_one_request_covering_everything():
    """The exact shape that OOMed: one request holding the whole video."""
    shot_count = 60
    judge = _run(shot_count)
    assert all(r["shots"] < shot_count for r in judge.requests)


def test_the_cap_splits_rather_than_drops_gaps():
    judge = _run(60)
    reviews = [r for r in judge.requests if r["kind"] == "consistency_review"]
    assert sum(r["gaps"] for r in reviews) == 59, "every gap must still be reviewed"
