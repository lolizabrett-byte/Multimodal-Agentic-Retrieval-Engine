"""Freeing GPU memory after an OOM has to drop the traceback first.

The 26/08 run unloaded the model, ran gc.collect() and empty_cache(), and still
reported 8.646 GB allocated — more than the 6.923 GB the live model held. An
exception's __traceback__ keeps every stack frame alive, and those frames hold
the activations the failed forward pass had just built.
"""

from __future__ import annotations

import gc

from system1.vlm.client import _drop_traceback


class _Tracked:
    """Stands in for a GPU tensor: says so when it is actually collected."""

    freed: list[str] = []

    def __init__(self, name: str) -> None:
        self.name = name

    def __del__(self) -> None:
        _Tracked.freed.append(self.name)


def _raise_holding(name: str) -> None:
    _local_tensor = _Tracked(name)  # noqa: F841 - the frame is the point
    raise RuntimeError("CUDA out of memory")


def test_a_held_exception_keeps_frame_locals_alive():
    """Baseline: this is the behaviour that cost 1.7 GB."""
    _Tracked.freed = []
    try:
        _raise_holding("kept")
    except RuntimeError:
        gc.collect()
        assert "kept" not in _Tracked.freed
    gc.collect()
    assert "kept" in _Tracked.freed


def test_dropping_the_traceback_frees_them_immediately():
    _Tracked.freed = []
    try:
        _raise_holding("dropped")
    except RuntimeError as exc:
        _drop_traceback(exc)
        gc.collect()
        assert "dropped" in _Tracked.freed


def test_drop_traceback_returns_the_same_exception():
    error = RuntimeError("boom")
    assert _drop_traceback(error) is error


def test_drop_traceback_survives_an_exception_that_refuses():
    class Stubborn(RuntimeError):
        @property
        def __traceback__(self):  # type: ignore[override]
            return None

    # Must not raise: cleanup can never be the thing that breaks the run.
    _drop_traceback(Stubborn("boom"))


def test_the_cause_chain_is_dropped_too():
    """`raise ... from exc` re-attaches the original; its frames count as well."""
    _Tracked.freed = []
    try:
        try:
            _raise_holding("cause")
        except RuntimeError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as outer:
        _drop_traceback(outer)
        gc.collect()
        assert "cause" in _Tracked.freed
