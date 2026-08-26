"""Local requests must be counted whichever local provider serves them.

The counter only recognised "qwen_local", so every run on Vintern reported zero
local requests — the 26/08 log shows qwen_request_count=0 next to 981 real ones.
A broken measurement is worse than none: it makes the next comparison silent.
"""

from __future__ import annotations

from types import SimpleNamespace

from system1.vlm.client import FallbackStructuredClient


def _client(provider: str):
    return SimpleNamespace(provider_name=provider)


def _fallback():
    return FallbackStructuredClient([_client("vintern_local"), _client("gemini")])


def test_vintern_requests_are_counted():
    fallback = _fallback()
    fallback._record_provider_requests(_client("vintern_local"), 5)
    assert fallback._counts["qwen_request_count"] == 5


def test_qwen_still_counted():
    fallback = _fallback()
    fallback._record_provider_requests(_client("qwen_local"), 3)
    assert fallback._counts["qwen_request_count"] == 3


def test_gemini_goes_to_its_own_bucket():
    fallback = _fallback()
    fallback._record_provider_requests(_client("gemini"), 4)
    assert fallback._counts["gemini_request_count"] == 4
    assert fallback._counts["qwen_request_count"] == 0


def test_unnamed_provider_is_not_counted():
    fallback = _fallback()
    fallback._record_provider_requests(_client(""), 7)
    fallback._record_provider_requests(SimpleNamespace(), 7)
    assert fallback._counts["qwen_request_count"] == 0
    assert fallback._counts["gemini_request_count"] == 0


def test_a_dead_provider_keeps_its_type_through_request():
    """Callers isolate one bad reply but must still abort on a dead GPU.

    request() used to rewrap every failure as a bare RuntimeError, so
    group_scenes' SystemicProviderError guard could never fire and a video
    nothing judged was checkpointed as clean.
    """
    from system1.vlm.client import BatchRequestError, SystemicProviderError

    class Dead:
        provider_name = "vintern_local"

        def request_many(self, requests):
            raise BatchRequestError(
                results=[None], errors={0: SystemicProviderError("CUDA OOM")}
            )

        def request(self, request):
            raise SystemicProviderError("CUDA OOM")

        def close(self):
            pass

    class DeadFallback:
        provider_name = "gemini"

        def request(self, request):
            raise RuntimeError("gemini down too")

    fallback = FallbackStructuredClient([Dead(), DeadFallback()])
    try:
        fallback.request(SimpleNamespace(request_kind="x", video_id="v"))
    except SystemicProviderError:
        pass
    except Exception as exc:  # noqa: BLE001 - the point of the test
        raise AssertionError(f"systemic failure arrived as {type(exc).__name__}") from exc
    else:
        raise AssertionError("a dead provider must not return normally")


def test_an_ordinary_failure_stays_an_ordinary_error():
    from system1.vlm.client import BatchRequestError, SystemicProviderError

    class Flaky:
        provider_name = "vintern_local"

        def request_many(self, requests):
            raise BatchRequestError(results=[None], errors={0: ValueError("bad json")})

        def request(self, request):
            raise ValueError("bad json")

    class DeadFallback:
        provider_name = "gemini"

        def request(self, request):
            raise RuntimeError("gemini down too")

    fallback = FallbackStructuredClient([Flaky(), DeadFallback()])
    try:
        fallback.request(SimpleNamespace(request_kind="x", video_id="v"))
    except SystemicProviderError as exc:
        raise AssertionError("a parse failure is not a dead provider") from exc
    except Exception:
        pass


def test_counts_accumulate():
    fallback = _fallback()
    fallback._record_provider_requests(_client("vintern_local"), 2)
    fallback._record_provider_requests(_client("vintern_local"), 3)
    assert fallback._counts["qwen_request_count"] == 5
