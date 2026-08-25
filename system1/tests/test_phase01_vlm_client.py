from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from system1.artifacts.store import ArtifactStore
from system1.gemini import GeminiRequest
from system1.vlm.client import (
    FallbackStructuredClient,
    LocalVisionStructuredClient,
    SystemicProviderError,
)


class _FakeTensor:
    dtype = "torch.float32"

    def __init__(self, tiles: int = 1) -> None:
        self.shape = (tiles, 3, 448, 448)

    def to(self, *_args, **_kwargs):
        return self


def _install_fake_torch(monkeypatch):
    module = ModuleType("torch")
    module.float16 = "torch.float16"
    module.bfloat16 = "torch.bfloat16"
    module.float32 = "torch.float32"
    module.cat = lambda tensors, dim=0: _FakeTensor(
        sum(tensor.shape[0] for tensor in tensors)
    )

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    module.no_grad = NoGrad
    module.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_request_error_falls_back_without_opening_circuit() -> None:
    events: list[str] = []

    class FailingClient:
        def request(self, _request):
            events.append("fail_request")
            raise RuntimeError("provider failed")

        def close(self) -> None:
            events.append("fail_close")

    class PassingClient:
        def request(self, _request):
            events.append("pass_request")
            return {"value": "ok"}

    request = GeminiRequest(
        request_kind="test",
        video_id="L21_V001",
        prompt="return json",
        prompt_version="test_prompt",
        response_schema_version="test_schema",
        response_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    client = FallbackStructuredClient([FailingClient(), PassingClient()])
    response = client.request(request)
    second = client.request(request)

    assert response == {"value": "ok"}
    assert second == {"value": "ok"}
    assert client.circuit_open is False
    assert events == [
        "fail_request",
        "pass_request",
        "fail_request",
        "pass_request",
    ]


def test_systemic_failure_closes_primary_and_circuits_chunk() -> None:
    resident: list[str] = []
    telemetry = []
    primary_calls = 0
    fallback_calls = 0

    class FailingPrimary:
        def request(self, _request):
            nonlocal primary_calls
            primary_calls += 1
            assert not resident
            resident.append("qwen")
            raise SystemicProviderError("qwen failed")

        def close(self) -> None:
            if "qwen" in resident:
                resident.remove("qwen")

    class PassingFallback:
        def request(self, _request):
            nonlocal fallback_calls
            fallback_calls += 1
            if not resident:
                resident.append("vintern")
            assert resident == ["vintern"]
            return {"value": "ok"}

        def close(self) -> None:
            if "vintern" in resident:
                resident.remove("vintern")

    request = GeminiRequest(
        request_kind="test",
        video_id="L21_V001",
        prompt="return json",
        prompt_version="test_prompt",
        response_schema_version="test_schema",
        response_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    primary = FailingPrimary()
    primary.provider_name = "qwen_local"
    fallback = PassingFallback()
    fallback.provider_name = "gemini"
    client = FallbackStructuredClient(
        [primary, fallback],
        telemetry_callback=lambda payload: telemetry.append(dict(payload)),
    )

    assert client.request(request) == {"value": "ok"}
    assert client.request(request) == {"value": "ok"}
    assert primary_calls == 1
    assert fallback_calls == 2
    assert client.circuit_open is True
    assert resident == ["vintern"]
    assert any(
        event["status"] == "circuit_breaker"
        and event["circuit_breaker_state"] == "open"
        for event in telemetry
    )
    fallback_event = [
        event for event in telemetry if event["status"] == "fallback"
    ][-1]
    assert fallback_event["qwen_request_count"] == 1
    assert fallback_event["gemini_request_count"] == 2
    assert fallback_event["fallback_request_count"] == 2

    client.close()
    assert resident == []


def test_local_vlm_client_close_releases_loaded_model_handles() -> None:
    client = LocalVisionStructuredClient(
        model_config={
            "provider": "qwen_local",
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": "test",
        }
    )
    client._loaded = (object(), object())

    client.close()

    assert client._loaded is None


def test_fallback_client_reports_all_provider_failures() -> None:
    class FailingClient:
        def __init__(self, name: str) -> None:
            self.name = name

        def request(self, _request):
            raise RuntimeError(self.name)

    request = GeminiRequest(
        request_kind="test",
        video_id="L21_V001",
        prompt="return json",
        prompt_version="test_prompt",
        response_schema_version="test_schema",
        response_schema={},
        image_paths=(Path("missing.jpg"),),
    )

    with pytest.raises(RuntimeError, match="first.*second"):
        FallbackStructuredClient(
            [FailingClient("first"), FailingClient("second")]
        ).request(request)


def test_local_vlm_adaptive_oom_reduces_to_one(monkeypatch) -> None:
    lifecycle: list[dict[str, object]] = []
    releases: list[str] = []
    calls = 0
    client = LocalVisionStructuredClient(
        model_config={
            "provider": "qwen_local",
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": "test",
            "total_attempts": 2,
            "inference_batch_size": 2,
        },
        lifecycle_callback=lambda payload: lifecycle.append(dict(payload)),
    )

    def call_models(requests) -> list[str]:
        nonlocal calls
        calls += 1
        if len(requests) > 1:
            raise RuntimeError("CUDA out of memory")
        return ['{"value": "ok"}']

    monkeypatch.setattr(client, "_call_models", call_models)
    monkeypatch.setattr(
        "system1.vlm.client._release_torch_memory",
        lambda: releases.append("released"),
    )
    request = GeminiRequest(
        request_kind="test",
        video_id="L21_V001",
        prompt="return json",
        prompt_version="test_prompt",
        response_schema_version="test_schema",
        response_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    response = client.request_many([request, request])

    assert [item["value"] for item in response] == ["ok", "ok"]
    assert calls == 3
    assert releases == ["released"]
    assert [event["status"] for event in lifecycle] == [
        "batch_start",
        "oom_reduction",
        "batch_complete",
    ]
    assert lifecycle[1]["effective_batch_size"] == 1


def test_local_vlm_rejects_non_positive_inference_batching() -> None:
    with pytest.raises(ValueError, match="inference_batch_size must be positive"):
        LocalVisionStructuredClient(
            model_config={
                "provider": "qwen_local",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "inference_batch_size": 0,
            }
        )


def test_two_qwen_requests_use_one_model_generate(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeTensor:
        shape = (2, 3)

        def to(self, _device):
            return self

        def __getitem__(self, _item):
            return self

    class FakeProcessor:
        def apply_chat_template(self, _conversation, **_kwargs):
            return "prompt"

        def __call__(self, *, text, **_kwargs):
            assert len(text) == 2
            return {"input_ids": FakeTensor()}

        def batch_decode(self, _generated, **_kwargs):
            return ['{"value": "first"}', '{"value": "second"}']

    class FakeModel:
        generate_calls = 0

        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])

        def generate(self, **_kwargs):
            self.generate_calls += 1
            return FakeTensor()

    model = FakeModel()
    client = LocalVisionStructuredClient(
        model_config={
            "provider": "qwen_local",
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": "test",
            "inference_batch_size": 2,
        }
    )
    monkeypatch.setattr(client, "_load_qwen", lambda: (FakeProcessor(), model))
    qwen_utils = ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = (  # type: ignore[attr-defined]
        lambda conversations: ([object() for _ in conversations], None)
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    images = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for image in images:
        image.write_bytes(image.name.encode())
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    requests = [
        GeminiRequest(
            request_kind="shot_caption",
            video_id="L21_V001",
            prompt=f"prompt {index}",
            prompt_version="test_prompt",
            response_schema_version="test_schema",
            response_schema=schema,
            image_paths=(image,),
        )
        for index, image in enumerate(images)
    ]

    responses = client.request_many(requests)

    assert [response["value"] for response in responses] == ["first", "second"]
    assert model.generate_calls == 1


def test_request_many_preserves_order_and_only_batches_cache_misses(
    tmp_path: Path, monkeypatch
) -> None:
    cache = ArtifactStore(tmp_path / "cache")
    calls: list[list[str]] = []
    client = LocalVisionStructuredClient(
        model_config={
            "provider": "qwen_local",
            "model_id": "model",
            "model_revision": "revision",
            "inference_batch_size": 2,
        },
        cache=cache,
    )

    def call_models(requests):
        calls.append([request.prompt for request in requests])
        return [f'{{"value": "{request.prompt}"}}' for request in requests]

    monkeypatch.setattr(client, "_call_models", call_models)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def make_request(prompt: str) -> GeminiRequest:
        return GeminiRequest(
            request_kind="shot_caption",
            video_id="L21_V001",
            prompt=prompt,
            prompt_version="prompt",
            response_schema_version="schema",
            response_schema=schema,
        )

    cached, missing = make_request("cached"), make_request("missing")
    assert client.request(cached)["value"] == "cached"
    calls.clear()

    responses = client.request_many([missing, cached])

    assert [response["value"] for response in responses] == ["missing", "cached"]
    assert calls == [["missing"]]


def test_qwen_loader_passes_explicit_nf4_quantization(monkeypatch) -> None:
    captured = {}
    _install_fake_torch(monkeypatch)

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class FakeModel:
        def eval(self):
            return None

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            captured.update(kwargs)
            return FakeModel()

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = FakeProcessorFactory
    transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    transformers.Qwen2_5_VLForConditionalGeneration = FakeModelFactory
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    client = LocalVisionStructuredClient(
        model_config={
            "provider": "qwen_local",
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": "revision",
            "torch_dtype": "float16",
            "device_map": "cuda",
            "quantization": {
                "method": "bitsandbytes",
                "mode": "4bit",
                "quant_type": "nf4",
                "compute_dtype": "float16",
                "double_quant": True,
            },
        }
    )

    client._load_qwen()

    quantization = captured["quantization_config"]
    assert quantization.kwargs["load_in_4bit"] is True
    assert quantization.kwargs["bnb_4bit_quant_type"] == "nf4"
    assert quantization.kwargs["bnb_4bit_use_double_quant"] is True
    assert str(quantization.kwargs["bnb_4bit_compute_dtype"]) == "torch.float16"
    assert captured["device_map"] == "cuda"


def test_vintern_uses_native_batch_chat_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    torch = _install_fake_torch(monkeypatch)

    class FakeTokenizer:
        pass

    class FakeModel:
        batch_calls = 0

        def parameters(self):
            return iter([SimpleNamespace(device="cpu", dtype=torch.float32)])

        def batch_chat(
            self,
            _tokenizer,
            _pixels,
            questions,
            _generation_config,
            **_kwargs,
        ):
            self.batch_calls += 1
            return [f'{{"value": "{question[-1]}"}}' for question in questions]

    model = FakeModel()
    client = LocalVisionStructuredClient(
        model_config={
            "provider": "vintern_local",
            "model_id": "vintern",
            "model_revision": "revision",
            "inference_batch_size": 2,
        }
    )
    monkeypatch.setattr(client, "_load_vintern", lambda: (FakeTokenizer(), model))
    monkeypatch.setattr(
        "system1.vlm.client._vintern_pixel_values",
        lambda _path, _max_num=1: _FakeTensor(),
    )
    requests = _image_requests(tmp_path, prompts=["a", "b"])

    responses = client.request_many(requests)

    assert len(responses) == 2
    assert model.batch_calls == 1


def test_vintern_without_native_batch_safely_uses_one_request(
    tmp_path: Path, monkeypatch
) -> None:
    torch = _install_fake_torch(monkeypatch)

    lifecycle = []

    class FakeModel:
        chat_calls = 0

        def parameters(self):
            return iter([SimpleNamespace(device="cpu", dtype=torch.float32)])

        def chat(self, _tokenizer, _pixels, prompt, _generation_config):
            self.chat_calls += 1
            return f'{{"value": "{prompt[-1]}"}}'

    model = FakeModel()
    client = LocalVisionStructuredClient(
        model_config={
            "provider": "vintern_local",
            "model_id": "vintern",
            "model_revision": "revision",
            "inference_batch_size": 2,
        },
        lifecycle_callback=lambda payload: lifecycle.append(dict(payload)),
    )
    monkeypatch.setattr(client, "_load_vintern", lambda: (object(), model))
    monkeypatch.setattr(
        "system1.vlm.client._vintern_pixel_values",
        lambda _path, _max_num=1: _FakeTensor(),
    )

    responses = client.request_many(_image_requests(tmp_path, prompts=["a", "b"]))

    assert len(responses) == 2
    assert model.chat_calls == 2
    assert any(
        event["status"] == "batch_capability_fallback"
        and event["effective_batch_size"] == 1
        for event in lifecycle
    )


def _image_requests(tmp_path: Path, *, prompts: list[str]) -> list[GeminiRequest]:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    requests = []
    for index, prompt in enumerate(prompts):
        image = tmp_path / f"image_{index}.jpg"
        image.write_bytes(prompt.encode())
        requests.append(
            GeminiRequest(
                request_kind="keyframe_ocr",
                video_id="L21_V001",
                prompt=prompt,
                prompt_version="prompt",
                response_schema_version="schema",
                response_schema=schema,
                image_paths=(image,),
            )
        )
    return requests


def test_invalid_json_falls_back_only_failed_request_and_keeps_qwen_primary(
    monkeypatch,
) -> None:
    calls = []
    primary = LocalVisionStructuredClient(
        model_config={
            "provider": "qwen_local",
            "model_id": "qwen",
            "model_revision": "revision",
            "inference_batch_size": 2,
        }
    )

    def call_models(requests):
        calls.append([request.prompt for request in requests])
        return [
            '{"value": "primary"}' if request.prompt != "bad" else "not-json"
            for request in requests
        ]

    monkeypatch.setattr(primary, "_call_models", call_models)

    class GeminiFallback:
        provider_name = "gemini"

        def request(self, request):
            return {"value": f"fallback:{request.prompt}"}

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def make_request(prompt):
        return GeminiRequest(
            request_kind="shot_caption",
            video_id="L21_V001",
            prompt=prompt,
            prompt_version="prompt",
            response_schema_version="schema",
            response_schema=schema,
        )

    client = FallbackStructuredClient([primary, GeminiFallback()])
    responses = client.request_many([make_request("good"), make_request("bad")])
    next_response = client.request(make_request("next"))

    assert [response["value"] for response in responses] == [
        "primary",
        "fallback:bad",
    ]
    assert next_response["value"] == "primary"
    assert client.circuit_open is False
    assert calls == [["good", "bad"], ["next"]]


def test_repeated_batch_one_oom_opens_circuit_and_uses_gemini(monkeypatch) -> None:
    attempts = 0
    primary = LocalVisionStructuredClient(
        model_config={
            "provider": "qwen_local",
            "model_id": "qwen",
            "model_revision": "revision",
            "inference_batch_size": 1,
            "total_attempts": 2,
        }
    )

    def oom(_requests):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(primary, "_call_models", oom)

    class GeminiFallback:
        provider_name = "gemini"

        def request(self, _request):
            return {"value": "fallback"}

    request = GeminiRequest(
        request_kind="scene_summary",
        video_id="L21_V001",
        prompt="summary",
        prompt_version="prompt",
        response_schema_version="schema",
        response_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    client = FallbackStructuredClient([primary, GeminiFallback()])

    response = client.request(request)

    assert response == {"value": "fallback"}
    assert attempts == 2
    assert client.circuit_open is True


def _write_image(path: Path, width: int, height: int) -> Path:
    from PIL import Image

    Image.new("RGB", (width, height), (128, 128, 128)).save(path)
    return path


def test_wide_frame_is_split_into_multiple_tiles(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from system1.vlm.client import _vintern_pixel_values

    image = _write_image(tmp_path / "wide.jpg", 1920, 1080)

    tensor = _vintern_pixel_values(image, 4)

    assert tensor.shape[0] > 1
    assert tensor.shape[1:] == (3, 448, 448)


def test_max_num_one_keeps_legacy_single_tile_behaviour(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from system1.vlm.client import _vintern_pixel_values

    image = _write_image(tmp_path / "wide.jpg", 1920, 1080)

    tensor = _vintern_pixel_values(image, 1)

    assert tensor.shape == (1, 3, 448, 448)


def test_tile_count_never_exceeds_max_num(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from system1.vlm.client import _vintern_pixel_values

    image = _write_image(tmp_path / "panorama.jpg", 2400, 480)

    tensor = _vintern_pixel_values(image, 4)

    assert 1 <= tensor.shape[0] <= 4


def test_square_frame_uses_square_grid(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from system1.vlm.client import _vintern_tiles
    from PIL import Image

    with Image.new("RGB", (800, 800), (10, 10, 10)) as image:
        tiles = _vintern_tiles(image, 4)

    assert len(tiles) == 4


def test_small_image_is_upscaled_without_error(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from system1.vlm.client import _vintern_pixel_values

    image = _write_image(tmp_path / "tiny.jpg", 120, 90)

    tensor = _vintern_pixel_values(image, 4)

    assert tensor.shape[1:] == (3, 448, 448)


def test_patch_bookkeeping_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    _install_fake_torch(monkeypatch)

    class FakeModel:
        def parameters(self):
            return iter([])

        def batch_chat(self, *_args, **_kwargs):
            raise AssertionError("model must not be called on a mismatch")

    client = LocalVisionStructuredClient(
        model_config={
            "provider": "vintern_local",
            "model_id": "vintern",
            "model_revision": "revision",
            "inference_batch_size": 2,
            "max_dynamic_patch": 4,
        }
    )
    monkeypatch.setattr(client, "_load_vintern", lambda: (object(), FakeModel()))
    # Tensor claims 2 tiles but torch.cat is stubbed to report 1 — the guard must catch it.
    monkeypatch.setattr(
        "system1.vlm.client._vintern_pixel_values",
        lambda _path, _max_num=1: _FakeTensor(2),
    )
    monkeypatch.setattr("torch.cat", lambda _tensors, dim=0: _FakeTensor(1), raising=False)

    with pytest.raises(Exception) as excinfo:
        client._call_vintern_many(_image_requests(tmp_path, prompts=["a", "b"]))

    assert "patch bookkeeping mismatch" in str(excinfo.value)
