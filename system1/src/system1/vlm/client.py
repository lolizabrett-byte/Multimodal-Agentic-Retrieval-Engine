from __future__ import annotations

import gc
import json
import re
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from PIL import Image

from system1.gemini import StructuredRequest, build_request_hash


class JsonCache(Protocol):
    def exists(self, relative_path: str | Path) -> bool: ...

    def read_json(self, relative_path: str | Path) -> dict[str, Any]: ...

    def write_json(self, relative_path: str | Path, payload: dict[str, Any]) -> Path: ...


class StructuredClient(Protocol):
    def request(self, request: StructuredRequest) -> dict[str, Any]: ...

    def request_many(
        self, requests: list[StructuredRequest]
    ) -> list[dict[str, Any]]: ...


class SystemicProviderError(RuntimeError):
    """The provider runtime cannot safely serve more requests in this chunk."""


class BatchRequestError(RuntimeError):
    """Carries completed results while exposing request-specific errors."""

    def __init__(
        self,
        *,
        results: list[dict[str, Any] | None],
        errors: Mapping[int, Exception],
    ) -> None:
        self.results = results
        self.errors = dict(errors)
        details = " | ".join(
            f"request[{index}] {type(error).__name__}: {error}"
            for index, error in sorted(self.errors.items())
        )
        super().__init__(details or "structured request batch failed")


class _NativeBatchUnavailable(RuntimeError):
    pass


class MetadataStructuredClient:
    def __init__(
        self,
        client: StructuredClient,
        *,
        provider_name: str,
        model_id: str,
        model_revision: str,
    ) -> None:
        self.client = client
        self.provider_name = provider_name
        self.model_id = model_id
        self.model_revision = model_revision

    def request(self, request: StructuredRequest) -> dict[str, Any]:
        return self._with_metadata(self.client.request(request))

    def request_many(
        self, requests: list[StructuredRequest]
    ) -> list[dict[str, Any]]:
        try:
            responses = _client_request_many(self.client, requests)
        except BatchRequestError as exc:
            raise BatchRequestError(
                results=[
                    self._with_metadata(result) if result is not None else None
                    for result in exc.results
                ],
                errors=exc.errors,
            ) from exc
        return [self._with_metadata(response) for response in responses]

    def _with_metadata(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload.setdefault("__provider", self.provider_name)
        payload.setdefault("__model_id", self.model_id)
        payload.setdefault("__model_revision", self.model_revision)
        return payload

    def close(self) -> None:
        _close_client(self.client)


class FallbackStructuredClient:
    """Request-level fallback with a systemic-failure circuit breaker."""

    def __init__(
        self,
        clients: list[StructuredClient],
        *,
        telemetry_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not clients:
            raise ValueError("at least one structured client is required")
        self.clients = clients
        self.telemetry_callback = telemetry_callback
        self._request_lock = threading.Lock()
        self._circuit_open = False
        self._counts = {
            "qwen_request_count": 0,
            "gemini_request_count": 0,
            "fallback_request_count": 0,
        }

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    def request(self, request: StructuredRequest) -> dict[str, Any]:
        try:
            return self.request_many([request])[0]
        except BatchRequestError as exc:
            raise RuntimeError(str(exc)) from exc

    def request_many(
        self, requests: list[StructuredRequest]
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        with self._request_lock:
            if self._circuit_open or len(self.clients) == 1:
                start = 1 if self._circuit_open and len(self.clients) > 1 else 0
                return self._request_all_from(start, requests)

            primary = self.clients[0]
            self._record_provider_requests(primary, len(requests))
            try:
                return _client_request_many(primary, requests)
            except BatchRequestError as exc:
                results = list(exc.results)
                failed_indices = sorted(exc.errors)
                systemic = any(
                    _is_systemic_provider_error(error)
                    for error in exc.errors.values()
                )
                reason = _fallback_reason(next(iter(exc.errors.values())))
                if systemic:
                    self._open_circuit(primary, reason=reason)
                self._fallback_indices(
                    requests,
                    results,
                    failed_indices,
                    errors=exc.errors,
                    reason=reason,
                )
                return _complete_batch_or_raise(results)
            except Exception as exc:  # noqa: BLE001 - provider boundary
                if _is_systemic_provider_error(exc):
                    self._open_circuit(primary, reason=_fallback_reason(exc))
                results: list[dict[str, Any] | None] = [None] * len(requests)
                self._fallback_indices(
                    requests,
                    results,
                    list(range(len(requests))),
                    errors={index: exc for index in range(len(requests))},
                    reason=_fallback_reason(exc),
                )
                return _complete_batch_or_raise(results)

    def _request_all_from(
        self, start: int, requests: list[StructuredRequest]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(requests)
        self._fallback_indices(
            requests,
            results,
            list(range(len(requests))),
            errors={},
            reason="circuit_open" if self._circuit_open else "primary_only",
            start=start,
        )
        return _complete_batch_or_raise(results)

    def _fallback_indices(
        self,
        requests: list[StructuredRequest],
        results: list[dict[str, Any] | None],
        indices: list[int],
        *,
        errors: Mapping[int, Exception],
        reason: str,
        start: int = 1,
    ) -> None:
        unresolved = list(indices)
        all_errors: dict[int, Exception] = dict(errors)
        for client in self.clients[start:]:
            if not unresolved:
                break
            next_unresolved: list[int] = []
            for index in unresolved:
                self._counts["fallback_request_count"] += 1
                self._record_provider_requests(client, 1)
                try:
                    results[index] = client.request(requests[index])
                    all_errors.pop(index, None)
                except Exception as exc:  # noqa: BLE001 - try next fallback
                    previous = all_errors.get(index)
                    all_errors[index] = (
                        RuntimeError(
                            f"{type(previous).__name__}: {previous} | "
                            f"{type(exc).__name__}: {exc}"
                        )
                        if previous is not None
                        else exc
                    )
                    next_unresolved.append(index)
            unresolved = next_unresolved
        self._emit(
            "fallback",
            fallback_reason=reason,
            circuit_breaker_state="open" if self._circuit_open else "closed",
            unresolved_count=len(unresolved),
            **self._counts,
        )
        if unresolved:
            raise BatchRequestError(results=results, errors=all_errors)

    def _record_provider_requests(self, client: StructuredClient, count: int) -> None:
        provider = str(getattr(client, "provider_name", ""))
        if provider == "qwen_local":
            self._counts["qwen_request_count"] += count
        elif provider == "gemini":
            self._counts["gemini_request_count"] += count

    def _open_circuit(self, primary: StructuredClient, *, reason: str) -> None:
        if self._circuit_open:
            return
        self._circuit_open = True
        _close_client(primary)
        _release_torch_memory()
        self._emit(
            "circuit_breaker",
            circuit_breaker_state="open",
            circuit_breaker_reason=reason,
            **self._counts,
        )

    def _emit(self, status: str, **details: Any) -> None:
        if self.telemetry_callback is None:
            return
        try:
            self.telemetry_callback({"status": status, **details})
        except Exception as exc:  # noqa: BLE001 - telemetry cannot break inference
            warnings.warn(
                f"structured-client telemetry callback failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def close(self) -> None:
        with self._request_lock:
            for client in self.clients:
                _close_client(client)
            _release_torch_memory()


class LocalVisionStructuredClient:
    def __init__(
        self,
        *,
        model_config: Mapping[str, Any],
        cache: JsonCache | None = None,
        cache_prefix: str | Path = "cache/local_vlm",
        lifecycle_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.model_config = dict(model_config)
        self.provider_name = str(self.model_config["provider"])
        self.model_id = str(self.model_config["model_id"])
        self.model_revision = str(self.model_config.get("model_revision") or self.model_id)
        self.cache = cache
        self.cache_prefix = Path(cache_prefix)
        self.lifecycle_callback = lifecycle_callback
        self.total_attempts = int(self.model_config.get("total_attempts", 2))
        self.inference_batch_size = int(
            self.model_config.get("inference_batch_size", 1)
        )
        self.max_dynamic_patch = int(self.model_config.get("max_dynamic_patch", 1))
        if self.total_attempts < 1:
            raise ValueError("local VLM total_attempts must be positive")
        if self.inference_batch_size < 1:
            raise ValueError("local VLM inference_batch_size must be positive")
        if self.max_dynamic_patch < 1:
            raise ValueError("local VLM max_dynamic_patch must be positive")
        self._cache_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._loaded: tuple[Any, ...] | None = None

    def generation_config(self) -> dict[str, Any]:
        """Sampling settings handed to the model, built from model_config.

        Optional keys are omitted rather than passed as None, so a config that
        says nothing keeps the model's own defaults.
        """
        config: dict[str, Any] = {
            "max_new_tokens": int(self.model_config.get("max_new_tokens", 768)),
            "do_sample": False,
        }
        penalty = self.model_config.get("repetition_penalty")
        if penalty is not None:
            try:
                value = float(penalty)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"repetition_penalty must be a number, got {penalty!r}"
                ) from exc
            if value <= 0:
                raise ValueError(
                    f"repetition_penalty must be positive, got {value!r}"
                )
            config["repetition_penalty"] = value
        ngram = self.model_config.get("no_repeat_ngram_size")
        if ngram is not None:
            try:
                size = int(ngram)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"no_repeat_ngram_size must be an integer, got {ngram!r}"
                ) from exc
            if size < 1:
                raise ValueError(
                    f"no_repeat_ngram_size must be positive, got {size!r}"
                )
            config["no_repeat_ngram_size"] = size
        return config

    def request(self, request: StructuredRequest) -> dict[str, Any]:
        try:
            return self.request_many([request])[0]
        except BatchRequestError as exc:
            raise next(iter(exc.errors.values())) from exc

    def request_many(
        self, requests: list[StructuredRequest]
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        results: list[dict[str, Any] | None] = [None] * len(requests)
        errors: dict[int, Exception] = {}
        failure_samples: list[dict[str, Any]] = []
        misses: list[int] = []
        cache_paths: dict[int, Path] = {}
        for index, request in enumerate(requests):
            request_hash = self._request_hash(request)
            cache_path = self.cache_prefix / f"{request_hash}.json"
            cache_paths[index] = cache_path
            cached = self._read_cached(cache_path, request)
            if cached is None:
                misses.append(index)
            else:
                results[index] = self._with_metadata(cached)

        requested_batch_size = min(self.inference_batch_size, max(1, len(misses)))
        effective_batch_size = requested_batch_size
        cache_hits = len(requests) - len(misses)
        self._emit_lifecycle(
            "batch_start",
            requested_batch_size=requested_batch_size,
            effective_batch_size=effective_batch_size,
            request_count=len(requests),
            cache_hits=cache_hits,
            cache_misses=len(misses),
            quantization_mode=self._quantization_mode(),
        )
        cursor = 0
        oom_reductions = 0
        batch_one_oom_attempts = 0
        systemic_attempts = 0
        while cursor < len(misses):
            batch_indices = misses[cursor : cursor + effective_batch_size]
            batch_requests = [requests[index] for index in batch_indices]
            try:
                raw_texts = self._call_models(batch_requests)
            except _NativeBatchUnavailable:
                if effective_batch_size == 1:
                    raise
                previous = effective_batch_size
                effective_batch_size = 1
                self._emit_lifecycle(
                    "batch_capability_fallback",
                    requested_batch_size=requested_batch_size,
                    previous_batch_size=previous,
                    effective_batch_size=1,
                    quantization_mode=self._quantization_mode(),
                )
                continue
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _release_torch_memory()
                    if effective_batch_size > 1:
                        previous = effective_batch_size
                        effective_batch_size = max(1, effective_batch_size // 2)
                        oom_reductions += 1
                        self._emit_lifecycle(
                            "oom_reduction",
                            requested_batch_size=requested_batch_size,
                            previous_batch_size=previous,
                            effective_batch_size=effective_batch_size,
                            oom_reductions=oom_reductions,
                            error_type=type(exc).__name__,
                            quantization_mode=self._quantization_mode(),
                        )
                        continue
                    batch_one_oom_attempts += 1
                    if batch_one_oom_attempts < self.total_attempts:
                        self.close()
                        self._emit_lifecycle(
                            "oom_retry",
                            requested_batch_size=requested_batch_size,
                            effective_batch_size=1,
                            attempt=batch_one_oom_attempts,
                            total_attempts=self.total_attempts,
                            oom_reductions=oom_reductions,
                            error_type=type(exc).__name__,
                            quantization_mode=self._quantization_mode(),
                        )
                        continue
                    self.close()
                    systemic = SystemicProviderError(
                        f"{self.provider_name} CUDA OOM at batch_size=1: {exc}"
                    )
                    for index in misses[cursor:]:
                        errors[index] = systemic
                    raise BatchRequestError(results=results, errors=errors) from exc
                if _is_systemic_runtime_error(exc):
                    systemic_attempts += 1
                    if systemic_attempts < self.total_attempts:
                        self.close()
                        self._emit_lifecycle(
                            "runtime_retry",
                            requested_batch_size=requested_batch_size,
                            effective_batch_size=effective_batch_size,
                            attempt=systemic_attempts,
                            total_attempts=self.total_attempts,
                            error_type=type(exc).__name__,
                            quantization_mode=self._quantization_mode(),
                        )
                        continue
                    self.close()
                    systemic = SystemicProviderError(
                        f"{self.provider_name} runtime unavailable: {exc}"
                    )
                    for index in misses[cursor:]:
                        errors[index] = systemic
                    raise BatchRequestError(results=results, errors=errors) from exc
                for index in batch_indices:
                    errors[index] = exc
                cursor += len(batch_indices)
                continue

            if len(raw_texts) != len(batch_requests):
                exc = ValueError(
                    f"local VLM returned {len(raw_texts)} responses for "
                    f"{len(batch_requests)} requests"
                )
                for index in batch_indices:
                    errors[index] = exc
                cursor += len(batch_indices)
                continue
            for index, request, raw_text in zip(
                batch_indices, batch_requests, raw_texts, strict=True
            ):
                try:
                    normalized = _parse_json_object(raw_text, request.response_schema)
                    self._write_cached(
                        cache_paths[index], request=request, normalized=normalized
                    )
                    results[index] = self._with_metadata(normalized)
                except Exception as exc:  # noqa: BLE001 - request-scoped fallback
                    errors[index] = exc
                    _record_failure_sample(failure_samples, exc, raw_text)
            cursor += len(batch_indices)
            batch_one_oom_attempts = 0
            systemic_attempts = 0

        self._emit_lifecycle(
            "batch_complete",
            requested_batch_size=requested_batch_size,
            effective_batch_size=effective_batch_size,
            request_count=len(requests),
            cache_hits=cache_hits,
            cache_misses=len(misses),
            oom_reductions=oom_reductions,
            failed_request_count=len(errors),
            failure_samples=failure_samples,
            quantization_mode=self._quantization_mode(),
        )
        if errors:
            raise BatchRequestError(results=results, errors=errors)
        return _complete_batch_or_raise(results)

    def _request_hash(self, request: StructuredRequest) -> str:
        request_hash = build_request_hash(
            request,
            model_id=self.model_id,
            cache_identity={
                "provider": self.provider_name,
                "model_revision": self.model_revision,
                "max_new_tokens": self.model_config.get("max_new_tokens"),
                "quantization": self.model_config.get("quantization"),
            },
        )
        return request_hash

    def _read_cached(
        self, cache_path: Path, request: StructuredRequest
    ) -> dict[str, Any] | None:
        if self.cache is None:
            return None
        with self._cache_lock:
            if not self.cache.exists(cache_path):
                return None
            cached = self.cache.read_json(cache_path)
            response = cached.get("normalized_response")
            if not isinstance(response, dict):
                return None
            try:
                validate(response, request.response_schema)
            except Exception:  # noqa: BLE001 - corrupt cache entries are misses
                return None
            return response

    def _write_cached(
        self,
        cache_path: Path,
        *,
        request: StructuredRequest,
        normalized: dict[str, Any],
    ) -> None:
        if self.cache is None:
            return
        with self._cache_lock:
            self.cache.write_json(
                cache_path,
                {
                    "schema_version": "local_vlm_cache_entry_v1",
                    "request_hash": cache_path.stem,
                    "request_kind": request.request_kind,
                    "video_id": request.video_id,
                    "provider": self.provider_name,
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "prompt_version": request.prompt_version,
                    "response_schema_version": request.response_schema_version,
                    "normalized_response": normalized,
                },
            )

    def _with_metadata(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = dict(payload)
        response["__provider"] = self.provider_name
        response["__model_id"] = self.model_id
        response["__model_revision"] = self.model_revision
        return response

    def _call_models(self, requests: list[StructuredRequest]) -> list[str]:
        if self.provider_name == "qwen_local":
            return self._call_qwen_many(requests)
        if self.provider_name == "vintern_local":
            return self._call_vintern_many(requests)
        raise SystemicProviderError(
            f"unsupported local VLM provider: {self.provider_name}"
        )

    def _call_qwen_many(self, requests: list[StructuredRequest]) -> list[str]:
        processor, model = self._load_qwen()
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - production dependency guard
            raise SystemicProviderError(
                "qwen-vl-utils is required for qwen_local"
            ) from exc

        inputs: Any = None
        generated_ids: Any = None
        image_inputs: Any = None
        video_inputs: Any = None
        try:
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            *[
                                {"type": "image", "image": str(path)}
                                for path in request.image_paths
                            ],
                            {"type": "text", "text": request.prompt},
                        ],
                    }
                ]
                for request in requests
            ]
            texts = [
                processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                for conversation in conversations
            ]
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            device = _model_device(model)
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            generated_ids = model.generate(**inputs, **self.generation_config())
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                generated_ids = generated_ids[:, input_ids.shape[1] :]
            return processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        finally:
            del inputs, generated_ids, image_inputs, video_inputs

    def _call_vintern_many(self, requests: list[StructuredRequest]) -> list[str]:
        if any(len(request.image_paths) != 1 for request in requests):
            raise RuntimeError("vintern_local expects exactly one image per request")
        tokenizer, model = self._load_vintern()
        import torch

        native_batch = getattr(model, "batch_chat", None)
        if len(requests) > 1 and not callable(native_batch):
            raise _NativeBatchUnavailable("Vintern runtime has no batch_chat")
        pixel_values: Any = None
        try:
            tensors = [
                _vintern_pixel_values(request.image_paths[0], self.max_dynamic_patch)
                for request in requests
            ]
            num_patches_list = [int(tensor.shape[0]) for tensor in tensors]
            pixel_values = torch.cat(tensors, dim=0).to(_model_device(model))
            if sum(num_patches_list) != int(pixel_values.shape[0]):
                raise RuntimeError(
                    "vintern_local patch bookkeeping mismatch: "
                    f"num_patches_list sums to {sum(num_patches_list)} but "
                    f"pixel_values has {int(pixel_values.shape[0])} tiles"
                )
            try:
                dtype = next(model.parameters()).dtype
                pixel_values = pixel_values.to(dtype=dtype)
            except StopIteration:  # pragma: no cover
                pass
            prompts = [
                request.prompt
                if "<image>" in request.prompt
                else "<image>\n" + request.prompt
                for request in requests
            ]
            generation_config = self.generation_config()
            with torch.no_grad():
                if len(requests) > 1:
                    outputs = native_batch(
                        tokenizer,
                        pixel_values,
                        prompts,
                        generation_config,
                        num_patches_list=num_patches_list,
                    )
                else:
                    output = model.chat(
                        tokenizer,
                        pixel_values,
                        prompts[0],
                        generation_config,
                    )
                    outputs = [output[0] if isinstance(output, tuple) else output]
            if not isinstance(outputs, Sequence) or isinstance(outputs, str):
                raise TypeError("vintern_local returned an invalid batch response")
            normalized = [str(output).strip() for output in outputs]
            if any(not output for output in normalized):
                raise ValueError("vintern_local returned an empty response")
            return normalized
        finally:
            del pixel_values

    def _load_qwen(self) -> tuple[Any, Any]:
        with self._model_lock:
            if self._loaded is not None:
                return self._loaded  # type: ignore[return-value]
            started = time.monotonic()
            _reset_cuda_peak_memory()
            try:
                import torch
                from transformers import AutoProcessor, BitsAndBytesConfig
                try:
                    from transformers import (
                        Qwen2_5_VLForConditionalGeneration as AutoModel,
                    )
                except ImportError:
                    try:
                        from transformers import (
                            AutoModelForImageTextToText as AutoModel,
                        )
                    except ImportError:
                        try:
                            from transformers import AutoModelForVision2Seq as AutoModel
                        except ImportError:
                            from transformers import AutoModel
            except ImportError as exc:  # pragma: no cover - production dependency guard
                raise SystemicProviderError(
                    "transformers, torch, and bitsandbytes are required for qwen_local"
                ) from exc
            quantization_config = _qwen_quantization_config(
                self.model_config, BitsAndBytesConfig, torch
            )
            try:
                processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    revision=self.model_revision,
                    trust_remote_code=bool(
                        self.model_config.get("trust_remote_code", False)
                    ),
                )
                model = AutoModel.from_pretrained(
                    self.model_id,
                    revision=self.model_revision,
                    torch_dtype=_torch_dtype(
                        self.model_config.get("torch_dtype", "float16"), torch
                    ),
                    device_map=self.model_config.get("device_map", "cuda"),
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=bool(
                        self.model_config.get("low_cpu_mem_usage", True)
                    ),
                    trust_remote_code=bool(
                        self.model_config.get("trust_remote_code", False)
                    ),
                )
                model.eval()
                self._loaded = (processor, model)
            except Exception as exc:
                _release_torch_memory()
                self._emit_lifecycle(
                    "load_failed",
                    load_seconds=round(time.monotonic() - started, 3),
                    error_type=type(exc).__name__,
                    quantization_mode=self._quantization_mode(),
                )
                raise SystemicProviderError(
                    f"failed to load qwen_local {self.model_id}: {exc}"
                ) from exc
            self._emit_lifecycle(
                "loaded",
                load_seconds=round(time.monotonic() - started, 3),
                quantization_mode=self._quantization_mode(),
            )
            return processor, model

    def _load_vintern(self) -> tuple[Any, Any]:
        with self._model_lock:
            if self._loaded is not None:
                return self._loaded  # type: ignore[return-value]
            started = time.monotonic()
            _reset_cuda_peak_memory()
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - production dependency guard
                raise SystemicProviderError(
                    "transformers is required for vintern_local"
                ) from exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    revision=self.model_revision,
                    trust_remote_code=bool(
                        self.model_config.get("trust_remote_code", True)
                    ),
                    use_fast=bool(
                        self.model_config.get("use_fast_tokenizer", False)
                    ),
                )
                device = str(self.model_config.get("device_map", "cuda:0"))
                # No device_map here: accelerate cannot fully dispatch InternVL's
                # remote code, so the model is materialised once and moved to a
                # single card. low_cpu_mem_usage stays on — Kaggle shares ~13 GB
                # of host RAM between both GPU workers.
                model = AutoModel.from_pretrained(
                    self.model_id,
                    revision=self.model_revision,
                    torch_dtype=_supported_torch_dtype(
                        self.model_config.get("torch_dtype", "float16"), torch
                    ),
                    low_cpu_mem_usage=bool(
                        self.model_config.get("low_cpu_mem_usage", True)
                    ),
                    trust_remote_code=bool(
                        self.model_config.get("trust_remote_code", True)
                    ),
                    use_flash_attn=bool(
                        self.model_config.get("use_flash_attn", False)
                    ),
                )
                if device not in {"cpu", "auto", "none"}:
                    if not torch.cuda.is_available():
                        raise SystemicProviderError(
                            f"vintern_local needs CUDA for device {device}"
                        )
                    model = model.to(device)
                model.eval()
                self._loaded = (tokenizer, model)
            except Exception as exc:
                _release_torch_memory()
                self._emit_lifecycle(
                    "load_failed",
                    load_seconds=round(time.monotonic() - started, 3),
                    error_type=type(exc).__name__,
                    quantization_mode="none",
                )
                raise SystemicProviderError(
                    f"failed to load vintern_local {self.model_id}: {exc}"
                ) from exc
            self._emit_lifecycle(
                "loaded",
                load_seconds=round(time.monotonic() - started, 3),
                quantization_mode="none",
                native_batch_capable=callable(getattr(model, "batch_chat", None)),
            )
            return tokenizer, model

    def _quantization_mode(self) -> str:
        quantization = self.model_config.get("quantization", {})
        if not isinstance(quantization, Mapping):
            return "none"
        return str(quantization.get("mode", "none"))

    def close(self) -> None:
        with self._model_lock:
            was_loaded = self._loaded is not None
            self._loaded = None
        _release_torch_memory()
        if was_loaded:
            self._emit_lifecycle(
                "unloaded", quantization_mode=self._quantization_mode()
            )

    def _emit_lifecycle(self, status: str, **details: Any) -> None:
        if self.lifecycle_callback is None:
            return
        try:
            self.lifecycle_callback(
                {
                    "status": status,
                    "provider": self.provider_name,
                    "model": self.model_id,
                    "model_revision": self.model_revision,
                    **details,
                }
            )
        except Exception:  # noqa: BLE001, S110 - telemetry cannot break inference
            pass


def _qwen_quantization_config(
    model_config: Mapping[str, Any], factory: Any, torch: Any
):
    quantization = model_config.get("quantization")
    if not isinstance(quantization, Mapping):
        raise SystemicProviderError("qwen_local requires explicit quantization config")
    if str(quantization.get("method")) != "bitsandbytes" or str(
        quantization.get("mode")
    ) != "4bit":
        raise SystemicProviderError(
            "qwen_local only supports configured bitsandbytes 4bit loading"
        )
    return factory(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(quantization.get("quant_type", "nf4")),
        bnb_4bit_compute_dtype=_torch_dtype(
            quantization.get("compute_dtype", "float16"), torch
        ),
        bnb_4bit_use_double_quant=bool(quantization.get("double_quant", True)),
    )


def _torch_dtype(value: Any, torch: Any):
    if not isinstance(value, str):
        return value
    normalized = value.lower()
    if normalized == "auto":
        return "auto"
    aliases = {
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
    }
    attribute = aliases.get(normalized)
    if attribute is None:
        raise ValueError(f"unsupported torch dtype: {value}")
    return getattr(torch, attribute)


def _supported_torch_dtype(value: Any, torch: Any):
    """Downgrade bfloat16 to float16 on GPUs that lack bf16 (Kaggle's T4 is Turing).

    Vintern checkpoints ship as bfloat16, so loading them verbatim on a T4 either
    errors or silently falls back to a slow emulated path.
    """
    resolved = _torch_dtype(value, torch)
    if resolved is not getattr(torch, "bfloat16", object()):
        return resolved
    try:
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            return torch.float16
    except Exception:  # pragma: no cover - probing must never break loading
        return torch.float16
    return resolved


def _client_request_many(
    client: StructuredClient, requests: list[StructuredRequest]
) -> list[dict[str, Any]]:
    request_many = getattr(client, "request_many", None)
    if callable(request_many):
        return request_many(requests)
    return [client.request(request) for request in requests]


def _complete_batch_or_raise(
    results: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        raise BatchRequestError(
            results=results,
            errors={
                index: RuntimeError("missing structured response") for index in missing
            },
        )
    return [result for result in results if result is not None]


_MAX_FAILURE_SAMPLES = 3
_MAX_RAW_TEXT_CHARS = 2000


def _record_failure_sample(
    samples: list[dict[str, Any]], exc: BaseException, raw_text: str
) -> None:
    """Keep what the model actually returned, not just the exception.

    Without this the raw text dies with the exception, so a batch that fails
    schema validation is indistinguishable from one that returned prose. Capped
    because a repeating model can emit thousands of characters per request.
    """
    if len(samples) >= _MAX_FAILURE_SAMPLES:
        return
    text = str(raw_text)
    samples.append(
        {
            "reason": _fallback_reason(exc),
            "error": str(exc)[:200],
            "raw_text": text[:_MAX_RAW_TEXT_CHARS],
            "raw_text_length": len(text),
        }
    )


def _fallback_reason(exc: BaseException) -> str:
    if isinstance(exc, SystemicProviderError):
        return "systemic_local_runtime"
    if isinstance(exc, json.JSONDecodeError):
        return "json_decode_error"
    if isinstance(exc, ValidationError):
        return "schema_validation_error"
    name = type(exc).__name__.lower()
    if "validation" in name or "json" in name or isinstance(
        exc, (ValueError, TypeError)
    ):
        return "invalid_structured_response"
    return "request_error"


def _is_systemic_provider_error(exc: BaseException) -> bool:
    return isinstance(exc, SystemicProviderError)


def _is_systemic_runtime_error(exc: BaseException) -> bool:
    if isinstance(exc, SystemicProviderError):
        return True
    message = str(exc).lower()
    markers = (
        "device-side assert",
        "illegal memory access",
        "cuda error",
        "cublas",
        "cudnn",
        "driver shutting down",
    )
    return any(marker in message for marker in markers)


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - optional runtime dependency guard
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except RuntimeError:
        pass


def _reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional runtime dependency guard
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except RuntimeError:
        pass


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "cuda out of memory" in message or (
        "cuda" in message and "out of memory" in message
    )


def _parse_json_object(raw_text: str, schema: dict[str, Any]) -> dict[str, Any]:
    text = raw_text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("structured local VLM response must be an object")
    validate(payload, schema)
    return payload


def _model_device(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return "cpu"


_VINTERN_TILE_SIZE = 448


def _closest_tile_grid(
    aspect_ratio: float, candidates: list[tuple[int, int]], width: int, height: int
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height
    for columns, rows in candidates:
        candidate_ratio = columns / rows
        diff = abs(aspect_ratio - candidate_ratio)
        if diff < best_diff:
            best_diff = diff
            best_ratio = (columns, rows)
        elif diff == best_diff:
            if area > 0.5 * _VINTERN_TILE_SIZE * _VINTERN_TILE_SIZE * columns * rows:
                best_ratio = (columns, rows)
    return best_ratio


def _vintern_tiles(image: Image.Image, max_num: int) -> list[Image.Image]:
    """Split into 448px tiles on the grid whose shape is closest to the source.

    Matches the InternVL dynamic_preprocess contract the model was trained on.
    A single squashed tile destroys small text — the exact content OCR exists to read.
    """
    if max_num <= 1:
        return [image.resize((_VINTERN_TILE_SIZE, _VINTERN_TILE_SIZE))]

    width, height = image.size
    candidates = sorted(
        {
            (columns, rows)
            for total in range(1, max_num + 1)
            for columns in range(1, total + 1)
            for rows in range(1, total + 1)
            if 1 <= columns * rows <= max_num
        },
        key=lambda grid: grid[0] * grid[1],
    )
    columns, rows = _closest_tile_grid(width / height, candidates, width, height)
    resized = image.resize((_VINTERN_TILE_SIZE * columns, _VINTERN_TILE_SIZE * rows))
    return [
        resized.crop(
            (
                (index % columns) * _VINTERN_TILE_SIZE,
                (index // columns) * _VINTERN_TILE_SIZE,
                ((index % columns) + 1) * _VINTERN_TILE_SIZE,
                ((index // columns) + 1) * _VINTERN_TILE_SIZE,
            )
        )
        for index in range(columns * rows)
    ]


def _vintern_pixel_values(image_path: Path, max_num: int = 1):
    import torch

    with Image.open(image_path) as opened:
        tiles = _vintern_tiles(opened.convert("RGB"), max_num)
    array = np.stack([np.asarray(tile) for tile in tiles]).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std
