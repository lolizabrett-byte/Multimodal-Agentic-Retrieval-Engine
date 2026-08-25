from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from system1.gemini import StructuredRequest
from system1.phase01.production import _ocr_text
from system1.vlm.client import LocalVisionStructuredClient

OCR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "full_text": {"type": "string"},
        "ocr_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": ["number", "null"]},
                },
                "required": ["text"],
                "additionalProperties": True,
            },
        },
        "language": {"type": "string"},
        "confidence": {"type": ["number", "null"]},
    },
    "required": ["full_text", "ocr_blocks"],
    "additionalProperties": False,
}

# Loading Vintern takes tens of seconds, so a module-level client keeps repeated
# calls usable interactively. Batch work should go through the Phase01 pipeline,
# which manages GPU residency properly.
_CLIENT: LocalVisionStructuredClient | None = None
_CLIENT_KEY: tuple[Any, ...] | None = None


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def _default_config() -> dict[str, Any]:
    from system1.config import load_configs

    return dict(load_configs(_config_dir())["models"]["phase01"]["ocr"])


def _prompt_text(prompt_version: str) -> str:
    prompts = Path(__file__).resolve().parents[3] / "prompts"
    return (prompts / f"{prompt_version}.txt").read_text(encoding="utf-8")


def _client_for(model_config: Mapping[str, Any]) -> LocalVisionStructuredClient:
    global _CLIENT, _CLIENT_KEY

    key = (
        model_config.get("model_id"),
        model_config.get("model_revision"),
        model_config.get("torch_dtype"),
        model_config.get("device_map"),
        model_config.get("max_dynamic_patch"),
    )
    if _CLIENT is None or _CLIENT_KEY != key:
        if _CLIENT is not None:
            _CLIENT.close()
        _CLIENT = LocalVisionStructuredClient(model_config=dict(model_config))
        _CLIENT_KEY = key
    return _CLIENT


def _as_image_path(image: Any, stack: list[Path]) -> Path:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        return path

    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        pil = image
    else:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TypeError(f"unsupported image type: {type(image).__name__}") from exc
        if not isinstance(image, np.ndarray):
            raise TypeError(f"unsupported image type: {type(image).__name__}")
        array = image
        if array.ndim == 3 and array.shape[2] == 3:
            # OpenCV hands back BGR; PIL expects RGB.
            array = array[:, :, ::-1]
        pil = PILImage.fromarray(array)

    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    handle.close()
    path = Path(handle.name)
    pil.convert("RGB").save(path)
    stack.append(path)
    return path


def extract_image_text(
    image: Any,
    *,
    preprocess: str | None = None,
    max_num: int | None = None,
    config: Mapping[str, Any] | None = None,
) -> str:
    """OCR one image and return the visible text.

    Accepts a path, a PIL image, or a numpy array. Returns an empty string when
    the image carries no readable text.
    """
    model_config = dict(config) if config is not None else _default_config()
    if max_num is not None:
        model_config["max_dynamic_patch"] = int(max_num)

    temporary: list[Path] = []
    try:
        path = _as_image_path(image, temporary)
        if preprocess:
            from system1.handoff.preprocess import apply_preprocess

            path = apply_preprocess(path, preprocess, temporary)

        request = StructuredRequest(
            request_kind="keyframe_ocr",
            video_id=path.stem,
            prompt=_prompt_text(str(model_config["prompt_version"])),
            prompt_version=str(model_config["prompt_version"]),
            response_schema_version=str(model_config["response_schema_version"]),
            response_schema=OCR_RESPONSE_SCHEMA,
            image_paths=(path,),
            identity={"keyframe_id": path.stem},
        )
        response = _client_for(model_config).request(request)
        return _ocr_text(response)
    finally:
        for leftover in temporary:
            leftover.unlink(missing_ok=True)
