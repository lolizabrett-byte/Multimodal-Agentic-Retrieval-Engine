from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from system1.asr import has_audio_stream, transcribe_video
from system1.config import load_configs


def _default_config(provider: str | None) -> dict[str, Any]:
    models = load_configs(_config_dir())["models"]["phase01"]
    if provider is None:
        return dict(models["asr"])
    providers = models.get("asr_providers", {})
    if provider not in providers:
        raise ValueError(
            f"unknown ASR provider {provider!r}; available: {sorted(providers)}"
        )
    return dict(providers[provider])


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def extract_audio_text(
    file: str | Path,
    *,
    provider: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Transcribe an audio or video file into timestamped segments.

    Returns one dict per segment with `start`, `end` (seconds) and `text`.
    An empty list means the file carries no speech — not an error.
    """
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"audio/video file not found: {path}")

    model_config = dict(config) if config is not None else _default_config(provider)

    if not has_audio_stream(path):
        return []

    result = transcribe_video(
        path,
        video_id=path.stem,
        frame_timeline=[],
        config=model_config,
    )
    return [
        {
            "start": float(row["start_sec"]),
            "end": float(row["end_sec"]),
            "text": str(row["text"]),
            "language": row.get("language"),
        }
        for row in result.rows
        if str(row.get("text", "")).strip()
    ]
