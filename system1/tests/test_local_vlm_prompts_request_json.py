from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SYSTEM1_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = SYSTEM1_ROOT / "prompts"
MODELS = SYSTEM1_ROOT / "configs" / "models.yaml"

PROMPT_KEYS = (
    "prompt_version",
    "focused_prompt_version",
    "consistency_prompt_version",
)


def _phase01_models() -> dict:
    return yaml.safe_load(MODELS.read_text(encoding="utf-8"))["phase01"]


def _local_prompt_versions() -> list[tuple[str, str]]:
    """Prompt versions served by a local VLM, which has no API-side schema."""
    found: list[tuple[str, str]] = []
    models = _phase01_models()
    for stage, config in models.items():
        if not isinstance(config, dict):
            continue
        provider = config.get("provider")
        if provider is None and config.get("model_key"):
            provider = models[config["model_key"]].get("provider")
        if provider not in {"vintern_local", "qwen_local"}:
            continue
        for key in PROMPT_KEYS:
            version = config.get(key)
            if version:
                found.append((f"{stage}.{key}", str(version)))
    return found


def test_every_configured_prompt_file_exists():
    models = _phase01_models()
    missing = []
    for stage, config in models.items():
        if not isinstance(config, dict):
            continue
        versions = [config.get(key) for key in PROMPT_KEYS]
        versions += [fb.get("prompt_version") for fb in config.get("fallbacks") or []]
        for version in filter(None, versions):
            if not (PROMPTS / f"{version}.txt").is_file():
                missing.append(f"{stage}: {version}")

    assert not missing, f"prompt files referenced but absent: {missing}"


def test_local_stages_are_actually_configured():
    """Guards the test below from silently passing on an empty list."""
    assert _local_prompt_versions(), "no local VLM stage found in models.yaml"


@pytest.mark.parametrize("stage,version", _local_prompt_versions())
def test_local_prompt_asks_for_a_bare_json_object(stage: str, version: str):
    """Local models return whatever the prompt asks for.

    Gemini enforces the response schema API-side, so a prompt that merely names
    the fields is enough there. Vintern has no such guardrail: naming the fields
    without demanding JSON yields plain prose, and every row fails to parse.
    """
    text = (PROMPTS / f"{version}.txt").read_text(encoding="utf-8")

    assert "{" in text and "}" in text, (
        f"{stage} ({version}) must show the exact JSON object to return"
    )
    assert "json" in text.lower(), f"{stage} ({version}) must say the reply is JSON"
    assert "nothing else" in text.lower(), (
        f"{stage} ({version}) must forbid prose around the JSON"
    )
