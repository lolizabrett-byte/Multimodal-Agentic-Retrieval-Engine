"""The judge/caption model must not run eager attention on a T4.

Vintern's remote code hardcodes `_attn_implementation = 'eager'` whenever
use_flash_attn is off, and eager is the one path where an fp16 overflow becomes
NaN (transformers#33294, fixed for eager only in #33312). Qwen2Attention reads
the setting on every forward, so overriding it after load is enough.
"""

from __future__ import annotations

from types import SimpleNamespace

from system1.vlm.client import _apply_attn_implementation


class _Config:
    def __init__(self, value: str = "eager") -> None:
        self._attn_implementation = value


class _Layer:
    def __init__(self, config: _Config) -> None:
        self.config = config


class _LanguageModel:
    def __init__(self, config: _Config, extra_configs: list[_Config] | None = None) -> None:
        self.config = config
        self._extra = extra_configs or []

    def modules(self):
        yield self
        for config in self._extra:
            yield _Layer(config)


def _model(value: str = "eager", extra: list[_Config] | None = None):
    return SimpleNamespace(language_model=_LanguageModel(_Config(value), extra))


def test_sdpa_replaces_the_eager_default():
    model = _model()
    assert _apply_attn_implementation(model, "sdpa") == "sdpa"
    assert model.language_model.config._attn_implementation == "sdpa"


def test_eager_request_leaves_the_model_alone():
    model = _model()
    assert _apply_attn_implementation(model, "eager") == "eager"
    assert model.language_model.config._attn_implementation == "eager"


def test_no_setting_keeps_whatever_the_remote_code_chose():
    model = _model()
    assert _apply_attn_implementation(model, None) == "eager"
    assert model.language_model.config._attn_implementation == "eager"


def test_child_modules_holding_their_own_config_are_updated():
    """Some transformers versions copy the config instead of sharing it."""
    extra = _Config("eager")
    model = _model(extra=[extra])
    _apply_attn_implementation(model, "sdpa")
    assert extra._attn_implementation == "sdpa"


def test_model_without_a_language_model_does_not_raise():
    assert _apply_attn_implementation(SimpleNamespace(), "sdpa") is None


def test_model_without_a_config_does_not_raise():
    assert _apply_attn_implementation(
        SimpleNamespace(language_model=SimpleNamespace()), "sdpa"
    ) is None
