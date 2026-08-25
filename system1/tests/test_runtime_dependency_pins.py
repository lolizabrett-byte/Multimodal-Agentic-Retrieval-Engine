from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

SYSTEM1_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_01 = SYSTEM1_ROOT / "notebooks" / "01_worker_structure_pipeline.ipynb"


def _production_requirements() -> list[str]:
    data = tomllib.loads((SYSTEM1_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["phase01-production"]


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_01.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_transformers_is_capped_below_five():
    """InternVL remote code calls .item() in __init__.

    transformers 5.x materialises models on the meta device, where .item()
    raises, so both Vintern models fail to load. Verified on Kaggle: 4/4 loading
    strategies died until the version was pinned.
    """
    spec = next(r for r in _production_requirements() if r.startswith("transformers"))

    assert "<5" in spec, f"transformers must stay below 5.0, got {spec!r}"


def test_torch_stays_within_the_verified_range():
    spec = next(r for r in _production_requirements() if r.startswith("torch>"))

    assert "<2.9" in spec


def test_notebook_pins_transformers_before_running_the_pipeline():
    """Kaggle preinstalls transformers 5.x, and `pip install -e` leaves it alone.

    Without an explicit downgrade the notebook loads the wrong version no matter
    what pyproject says.
    """
    source = _notebook_source()

    assert re.search(r'"transformers>=[\d.]+,<5\.0"', source), (
        "notebook must force transformers below 5.0 before the pipeline runs"
    )
    assert "huggingface_hub" in source, "hub must be pinned alongside transformers"


def test_notebook_fails_loudly_when_the_pin_does_not_take():
    source = _notebook_source()

    assert "must stay below 5.0" in source, (
        "notebook must raise when transformers 5.x survives the install step"
    )


def test_torchaudio_is_pinned_alongside_torch():
    """nemo imports torchaudio, and torchaudio pins an exact torch version.

    Unpinned, pip keeps Kaggle's newer torchaudio against our capped torch and
    the shared library will not load:
    libtorchaudio.so: undefined symbol: aoti_torch_create_device_guard.
    Measured on Kaggle 25/08/2026 — it killed the run before any stage started.
    """
    specs = _production_requirements()
    spec = next((r for r in specs if r.startswith("torchaudio")), None)

    assert spec is not None, "torchaudio must be pinned; nemo imports it"
    assert "<2.9" in spec, f"torchaudio must track the torch cap, got {spec!r}"


def _core_requirements() -> list[str]:
    data = tomllib.loads((SYSTEM1_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def test_numpy_is_capped_below_two():
    """nemo_toolkit[asr] requires numpy<2.0, so the runtime always lands on 1.x.

    Without the cap here, pip downgrades numpy underneath wheels that were
    compiled against 2.x, and importing pandas raises
    "numpy.dtype size changed ... Expected 96 from C header, got 88".
    """
    spec = next(r for r in _core_requirements() if r.startswith("numpy"))

    assert "<2.0" in spec, f"numpy must stay below 2.0 for nemo, got {spec!r}"
