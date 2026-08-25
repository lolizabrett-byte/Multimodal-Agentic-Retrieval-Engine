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
