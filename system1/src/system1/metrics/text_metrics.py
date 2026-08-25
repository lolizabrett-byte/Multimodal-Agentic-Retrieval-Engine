from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

# Tiếng Việt có hai cách mã hoá cùng một chữ (ề = 1 code point gộp, hoặc e + 2 dấu
# rời). Không quy về một dạng thì CER/WER báo sai dù model đọc đúng hoàn toàn.
_UNICODE_FORM = "NFC"


def normalize_vi(text: str, *, lowercase: bool = True) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize(_UNICODE_FORM, text)
    if lowercase:
        normalized = normalized.lower()
    return " ".join(normalized.split())


def _require_jiwer() -> Any:
    try:
        import jiwer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "jiwer is required for CER/WER metrics: uv sync --group dev"
        ) from exc
    return jiwer


def compute_cer(reference: str, hypothesis: str, *, lowercase: bool = True) -> float:
    ref = normalize_vi(reference, lowercase=lowercase)
    hyp = normalize_vi(hypothesis, lowercase=lowercase)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return float(_require_jiwer().cer(ref, hyp))


def compute_wer(reference: str, hypothesis: str, *, lowercase: bool = True) -> float:
    ref = normalize_vi(reference, lowercase=lowercase)
    hyp = normalize_vi(hypothesis, lowercase=lowercase)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return float(_require_jiwer().wer(ref, hyp))


def compute_batch(
    pairs: Sequence[tuple[str, str]],
    *,
    lowercase: bool = True,
    group_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    if group_by is not None and len(group_by) != len(pairs):
        raise ValueError("group_by must match pairs length")

    details: list[dict[str, Any]] = []
    for index, (reference, hypothesis) in enumerate(pairs):
        entry: dict[str, Any] = {
            "index": index,
            "cer": compute_cer(reference, hypothesis, lowercase=lowercase),
            "wer": compute_wer(reference, hypothesis, lowercase=lowercase),
        }
        if group_by is not None:
            entry["group"] = group_by[index]
        details.append(entry)

    if not details:
        return {"count": 0, "cer": 0.0, "wer": 0.0, "details": [], "groups": {}}

    groups: dict[str, dict[str, Any]] = {}
    if group_by is not None:
        for name in dict.fromkeys(group_by):
            member = [item for item in details if item["group"] == name]
            groups[name] = {
                "count": len(member),
                "cer": sum(item["cer"] for item in member) / len(member),
                "wer": sum(item["wer"] for item in member) / len(member),
            }

    return {
        "count": len(details),
        "cer": sum(item["cer"] for item in details) / len(details),
        "wer": sum(item["wer"] for item in details) / len(details),
        "details": details,
        "groups": groups,
    }
