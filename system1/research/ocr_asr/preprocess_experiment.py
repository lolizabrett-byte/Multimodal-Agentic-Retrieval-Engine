"""So CER giữa OCR gốc và OCR sau từng kỹ thuật tiền xử lý OpenCV.

Bản phân công yêu cầu thử threshold / morphology / grayscale. Những kỹ thuật này
sinh ra cho OCR đời cũ (Tesseract, PaddleOCR). Vintern là VLM học trên ảnh màu tự
nhiên, nên chúng có thể làm HẠI. Script này đo để biết, thay vì đoán.

    python preprocess_experiment.py --labels ground_truth/labels.jsonl \
        --out ground_truth/preprocess-comparison.json

Kỹ thuật nào làm CER tăng thì ghi vào report kèm số — đó là kết quả có giá trị,
không phải thất bại.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from system1.handoff import extract_image_text
from system1.handoff.preprocess import RISKY, SAFER
from system1.metrics import compute_batch

DEFAULT_VARIANTS = ("none", *SAFER, *RISKY, "clahe+sharpen")


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("text", "")).strip():
                rows.append(row)
    return rows


def measure(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    preprocess = None if variant == "none" else variant
    pairs: list[tuple[str, str]] = []
    groups: list[str] = []
    durations: list[float] = []
    failures = 0

    for row in rows:
        started = time.monotonic()
        try:
            hypothesis = extract_image_text(row["image"], preprocess=preprocess)
        except Exception:
            failures += 1
            continue
        durations.append(time.monotonic() - started)
        pairs.append((row["text"], hypothesis or ""))
        groups.append(str(row.get("difficulty", "unknown")))

    if not pairs:
        return {"variant": variant, "status": "all_failed", "failures": failures}

    result = compute_batch(pairs, group_by=groups)
    return {
        "variant": variant,
        "status": "ok",
        "measured": len(pairs),
        "failures": failures,
        "cer": round(result["cer"], 4),
        "wer": round(result["wer"], 4),
        "seconds_per_image": round(sum(durations) / len(durations), 4),
        "by_difficulty": {
            name: round(stats["cer"], 4) for name, stats in result["groups"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_labels(args.labels)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no labelled rows in {args.labels}")

    results = [measure(rows, variant) for variant in args.variants]
    baseline = next((item for item in results if item["variant"] == "none"), None)

    for item in results:
        if baseline and item.get("status") == "ok" and baseline.get("status") == "ok":
            item["cer_delta_vs_none"] = round(item["cer"] - baseline["cer"], 4)
            item["verdict"] = "helps" if item["cer_delta_vs_none"] < 0 else "hurts"

    payload = {
        "labels": str(args.labels),
        "images": len(rows),
        "results": results,
        "recommended": sorted(
            (item["variant"] for item in results if item.get("verdict") == "helps"),
            key=lambda name: next(
                entry["cer"] for entry in results if entry["variant"] == name
            ),
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(rows)} ảnh\n")
    print(f"{'kỹ thuật':22s} {'CER':>8s} {'Δ vs gốc':>10s} {'s/ảnh':>8s}")
    for item in results:
        if item.get("status") != "ok":
            print(f"{item['variant']:22s} {'lỗi hết':>8s}")
            continue
        delta = item.get("cer_delta_vs_none")
        delta_text = f"{delta:+.4f}" if delta is not None else "-"
        print(f"{item['variant']:22s} {item['cer']:8.4f} {delta_text:>10s} {item['seconds_per_image']:8.3f}")
    print(f"\nnên dùng: {payload['recommended'] or 'không kỹ thuật nào cải thiện CER'}")


if __name__ == "__main__":
    main()
