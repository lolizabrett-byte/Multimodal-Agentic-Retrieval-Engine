"""Đo phân bố chỉ số cổng lọc OCR trên keyframe thật, đề xuất ngưỡng.

Ngưỡng mặc định (`max_no_text_edge_density: 0.0015`) đặt thấp hơn dữ liệu thật
25–100 lần, nên cổng gần như không loại được gì. Script này đo trước, đề xuất sau
— không đoán số.

    python gate_calibration.py \
        --keyframes-parquet <v1>/keyframes.parquet <v2>/keyframes.parquet ... \
        --media-root output/<release> \
        --labels ground_truth/labels.jsonl \
        --out ground_truth/gate-distribution.json

`--labels` là tuỳ chọn nhưng nên có: nó cho biết ảnh nào THẬT SỰ có chữ, nhờ đó
đo được tỉ lệ bỏ sót thay vì chỉ nhìn phân bố.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

MAX_LONG_SIDE = 960
CANNY_LOW = 50
CANNY_HIGH = 150
MISS_BUDGET = 0.02


def gate_features(image_path: Path) -> dict[str, float] | None:
    grayscale = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None or grayscale.size == 0:
        return None
    height, width = grayscale.shape[:2]
    if max(height, width) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / float(max(height, width))
        grayscale = cv2.resize(
            grayscale,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    edges = cv2.Canny(grayscale, CANNY_LOW, CANNY_HIGH)
    regions, boxes = cv2.MSER_create().detectRegions(grayscale)
    image_area = float(grayscale.shape[0] * grayscale.shape[1])
    plausible = 0
    for region, box in zip(regions, boxes, strict=False):
        _, _, region_width, region_height = (int(value) for value in box)
        area = float(region_width * region_height)
        aspect = region_width / float(max(1, region_height))
        if (
            len(region) >= 10
            and 0.00002 <= area / image_area <= 0.08
            and 0.15 <= aspect <= 20.0
        ):
            plausible += 1
            if plausible >= 2:
                break
    return {
        "edge_density": float((edges > 0).mean()),
        "gray_std": float(grayscale.std()),
        "plausible_regions": float(plausible),
    }


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.array(values)
    return {
        f"p{point}": round(float(np.percentile(array, point)), 6)
        for point in (1, 5, 10, 25, 50, 75, 95)
    }


def propose_thresholds(
    measured: list[dict[str, Any]], with_text: list[dict[str, Any]]
) -> dict[str, Any]:
    """Ngưỡng cao nhất mà vẫn giữ tỉ lệ bỏ sót chữ thật dưới MISS_BUDGET."""
    if not with_text:
        return {
            "status": "no_labels",
            "note": "cần --labels để biết ảnh nào thật sự có chữ; không đề xuất ngưỡng nếu chỉ nhìn phân bố",
        }

    text_edges = sorted(item["edge_density"] for item in with_text)
    text_stds = sorted(item["gray_std"] for item in with_text)
    allowed_misses = int(len(with_text) * MISS_BUDGET)

    # Ngưỡng nằm ngay dưới ảnh có chữ "yếu" thứ (allowed_misses+1).
    edge_cut = text_edges[allowed_misses] * 0.95 if allowed_misses < len(text_edges) else 0.0
    std_cut = text_stds[allowed_misses] * 0.95 if allowed_misses < len(text_stds) else 0.0

    skipped = [
        item
        for item in measured
        if item["plausible_regions"] == 0
        and item["edge_density"] <= edge_cut
        and item["gray_std"] <= std_cut
    ]
    missed = [
        item
        for item in with_text
        if item["plausible_regions"] == 0
        and item["edge_density"] <= edge_cut
        and item["gray_std"] <= std_cut
    ]
    miss_rate = len(missed) / len(with_text)

    return {
        "status": "ok" if miss_rate <= MISS_BUDGET else "miss_budget_exceeded",
        "max_no_text_edge_density": round(edge_cut, 6),
        "max_no_text_gray_std": round(std_cut, 3),
        "skip_rate": round(len(skipped) / len(measured), 4),
        "text_miss_rate": round(miss_rate, 4),
        "miss_budget": MISS_BUDGET,
        "note": (
            "áp dụng được"
            if miss_rate <= MISS_BUDGET
            else "KHÔNG áp dụng — mọi ngưỡng đều bỏ sót quá nhiều chữ thật; giữ nguyên config cũ"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes-parquet", nargs="+", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    labelled_text: dict[str, bool] = {}
    if args.labels and args.labels.exists():
        with args.labels.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                labelled_text[str(row.get("image"))] = bool(str(row.get("text", "")).strip())

    measured: list[dict[str, Any]] = []
    videos: set[str] = set()
    for parquet_path in args.keyframes_parquet:
        frame = pd.read_parquet(parquet_path)
        for row in frame.to_dict("records"):
            image_path = args.media_root / str(row.get("keyframe_ref", "")).lstrip("/")
            features = gate_features(image_path)
            if features is None:
                continue
            videos.add(str(row.get("video_id")))
            measured.append(
                {
                    "image": str(image_path),
                    "video_id": str(row.get("video_id")),
                    "has_text": labelled_text.get(str(image_path)),
                    **features,
                }
            )

    if not measured:
        raise SystemExit("no keyframes could be read")

    with_text = [item for item in measured if item["has_text"] is True]
    current_skips = [
        item
        for item in measured
        if item["plausible_regions"] == 0
        and item["edge_density"] <= 0.0015
        and item["gray_std"] <= 12
    ]

    payload = {
        "measured": len(measured),
        "videos": len(videos),
        "labelled_with_text": len(with_text),
        "current_config": {
            "max_no_text_edge_density": 0.0015,
            "max_no_text_gray_std": 12,
            "skip_rate": round(len(current_skips) / len(measured), 4),
        },
        "distribution": {
            "edge_density": percentiles([item["edge_density"] for item in measured]),
            "gray_std": percentiles([item["gray_std"] for item in measured]),
        },
        "proposed": propose_thresholds(measured, with_text),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"đo {len(measured)} keyframe từ {len(videos)} video")
    print(f"config hiện tại loại được: {payload['current_config']['skip_rate']:.1%}")
    proposed = payload["proposed"]
    if proposed.get("status") == "ok":
        print(f"đề xuất: edge<={proposed['max_no_text_edge_density']} "
              f"std<={proposed['max_no_text_gray_std']} "
              f"→ loại {proposed['skip_rate']:.1%}, bỏ sót {proposed['text_miss_rate']:.1%}")
    else:
        print(f"đề xuất: {proposed.get('note')}")


if __name__ == "__main__":
    main()
