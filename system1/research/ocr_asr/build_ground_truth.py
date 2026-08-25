"""Chọn mẫu keyframe phân tầng theo độ khó để gán nhãn OCR thủ công.

Chạy trên Kaggle sau khi Phase01 đã sinh keyframe cho vài video:

    python build_ground_truth.py \
        --keyframes-parquet output/<release>/artifacts/structure/<video>/keyframes.parquet \
        --media-root output/<release> \
        --out research/ocr_asr/ground_truth

Sinh sample.jsonl (ảnh + nhãn rỗng) để người gán nhãn điền trường `text`.
Phân tầng dựa trên độ tương phản và mật độ cạnh — hai chỉ số rẻ, tương quan với
độ khó đọc chữ. Không thay cho mắt người, chỉ để mẫu trải đều thay vì dồn vào
ảnh dễ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

DIFFICULTY_ORDER = ("easy", "medium", "hard")


def difficulty_features(image_path: Path) -> dict[str, float] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    long_side = max(image.shape[:2])
    if long_side > 960:
        scale = 960 / long_side
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150)
    return {
        "edge_density": float((edges > 0).mean()),
        "gray_std": float(grayscale.std()),
        "laplacian_var": float(cv2.Laplacian(grayscale, cv2.CV_64F).var()),
    }


def assign_difficulty(features: dict[str, float], thresholds: dict[str, tuple[float, float]]) -> str:
    blur_low, blur_high = thresholds["laplacian_var"]
    contrast_low, contrast_high = thresholds["gray_std"]
    if features["laplacian_var"] <= blur_low or features["gray_std"] <= contrast_low:
        return "hard"
    if features["laplacian_var"] >= blur_high and features["gray_std"] >= contrast_high:
        return "easy"
    return "medium"


def resolve_media_path(media_root: Path, keyframe_ref: str) -> Path:
    reference = str(keyframe_ref).lstrip("/")
    return media_root / reference


def build_samples(
    keyframe_frames: list[pd.DataFrame],
    media_root: Path,
    per_bucket: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for frame in keyframe_frames:
        for row in frame.to_dict("records"):
            path = resolve_media_path(media_root, row.get("keyframe_ref", ""))
            features = difficulty_features(path)
            if features is None:
                continue
            scored.append(
                {
                    "keyframe_id": row.get("keyframe_id"),
                    "video_id": row.get("video_id"),
                    "image": str(path),
                    **features,
                }
            )

    if not scored:
        return []

    blur_values = np.array([item["laplacian_var"] for item in scored])
    contrast_values = np.array([item["gray_std"] for item in scored])
    thresholds = {
        "laplacian_var": (float(np.percentile(blur_values, 33)), float(np.percentile(blur_values, 67))),
        "gray_std": (float(np.percentile(contrast_values, 33)), float(np.percentile(contrast_values, 67))),
    }

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in DIFFICULTY_ORDER}
    for item in scored:
        item["difficulty"] = assign_difficulty(item, thresholds)
        buckets[item["difficulty"]].append(item)

    samples: list[dict[str, Any]] = []
    for name in DIFFICULTY_ORDER:
        members = buckets[name]
        if not members:
            continue
        # Trải đều theo video thay vì lấy liên tiếp, tránh dồn mẫu vào một cảnh.
        members.sort(key=lambda entry: (str(entry["video_id"]), str(entry["keyframe_id"])))
        step = max(1, len(members) // per_bucket)
        picked = members[::step][:per_bucket]
        samples.extend(picked)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes-parquet", nargs="+", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-bucket", type=int, default=50)
    args = parser.parse_args()

    frames = [pd.read_parquet(path) for path in args.keyframes_parquet]
    samples = build_samples(frames, args.media_root, args.per_bucket)

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "sample.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(
                json.dumps(
                    {
                        "keyframe_id": item["keyframe_id"],
                        "video_id": item["video_id"],
                        "image": item["image"],
                        "difficulty": item["difficulty"],
                        "text": "",
                        "edge_density": round(item["edge_density"], 5),
                        "gray_std": round(item["gray_std"], 2),
                        "laplacian_var": round(item["laplacian_var"], 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    counts = {name: sum(1 for item in samples if item["difficulty"] == name) for name in DIFFICULTY_ORDER}
    print(f"wrote {len(samples)} samples to {target}")
    print(f"per difficulty: {counts}")
    print('next: fill the "text" field for every row, then rename to labels.jsonl')


if __name__ == "__main__":
    main()
