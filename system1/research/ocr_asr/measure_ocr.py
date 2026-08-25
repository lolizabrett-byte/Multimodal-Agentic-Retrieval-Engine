"""Đo CER/WER + tốc độ của OCR trên tập nhãn tay.

Dùng cho mọi lần so sánh trong kế hoạch: baseline, sau khi bật tiling, sau khi
đổi tiền xử lý. Cùng một script, khác tham số — để số liệu so được với nhau.

    python measure_ocr.py --labels ground_truth/labels.jsonl \
        --out ground_truth/baseline-1b.json --label baseline-1b
    python measure_ocr.py --labels ground_truth/labels.jsonl \
        --max-num 4 --out ground_truth/tiling-max4.json --label tiling-max4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from system1.metrics import compute_batch


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not str(row.get("text", "")).strip():
                continue
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", required=True, help="tên cấu hình, ghi vào kết quả")
    parser.add_argument("--max-num", type=int, default=None, help="số patch tối đa; None = theo config")
    parser.add_argument("--preprocess", default=None, help="tên kỹ thuật tiền xử lý")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from system1.handoff import extract_image_text

    rows = load_labels(args.labels)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no labelled rows in {args.labels} (did you fill the 'text' field?)")

    pairs: list[tuple[str, str]] = []
    groups: list[str] = []
    durations: list[float] = []
    failures: list[dict[str, Any]] = []

    for row in rows:
        started = time.monotonic()
        try:
            hypothesis = extract_image_text(
                row["image"],
                max_num=args.max_num,
                preprocess=args.preprocess,
            )
        except Exception as exc:  # ghi lại, không bỏ cả lần đo vì một ảnh hỏng
            failures.append({"image": row["image"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        durations.append(time.monotonic() - started)
        pairs.append((row["text"], hypothesis or ""))
        groups.append(str(row.get("difficulty", "unknown")))

    if not pairs:
        raise SystemExit("every image failed; see errors above")

    result = compute_batch(pairs, group_by=groups)
    payload = {
        "label": args.label,
        "max_num": args.max_num,
        "preprocess": args.preprocess,
        "measured": len(pairs),
        "failed": len(failures),
        "seconds_per_image": round(sum(durations) / len(durations), 4),
        "cer": round(result["cer"], 4),
        "wer": round(result["wer"], 4),
        "by_difficulty": {
            name: {"count": stats["count"], "cer": round(stats["cer"], 4), "wer": round(stats["wer"], 4)}
            for name, stats in result["groups"].items()
        },
        "failures": failures[:20],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{args.label}] {len(pairs)} ảnh | CER {payload['cer']} | WER {payload['wer']} "
          f"| {payload['seconds_per_image']}s/ảnh")
    for name, stats in payload["by_difficulty"].items():
        print(f"  {name:8s} n={stats['count']:3d} CER={stats['cer']}")
    if failures:
        print(f"  {len(failures)} ảnh lỗi — xem {args.out}")


if __name__ == "__main__":
    main()
