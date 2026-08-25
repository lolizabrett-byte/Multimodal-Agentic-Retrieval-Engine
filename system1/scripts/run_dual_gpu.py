"""Chạy Phase01 song song trên nhiều GPU bằng cách chia đôi danh sách video.

Kaggle cấp 2x T4 nhưng pipeline chỉ dùng 1. Cả OCR (1.9 GB) lẫn caption (7.4 GB)
ở fp16 cộng lại 9.3 GB — vừa một card, nên mỗi tiến trình chạy trọn pipeline trên
GPU riêng thay vì phải xẻ model.

Script chỉ chia manifest và gọi lại `system1 process-batch`. Không sửa gì trong
pipeline lõi.

    python scripts/run_dual_gpu.py --batch-id batch_001 --worker-id worker_001 \
        --release-id canonical_release_v001 --output output

`--num-gpus 1` chạy y hệt cách cũ (đường lui khi Kaggle chỉ cấp 1 GPU).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path


def detect_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 1


def split_evenly(video_ids: list[str], parts: int) -> list[list[str]]:
    """Chia đều theo số lượng, phần dư rải vào các nhóm đầu."""
    if parts < 1:
        raise ValueError("parts must be positive")
    base, remainder = divmod(len(video_ids), parts)
    chunks: list[list[str]] = []
    cursor = 0
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        chunks.append(video_ids[cursor : cursor + size])
        cursor += size
    return chunks


def split_by_weight(
    video_ids: list[str], weights: dict[str, float], parts: int
) -> list[list[str]]:
    """Chia sao cho tổng thời lượng mỗi phần xấp xỉ nhau.

    Video trong một batch chênh nhau tới 70 lần (36 giây đến 42 phút). Chia theo
    số lượng để một GPU ôm hết video dài — nó chạy thêm nhiều giờ trong khi GPU
    kia đã xong và ngồi không.

    Xếp video dài trước, mỗi lần bỏ vào phần đang nhẹ nhất.
    """
    if parts < 1:
        raise ValueError("parts must be positive")

    buckets: list[list[str]] = [[] for _ in range(parts)]
    totals = [0.0] * parts
    counts = [0] * parts
    ordered = sorted(video_ids, key=lambda vid: (-weights.get(vid, 0.0), vid))
    for video_id in ordered:
        # Đếm số video là tiêu chí phụ: nếu thiếu thời lượng thì mọi trọng số
        # bằng 0 và mọi video sẽ dồn vào nhóm đầu, để GPU kia ngồi không.
        target = min(range(parts), key=lambda index: (totals[index], counts[index]))
        buckets[target].append(video_id)
        totals[target] += weights.get(video_id, 0.0)
        counts[target] += 1
    return buckets


def read_durations(release_dir: Path, video_ids: list[str]) -> dict[str, float]:
    """Đọc thời lượng từ videos.parquet của Phase00. Thiếu thì trả rỗng."""
    table = release_dir / "tables" / "videos.parquet"
    if not table.is_file():
        return {}
    try:
        import pandas as pd

        frame = pd.read_parquet(table)
    except Exception:
        return {}
    if not {"video_id", "duration_seconds"} <= set(frame.columns):
        return {}
    wanted = set(video_ids)
    return {
        str(row.video_id): float(row.duration_seconds)
        for row in frame.itertuples()
        if str(row.video_id) in wanted and row.duration_seconds
    }


def resolve_release_id(output_root: Path, batch_id: str) -> str:
    """Tìm release chứa manifest của batch này.

    process-batch tự resolve release Phase00 mới nhất, nên notebook không có sẵn
    biến release_id trước khi chạy. Đọc từ đĩa thay vì bắt người dùng gõ tay.
    """
    candidates = sorted(
        path.parent.parent
        for path in output_root.glob(f"*/manifests/{batch_id}.txt")
        if path.is_file()
    )
    if not candidates:
        raise SystemExit(
            f"no release under {output_root} contains manifests/{batch_id}.txt "
            "— run the Phase00 restore step first, or pass --release-id"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise SystemExit(f"multiple releases contain {batch_id}: {names} — pass --release-id")
    return candidates[0].name


def read_manifest(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        raise SystemExit(f"batch manifest not found: {manifest_path}")
    return [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stream_output(stream, prefix: str) -> None:
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()
    stream.close()


def launch(
    *,
    gpu_index: int,
    batch_id: str,
    worker_id: str,
    passthrough: list[str],
) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    command = [
        sys.executable,
        "-m",
        "system1.cli",
        "process-batch",
        "--batch-id",
        batch_id,
        "--worker-id",
        worker_id,
        *passthrough,
    ]
    return subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument(
        "--release-id",
        default=None,
        help="bỏ trống để tự tìm release duy nhất trong --output",
    )
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--num-gpus", type=int, default=None)
    args, passthrough = parser.parse_known_args()

    release_id = args.release_id or resolve_release_id(args.output, args.batch_id)

    requested = args.num_gpus if args.num_gpus is not None else detect_gpu_count()
    available = detect_gpu_count()
    if requested > available:
        print(f"[dual-gpu] requested {requested} GPU(s) but only {available} visible; using {available}")
        requested = available
    requested = max(1, requested)

    release_dir = args.output / release_id
    manifest_dir = release_dir / "manifests"
    video_ids = read_manifest(manifest_dir / f"{args.batch_id}.txt")
    print(f"[dual-gpu] {len(video_ids)} video in {args.batch_id}, splitting across {requested} GPU(s)")

    if requested == 1:
        process = launch(
            gpu_index=0,
            batch_id=args.batch_id,
            worker_id=args.worker_id,
            passthrough=[*passthrough, "--release-id-override", release_id, "--output", str(args.output)],
        )
        thread = threading.Thread(target=stream_output, args=(process.stdout, "[gpu0]"))
        thread.start()
        process.wait()
        thread.join()
        return process.returncode

    durations = read_durations(release_dir, video_ids)
    if durations:
        shards = split_by_weight(video_ids, durations, requested)
        for index, shard in enumerate(shards):
            minutes = sum(durations.get(vid, 0.0) for vid in shard) / 60
            print(f"[dual-gpu] gpu{index}: {len(shard)} video, {minutes:.0f} phút nguồn")
    else:
        print("[dual-gpu] no durations in videos.parquet, splitting by count")
        shards = split_evenly(video_ids, requested)

    processes: list[tuple[int, str, subprocess.Popen]] = []
    threads: list[threading.Thread] = []

    for gpu_index, shard in enumerate(shards):
        if not shard:
            print(f"[dual-gpu] gpu{gpu_index} has no videos, skipping")
            continue
        shard_batch_id = f"{args.batch_id}_gpu{gpu_index}"
        shard_worker_id = f"{args.worker_id}_gpu{gpu_index}"
        # Each shard is a normal batch manifest, so process-batch needs no new flags.
        (manifest_dir / f"{shard_batch_id}.txt").write_text(
            "\n".join(shard) + "\n", encoding="utf-8"
        )
        print(f"[dual-gpu] gpu{gpu_index}: {len(shard)} video -> {shard_batch_id}")
        process = launch(
            gpu_index=gpu_index,
            batch_id=shard_batch_id,
            worker_id=shard_worker_id,
            passthrough=[*passthrough, "--release-id-override", release_id, "--output", str(args.output)],
        )
        thread = threading.Thread(target=stream_output, args=(process.stdout, f"[gpu{gpu_index}]"))
        thread.start()
        threads.append(thread)
        processes.append((gpu_index, shard_batch_id, process))

    # One shard dying must not kill the other — a half-finished batch still
    # leaves usable checkpoints for the next session.
    results: list[tuple[int, str, int]] = []
    for gpu_index, shard_batch_id, process in processes:
        process.wait()
        results.append((gpu_index, shard_batch_id, process.returncode))
    for thread in threads:
        thread.join()

    print("\n[dual-gpu] summary")
    failed = 0
    for gpu_index, shard_batch_id, code in results:
        state = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  gpu{gpu_index} {shard_batch_id}: {state}")
        failed += 1 if code != 0 else 0
    if failed:
        print(f"[dual-gpu] {failed}/{len(results)} shard(s) failed; checkpoints from the rest are intact")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
