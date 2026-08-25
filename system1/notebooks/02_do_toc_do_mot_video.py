"""Đo tốc độ pipeline thật trên đúng MỘT video.

Vì sao cần: con số 1,35s/ảnh trong báo cáo ngân sách được đo bằng cách gọi model
từng ảnh một. Pipeline thật gom 4 ảnh mỗi lần gọi (batch_chat), nên số đó nhiều
khả năng bi quan hơn thực tế. Toàn bộ kế hoạch 49 giờ GPU dựng trên nó.

Một lần chạy này trả lời bốn câu cùng lúc:
  1. giây/ảnh thật khi chạy theo lô
  2. tốc độ caption thật (từ trước tới nay mới chỉ là ước lượng)
  3. batch có bị tụt về 1 không (đọc effective_batch_size trong log)
  4. stage scene_summaries có chết không (nó gửi 12 ảnh, client chỉ nhận 1)

Chạy trên Kaggle bằng %run, sau khi các cell môi trường của notebook 01 đã chạy.
Cần sẵn trong biến toàn cục: repo_root, output_root, scratch_dir, batch_id,
worker_id, run_command.

KHÔNG bật --sync: đây là phép đo, không phải lượt chạy chính thức.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

MEASURE_BATCH_SUFFIX = "_do1"

# Đọc từ videos.parquet của canonical_release_v001 ngày 25/08: 873 video,
# 7.840 phút nội dung. Suy ngân sách theo tổng thời lượng chứ không theo số
# video — video ngắn nhất 0,5 phút, dài nhất 45,6 phút.
TOTAL_VIDEOS = 873
TOTAL_CONTENT_MINUTES = 7840.0


def _require(name: str):
    value = globals().get(name)
    if value is None:
        raise RuntimeError(
            f"Thiếu biến {name!r} — chạy các cell môi trường của notebook 01 trước."
        )
    return value


def restore_phase00_only(*, repo_root: Path, output_root: Path, batch_id: str,
                         worker_id: str) -> Path:
    """Tải manifest + bảng Phase00 về mà KHÔNG xử lý video nào.

    Lặp lại đúng phần đầu của run_phase01_pipeline. Chạy trọn process-batch chỉ
    để lấy manifest sẽ khởi động luôn cả 88 video.
    """
    import shutil

    from system1.config import (
        load_configs,
        require_phase01_production_ready,
        resolve_phase01_config,
    )
    from system1.phase01.phase00 import (
        discover_phase00_candidates,
        resolve_phase00_release,
    )
    from system1.phase01.runner import (
        _hf_store,
        _restore_phase00_if_needed,
        _scratch_root,
    )

    config_dir = repo_root / "system1" / "configs"
    configs = load_configs(config_dir)
    release_storage = dict(configs["storage"]["release"])

    discovery_cache = output_root.resolve().parent / ".phase01_hf_cache" / "discovery"
    try:
        store = _hf_store(release_storage, cache_dir=discovery_cache)
        selected = resolve_phase00_release(discover_phase00_candidates(store))
    finally:
        shutil.rmtree(discovery_cache, ignore_errors=True)

    user_settings = {"batch_id": batch_id, "worker_id": worker_id}
    resolved = resolve_phase01_config(
        config_dir, user_settings=user_settings, phase00_release_id=selected.release_id
    )
    require_phase01_production_ready(resolved)

    release_id = str(resolved.payload["runtime"]["release_id"])
    scratch_root = _scratch_root(resolved.payload["storage"], output_root)
    restore_cache = scratch_root / ".hf_cache" / "phase00_restore"
    try:
        _restore_phase00_if_needed(
            output_root=output_root,
            release_id=release_id,
            batch_id=batch_id,
            storage=resolved.payload["storage"]["release"],
            selected_manifest=selected.manifest,
            cache_dir=restore_cache,
        )
    finally:
        shutil.rmtree(restore_cache, ignore_errors=True)

    release_dir = output_root.resolve() / release_id
    print(f"[restore] release={release_id}")
    return release_dir


TARGET_MIN_SEC = 300.0
TARGET_MAX_SEC = 600.0


def pick_representative_video(
    durations: dict[str, float],
    *,
    target_min_sec: float = TARGET_MIN_SEC,
    target_max_sec: float = TARGET_MAX_SEC,
) -> tuple[str, float, str | None]:
    """Video gần giữa khoảng mục tiêu, kèm cảnh báo nếu phải chọn ngoài khoảng.

    Đo trên video ngắn nhất cho số vô dụng: chi phí nạp model (~77s) chia cho một
    video 30 giây làm đơn giá phồng lên nhiều lần.
    """
    if not durations:
        raise ValueError("không có video nào kèm thời lượng để chọn")

    middle = (target_min_sec + target_max_sec) / 2
    in_range = {
        video_id: seconds
        for video_id, seconds in durations.items()
        if target_min_sec <= seconds <= target_max_sec
    }
    if in_range:
        chosen = min(in_range, key=lambda vid: abs(in_range[vid] - middle))
        return chosen, durations[chosen], None

    chosen = min(durations, key=lambda vid: abs(durations[vid] - middle))
    warning = (
        f"không có video nào trong khoảng {target_min_sec / 60:.0f}-"
        f"{target_max_sec / 60:.0f} phút — lấy gần nhất "
        f"({durations[chosen] / 60:.1f} phút); đơn giá sẽ kém đại diện"
    )
    return chosen, durations[chosen], warning


def caption_valid_ratio(status_counts: dict[str, int]) -> float | None:
    """Tỉ lệ shot có caption dùng được. None khi chưa caption shot nào."""
    total = sum(status_counts.values())
    if not total:
        return None
    return status_counts.get("pass", 0) / total


def unit_cost_excluding_load(
    *, stage_seconds: float, load_seconds: float | None, units: int
) -> float | None:
    """Giây mỗi đơn vị sau khi trừ thời gian nạp model.

    Nạp model chỉ trả một lần mỗi chunk, nên để nó trong đơn giá sẽ ước tính thừa
    khi suy ra cho hàng trăm video.
    """
    if units <= 0:
        return None
    net = stage_seconds - (load_seconds or 0.0)
    return max(net, 0.0) / units


def pick_video_to_measure(release_dir: Path, batch_id: str) -> tuple[str, float]:
    import pandas as pd

    manifest = release_dir / "manifests" / f"{batch_id}.txt"
    video_ids = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not video_ids:
        raise RuntimeError(f"Manifest rỗng: {manifest}")

    frame = pd.read_parquet(release_dir / "tables" / "videos.parquet")
    durations = {
        str(row.video_id): float(row.duration_seconds)
        for row in frame.itertuples()
        if str(row.video_id) in set(video_ids) and row.duration_seconds
    }
    if not durations:
        chosen = video_ids[0]
        print(f"[chon] videos.parquet không có duration — lấy video đầu: {chosen}")
        return chosen, 0.0

    chosen, seconds, warning = pick_representative_video(durations)
    print(
        f"[chon] {len(video_ids)} video trong batch, "
        f"chọn {chosen} ({seconds / 60:.1f} phút)"
    )
    if warning:
        print(f"[chon] CẢNH BÁO: {warning}")
    return chosen, seconds


def write_single_video_manifest(release_dir: Path, batch_id: str, video_id: str) -> str:
    """Manifest 1 dòng. process-batch không có cờ giới hạn số video, nhưng nó đọc
    thẳng từ manifest — nên đây là cách cắt phạm vi mà không phải sửa mã."""
    measure_batch_id = f"{batch_id}{MEASURE_BATCH_SUFFIX}"
    path = release_dir / "manifests" / f"{measure_batch_id}.txt"
    path.write_text(video_id + "\n", encoding="utf-8")
    print(f"[manifest] {path.name} -> {video_id}")
    return measure_batch_id


def run_and_capture(*, repo_root: Path, output_root: Path, scratch_dir: Path,
                    release_id: str, measure_batch_id: str, worker_id: str,
                    log_path: Path) -> tuple[int, float]:
    """Chạy process-batch trên 1 video, ghi toàn bộ log ra đĩa.

    Ghi ra đĩa vì log tiến trình là nguồn duy nhất có elapsed_seconds mỗi stage và
    effective_batch_size — Kaggle mất output khi kernel restart.
    """
    command = [
        sys.executable, "-m", "system1.cli", "process-batch",
        "--batch-id", measure_batch_id,
        "--worker-id", f"{worker_id}_do1",
        "--output", str(output_root),
        "--scratch-dir", str(scratch_dir),
        "--release-id-override", release_id,
        "--require-frame-timeline",
        "--no-restore-phase00",
        "--no-validate-remote",
        "--no-sync",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print("[chay]", " ".join(command[2:]))
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            command,
            cwd=repo_root / "system1",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
        code = process.wait()
    return code, time.monotonic() - started


def parse_progress(log_path: Path) -> dict:
    """Rút số đo từ log.

    Sự kiện tiến trình là JSON một dòng. Bắt cả dòng có tiền tố nên dùng tìm kiếm
    thay vì json.loads thẳng.
    """
    stages: dict[str, float] = {}
    batch_events: list[dict] = []
    model_events: list[dict] = []
    counts: dict[str, int] = defaultdict(int)

    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\{.*\}", raw)
        if not match:
            continue
        try:
            event = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        kind = event.get("event")
        if kind == "stage" and event.get("elapsed_seconds") is not None:
            stages[str(event.get("stage"))] = float(event["elapsed_seconds"])
        elif kind == "model":
            model_events.append(event)
        if "effective_batch_size" in event:
            batch_events.append(event)
        for key in ("gate_checked", "gate_no_text", "vintern_processed", "dedup_reused"):
            if key in event:
                counts[key] += int(event[key] or 0)

    return {
        "stages": stages,
        "batch_events": batch_events,
        "model_events": model_events,
        "counts": dict(counts),
    }


def summarise(parsed: dict, *, video_id: str, duration_sec: float,
              wall_seconds: float, exit_code: int) -> dict:
    stages = parsed["stages"]
    counts = parsed["counts"]

    ocr_images = counts.get("vintern_processed", 0)
    ocr_seconds = stages.get("ocr", 0.0)
    sec_per_image = (ocr_seconds / ocr_images) if ocr_images else None

    load_by_stage: dict[str, float] = {}
    for event in parsed["model_events"]:
        if event.get("status") == "loaded" and event.get("load_seconds") is not None:
            load_by_stage[str(event.get("stage"))] = float(event["load_seconds"])

    # Stage caption không phát sự kiện đếm riêng. Sự kiện batch của model có đủ:
    # mỗi shot đúng một request, nên request_count chính là số shot.
    caption_batch = next(
        (
            event
            for event in parsed["model_events"]
            if event.get("stage") == "shot_captions"
            and event.get("status") == "batch_complete"
        ),
        {},
    )
    caption_shots = int(caption_batch.get("request_count") or 0)
    caption_failed = int(caption_batch.get("failed_request_count") or 0)
    caption_ratio = caption_valid_ratio(
        {"pass": caption_shots - caption_failed, "failed": caption_failed}
    )
    ocr_net = unit_cost_excluding_load(
        stage_seconds=ocr_seconds,
        load_seconds=load_by_stage.get("ocr"),
        units=ocr_images,
    )
    caption_net = unit_cost_excluding_load(
        stage_seconds=stages.get("shot_captions", 0.0),
        load_seconds=load_by_stage.get("shot_captions"),
        units=caption_shots,
    )
    shots_per_minute = (
        (caption_shots / (duration_sec / 60)) if duration_sec and caption_shots else None
    )

    # Mỗi stage có batch size riêng (ocr=4, shot_captions=2). Gộp chung rồi so
    # min với max sẽ báo "tụt xuống" cho một hệ thống hoàn toàn bình thường.
    by_stage: dict[str, dict[str, set[int]]] = {}
    for event in parsed["batch_events"]:
        stage = str(event.get("stage") or "?")
        slot = by_stage.setdefault(stage, {"requested": set(), "effective": set()})
        for key in ("requested", "effective"):
            value = event.get(f"{key}_batch_size")
            if value is not None:
                slot[key].add(int(value))

    print("\n" + "=" * 62)
    print(f"KET QUA DO — {video_id}")
    print("=" * 62)
    print(f"exit code        : {exit_code} {'(OK)' if exit_code == 0 else '(THAT BAI)'}")
    if duration_sec:
        print(f"do dai video     : {duration_sec / 60:.1f} phut")
    print(f"tong thoi gian   : {wall_seconds / 60:.1f} phut")

    print("\n-- thoi gian tung stage --")
    for name, seconds in sorted(stages.items(), key=lambda kv: -kv[1]):
        share = (seconds / wall_seconds * 100) if wall_seconds else 0
        print(f"  {name:24} {seconds / 60:7.2f} phut  ({share:4.1f}%)")

    print("\n-- OCR --")
    print(f"  anh qua Vintern      : {ocr_images}")
    print(f"  cong loc kiem tra    : {counts.get('gate_checked', 0)}")
    print(f"  cong loc bo qua      : {counts.get('gate_no_text', 0)}")
    if sec_per_image is not None:
        print(f"  giay/anh THAT        : {sec_per_image:.3f}s")
        print("  (so cu do le         : 1.350s)")
        if sec_per_image > 0:
            print(f"  nhanh hon            : {1.35 / sec_per_image:.2f} lan")

    print("\n-- caption --")
    print(f"  so shot              : {caption_shots}")
    if caption_ratio is not None:
        print(f"  ti le hop le         : {caption_ratio * 100:.1f}%  ({caption_failed} loi)")
    if caption_net is not None:
        print(f"  giay/shot (tru nap)  : {caption_net:.2f}s")

    print("\n-- don gia da tru chi phi nap model --")
    for stage, seconds in sorted(load_by_stage.items()):
        print(f"  nap {stage:16} {seconds:6.1f}s")
    if ocr_net is not None:
        print(f"  OCR     : {ocr_net:.3f} s/anh   (tho: {sec_per_image:.3f})")
    if caption_net is not None:
        print(f"  caption : {caption_net:.2f} s/shot")
    if shots_per_minute is not None:
        print(f"  mat do  : {shots_per_minute:.1f} shot/phut")

    print("\n-- batch (tung stage) --")
    if by_stage:
        for stage, slot in sorted(by_stage.items()):
            want = sorted(slot["requested"])
            got = sorted(slot["effective"])
            tut = got and want and min(got) < max(want)
            note = "TUT XUONG — xem OOM / batch_capability_fallback" if tut else "giu nguyen"
            print(f"  {stage:16} requested={want} effective={got}  {note}")
    else:
        print("  KHONG thay su kien batch nao trong log")

    samples = []
    for event in parsed["model_events"]:
        samples.extend(event.get("failure_samples") or [])
    if samples:
        print("\n-- model tra ve gi khi loi --")
        for sample in samples[:3]:
            print(f"  [{sample.get('reason')}] dai {sample.get('raw_text_length')} ky tu")
            print(f"    {str(sample.get('raw_text'))[:300]}")

    scene_ok = "scene_summaries" in stages
    print("\n-- scene_summaries (nghi ngo bug 12 anh) --")
    print(f"  stage chay xong : {'CO' if scene_ok else 'KHONG'}")
    if not scene_ok:
        print("  -> stage khong hoan tat. Doc log tim 'expects exactly one image'.")

    # Suy ra ngân sách theo TỔNG THỜI LƯỢNG, không theo số video: video dài ngắn
    # chênh nhau 90 lần (0,5 đến 45,6 phút) nên nhân số video ra số vô nghĩa.
    projected = None
    if exit_code == 0 and wall_seconds > 0 and duration_sec:
        minutes_per_minute = (wall_seconds / 60) / (duration_sec / 60)
        projected = TOTAL_CONTENT_MINUTES * minutes_per_minute / 60 / 2
        print("\n-- ngan sach suy ra --")
        print(f"  video nay: {duration_sec / 60:.1f} phut -> {wall_seconds / 60:.1f} phut xu ly")
        print(f"  ti le    : {minutes_per_minute:.1f} phut xu ly / phut video")
        print(f"  {TOTAL_VIDEOS} video ({TOTAL_CONTENT_MINUTES:,.0f} phut) / 2 GPU = {projected:.0f} gio")
        print(f"  han muc  : 30 gio/tuan/tai khoan")
        for accounts in (1, 2, 4):
            print(f"     {accounts} tai khoan -> {projected / 30 / accounts:.1f} tuan")

    return {
        "video_id": video_id,
        "duration_seconds": duration_sec,
        "wall_seconds": wall_seconds,
        "exit_code": exit_code,
        "stages": stages,
        "counts": counts,
        "ocr_seconds_per_image": sec_per_image,
        "batch_by_stage": {
            stage: {
                "requested": sorted(slot["requested"]),
                "effective": sorted(slot["effective"]),
            }
            for stage, slot in sorted(by_stage.items())
        },
        "scene_summaries_completed": scene_ok,
        "caption_shots": caption_shots,
        "caption_failed": caption_failed,
        "caption_valid_ratio": caption_ratio,
        "model_load_seconds": load_by_stage,
        "ocr_seconds_per_image_net": ocr_net,
        "caption_seconds_per_shot_net": caption_net,
        "shots_per_minute": shots_per_minute,
        "failure_samples": samples[:3],
        "projected_hours_all_videos_2gpu": projected,
    }


def main() -> dict:
    repo_root = Path(_require("repo_root"))
    output_root = Path(_require("output_root"))
    scratch_dir = Path(_require("scratch_dir"))
    batch_id = str(_require("batch_id"))
    worker_id = str(_require("worker_id"))

    release_dir = restore_phase00_only(
        repo_root=repo_root,
        output_root=output_root,
        batch_id=batch_id,
        worker_id=worker_id,
    )
    video_id, duration_sec = pick_video_to_measure(release_dir, batch_id)
    measure_batch_id = write_single_video_manifest(release_dir, batch_id, video_id)

    log_path = output_root / f"do-toc-do-{measure_batch_id}.log"
    exit_code, wall_seconds = run_and_capture(
        repo_root=repo_root,
        output_root=output_root,
        scratch_dir=scratch_dir,
        release_id=release_dir.name,
        measure_batch_id=measure_batch_id,
        worker_id=worker_id,
        log_path=log_path,
    )

    parsed = parse_progress(log_path)
    result = summarise(
        parsed,
        video_id=video_id,
        duration_sec=duration_sec,
        wall_seconds=wall_seconds,
        exit_code=exit_code,
    )

    # Lưu ngay: Kaggle xoá sạch khi kernel restart.
    result_path = output_root / f"ket-qua-do-{measure_batch_id}.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[luu] {result_path}")
    print(f"[luu] {log_path}")
    return result


if __name__ == "__main__":
    main()
