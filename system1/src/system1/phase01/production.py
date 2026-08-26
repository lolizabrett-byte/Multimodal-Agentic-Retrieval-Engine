from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import tempfile
import time
import weakref
import zipfile
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema.exceptions import ValidationError

from system1.artifacts.checkpoint import sha256_file
from system1.artifacts.hf_store import HuggingFaceDatasetArtifactStore
from system1.artifacts.package import validate_artifact_zip, write_artifact_zip
from system1.artifacts.reports import utc_now, write_worker_report
from system1.artifacts.store import ArtifactStore
from system1.asr import build_shot_transcript_links, transcribe_video
from system1.config import ResolvedPhase01Config, persist_resolved_phase01_config
from system1.gemini import GeminiStructuredClient, StructuredRequest
from system1.ingest.discovery import read_metadata
from system1.keyframes import (
    candidate_frame_ids_for_shot,
    iter_decode_frame_groups,
    select_keyframes_for_shot,
    write_keyframe_images,
)
from system1.media.contact_sheet import write_contact_sheet
from system1.phase01.checkpoint import CheckpointManager, compute_fingerprint
from system1.phase01.qa import write_manual_review_report
from system1.phase01.scheduler import plan_runtime_chunks
from system1.phase01.validation import validate_phase01_package, validate_rows
from system1.scenes import group_scenes
from system1.scenes.gemini_judge import StructuredSceneBoundaryJudge
from system1.shots import (
    detect_shot_scenes,
    load_transnet_artifact,
    scenes_to_shot_rows,
)
from system1.vlm import (
    BatchRequestError,
    FallbackStructuredClient,
    LocalVisionStructuredClient,
    MetadataStructuredClient,
    SystemicProviderError,
)

PARQUET_COLUMNS: dict[str, list[str]] = {
    "asr_segments": [
        "asr_segment_id", "video_id", "start_sec", "end_sec", "start_frame", "end_frame",
        "text", "language", "confidence", "avg_logprob", "no_speech_prob", "provider",
        "model_name", "model_version", "status",
    ],
    "shot_transcript_links": ["video_id", "shot_id", "asr_segment_id", "coverage"],
    "scene_transcript_links": ["video_id", "scene_id", "asr_segment_id", "coverage"],
    "ocr": [
        "ocr_id", "video_id", "keyframe_id", "shot_id", "frame_id", "text", "raw_text",
        "provider", "model_name", "model_version", "language", "confidence", "status",
    ],
}


@dataclass
class _VideoFlow:
    video_id: str
    video_index: int
    scratch: Path
    manager: CheckpointManager
    pipeline: Generator[str, Any, dict[str, Any]]


_MANAGER_RUNTIME_CONTEXT: weakref.WeakKeyDictionary[
    CheckpointManager, dict[str, int]
] = weakref.WeakKeyDictionary()
_STAGE_TIMERS: dict[tuple[int, str], float] = {}


def process_production_batch(
    *,
    release_dir: Path,
    config: ResolvedPhase01Config,
    scratch_root: Path,
    transnet_artifact_dir: Path,
    sync_release: bool = True,
) -> Path:
    runtime = config.payload["runtime"]
    release_id = str(runtime["release_id"])
    batch_id = str(runtime["batch_id"])
    worker_id = str(runtime["worker_id"])
    started_at = utc_now()
    batch_path = release_dir / "manifests" / f"{batch_id}.txt"
    video_ids = [line.strip() for line in batch_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicate_video_ids = sorted(
        video_id for video_id in set(video_ids) if video_ids.count(video_id) > 1
    )
    if duplicate_video_ids:
        raise ValueError(
            "Phase01 batch manifest contains duplicate video IDs: "
            + ", ".join(duplicate_video_ids)
        )
    videos = pd.read_parquet(release_dir / "tables" / "videos.parquet")
    media = pd.read_parquet(release_dir / "raw_mapping" / "media_store_manifest.parquet")
    videos_by_id = {str(row["video_id"]): row for row in videos.to_dict("records")}
    media_by_id = {str(row["video_id"]): row for row in media.to_dict("records")}
    result_slots: list[dict[str, Any] | None] = [None] * len(video_ids)
    scratch_root.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        event="batch",
        status="start",
        scratch=scratch_root,
        release_id=release_id,
        batch_id=batch_id,
        video_count=len(video_ids),
    )

    raw_bytes_by_video = {
        video_id: _mapping_raw_bytes(mapping)
        for video_id, mapping in media_by_id.items()
    }
    pending = list(video_ids)
    video_offsets = {video_id: index for index, video_id in enumerate(video_ids)}
    chunk_index = 0
    scheduler_policy = config.payload["phase01"]["execution"]["chunk_scheduler"]
    while pending:
        planned = plan_runtime_chunks(
            pending,
            raw_bytes_by_video=raw_bytes_by_video,
            free_disk_gb=_scratch_free_gb(scratch_root),
            policy=scheduler_policy,
        )[0]
        chunk_index += 1
        chunk_video_ids = list(planned.video_ids)
        pending = pending[len(chunk_video_ids) :]
        chunk_size = len(chunk_video_ids)
        chunk_scratch = (
            scratch_root
            / release_id
            / batch_id
            / ".runtime_chunks"
            / f"chunk_{chunk_index:04d}"
        )
        shutil.rmtree(chunk_scratch, ignore_errors=True)
        chunk_scratch.mkdir(parents=True)
        _emit_progress(
            event="chunk",
            status="start",
            scratch=scratch_root,
            release_id=release_id,
            batch_id=batch_id,
            chunk_index=chunk_index,
            chunk_size=chunk_size,
            chunk_raw_bytes=planned.raw_bytes,
        )

        active: list[_VideoFlow] = []
        for video_id in chunk_video_ids:
            video_index = video_offsets[video_id] + 1
            video_scratch = scratch_root / release_id / batch_id / video_id
            shutil.rmtree(video_scratch, ignore_errors=True)
            video_scratch.mkdir(parents=True)
            checkpoint_store = _hf_store(
                config.payload["storage"]["checkpoint"],
                cache_dir=video_scratch / "hf_cache" / "checkpoint",
            )
            release_store = _hf_store(
                config.payload["storage"]["release"],
                cache_dir=video_scratch / "hf_cache" / "release",
            )
            manager = CheckpointManager(
                checkpoint_store,
                release_id=release_id,
                video_id=video_id,
                config_hash=config.config_hash,
                stage_config_hashes=config.stage_config_hashes,
                verify_remote_checksum=bool(
                    config.payload["storage"]["checkpoint"].get(
                        "verify_remote_checksum", True
                    )
                ),
                root_template=str(
                    config.payload["artifact"]["checkpoint"]["root"]
                ),
                state_filename=str(
                    config.payload["artifact"]["checkpoint"]["state_filename"]
                ),
            )
            _MANAGER_RUNTIME_CONTEXT[manager] = {
                "chunk_index": chunk_index,
                "chunk_size": chunk_size,
            }
            _emit_progress(
                event="video",
                status="start",
                scratch=scratch_root,
                release_id=release_id,
                batch_id=batch_id,
                video_id=video_id,
                video_index=video_index,
                video_count=len(video_ids),
                chunk_index=chunk_index,
                chunk_size=chunk_size,
            )
            flow: _VideoFlow | None = None
            try:
                if video_id not in videos_by_id or video_id not in media_by_id:
                    raise ValueError(
                        f"Phase00 batch references unknown video_id={video_id}"
                    )
                pipeline = _process_video_flow(
                    video_id=video_id,
                    video_row=videos_by_id[video_id],
                    mapping=media_by_id[video_id],
                    release_dir=release_dir,
                    scratch=video_scratch,
                    manager=manager,
                    config=config,
                    transnet_artifact_dir=transnet_artifact_dir,
                    release_store=release_store,
                    sync_release=sync_release,
                )
                flow = _VideoFlow(
                    video_id=video_id,
                    video_index=video_index,
                    scratch=video_scratch,
                    manager=manager,
                    pipeline=pipeline,
                )
                yielded = next(pipeline)
                if yielded != "ocr":
                    raise RuntimeError(
                        f"Phase01 video flow expected ocr, received {yielded!r}"
                    )
                active.append(flow)
            except Exception as exc:  # noqa: BLE001 - isolate failures per video
                if flow is None:
                    flow = _VideoFlow(
                        video_id=video_id,
                        video_index=video_index,
                        scratch=video_scratch,
                        manager=manager,
                        pipeline=_empty_video_flow(),
                    )
                _finish_failed_video(
                    flow,
                    exc,
                    result_slots=result_slots,
                    scratch_root=scratch_root,
                    release_id=release_id,
                    batch_id=batch_id,
                    video_count=len(video_ids),
                )

        active = _run_chunk_model_stage(
            active,
            model_config=config.payload["models"]["ocr"],
            phase01=config.payload["phase01"],
            cache=ArtifactStore(chunk_scratch / "ocr_api_cache"),
            cache_prefix="ocr",
            expected_yield="shot_captions",
            caption_chain=False,
            result_slots=result_slots,
            scratch_root=scratch_root,
            release_id=release_id,
            batch_id=batch_id,
            video_count=len(video_ids),
            chunk_index=chunk_index,
            chunk_size=chunk_size,
        )
        active = _run_chunk_model_stage(
            active,
            model_config=config.payload["models"]["shot_caption"],
            phase01=config.payload["phase01"],
            cache=ArtifactStore(chunk_scratch / "caption_api_cache"),
            cache_prefix="shot_caption",
            expected_yield="finalize",
            caption_chain=True,
            result_slots=result_slots,
            scratch_root=scratch_root,
            release_id=release_id,
            batch_id=batch_id,
            video_count=len(video_ids),
            chunk_index=chunk_index,
            chunk_size=chunk_size,
        )
        for flow in active:
            try:
                yielded = next(flow.pipeline)
            except StopIteration as completed:
                result = completed.value
                if not isinstance(result, dict):
                    _finish_failed_video(
                        flow,
                        RuntimeError("Phase01 video flow returned an invalid result"),
                        result_slots=result_slots,
                        scratch_root=scratch_root,
                        release_id=release_id,
                        batch_id=batch_id,
                        video_count=len(video_ids),
                    )
                    continue
                _finish_video(
                    flow,
                    result,
                    result_slots=result_slots,
                    scratch_root=scratch_root,
                    release_id=release_id,
                    batch_id=batch_id,
                    video_count=len(video_ids),
                )
            except Exception as exc:  # noqa: BLE001 - isolate failures per video
                _finish_failed_video(
                    flow,
                    exc,
                    result_slots=result_slots,
                    scratch_root=scratch_root,
                    release_id=release_id,
                    batch_id=batch_id,
                    video_count=len(video_ids),
                )
            else:
                _finish_failed_video(
                    flow,
                    RuntimeError(
                        "Phase01 video flow yielded unexpectedly during finalize: "
                        f"{yielded!r}"
                    ),
                    result_slots=result_slots,
                    scratch_root=scratch_root,
                    release_id=release_id,
                    batch_id=batch_id,
                    video_count=len(video_ids),
                )
        shutil.rmtree(chunk_scratch, ignore_errors=True)
        _emit_progress(
            event="chunk",
            status="complete",
            scratch=scratch_root,
            release_id=release_id,
            batch_id=batch_id,
            chunk_index=chunk_index,
            chunk_size=chunk_size,
            chunk_raw_bytes=planned.raw_bytes,
        )

    shutil.rmtree(
        scratch_root / release_id / batch_id / ".runtime_chunks",
        ignore_errors=True,
    )
    if any(result is None for result in result_slots):
        raise RuntimeError("Phase01 scheduler finished without a result for every video")
    results = [result for result in result_slots if result is not None]

    failed = [row for row in results if not row["status"].startswith("complete")]
    manual_review_path = write_manual_review_report(
        release_dir=release_dir,
        batch_id=batch_id,
        worker_id=worker_id,
        video_results=results,
        sample_size=int(config.payload["phase01"]["manual_review"]["sample_size"]),
    )
    report = write_worker_report(
        release_dir,
        phase="structure",
        batch_id=batch_id,
        worker_id=worker_id,
        started_at=started_at,
        finished_at=utc_now(),
        videos_processed=len(results),
        videos_failed=len(failed),
        payload={
            "schema_version": "phase01_worker_report_v2",
            "release_id": release_id,
            "config_hash": config.config_hash,
            "manual_review": {
                "status": "pending_manual_review",
                "path": str(manual_review_path),
            },
            "counts": {
                "complete": sum(row["status"] == "complete" for row in results),
                "complete_local": sum(row["status"] == "complete_local" for row in results),
                "failed_retryable": sum(row["status"] == "failed_retryable" for row in results),
                "failed_terminal": sum(row["status"] == "failed_terminal" for row in results),
            },
            "videos": results,
        },
    )
    errors_path = release_dir / "manifests" / "phase01" / f"errors_{batch_id}_{worker_id}.jsonl"
    _write_jsonl(errors_path, failed)
    if sync_release:
        release_store = _hf_store(config.payload["storage"]["release"])
        remote_root = f"{release_id}/phase01_structure"
        release_store.upload_files(
            [
                (report, f"{remote_root}/worker_reports/{report.name}"),
                (errors_path, f"{remote_root}/worker_reports/{errors_path.name}"),
                (
                    manual_review_path,
                    f"{remote_root}/worker_reports/{manual_review_path.name}",
                ),
            ],
            commit_message=f"Upload Phase01 worker report {batch_id}/{worker_id}",
            num_threads=2,
        )
    _emit_progress(
        event="batch",
        status="complete" if not failed else "failed",
        scratch=scratch_root,
        release_id=release_id,
        batch_id=batch_id,
        video_count=len(video_ids),
        failed_count=len(failed),
    )
    if failed:
        raise RuntimeError(
            f"Phase01 batch completed with {len(failed)} failed video(s); report={report}"
        )
    return report


def _run_chunk_model_stage(
    flows: list[_VideoFlow],
    *,
    model_config: Mapping[str, Any],
    phase01: Mapping[str, Any],
    cache: ArtifactStore,
    cache_prefix: str,
    expected_yield: str,
    caption_chain: bool,
    result_slots: list[dict[str, Any] | None],
    scratch_root: Path,
    release_id: str,
    batch_id: str,
    video_count: int,
    chunk_index: int,
    chunk_size: int,
) -> list[_VideoFlow]:
    if not flows:
        return []

    stage = "shot_captions" if caption_chain else "ocr"
    lifecycle_callback = _model_lifecycle_callback(
        scratch_root=scratch_root,
        release_id=release_id,
        batch_id=batch_id,
        chunk_index=chunk_index,
        chunk_size=chunk_size,
        stage=stage,
    )
    try:
        if caption_chain:
            client = _caption_client_for_model(
                model_config,
                phase01=phase01,
                cache=cache,
                lifecycle_callback=lifecycle_callback,
            )
        else:
            client = _structured_client_for_model(
                model_config,
                phase01=phase01,
                cache=cache,
                cache_prefix=cache_prefix,
                lifecycle_callback=lifecycle_callback,
            )
    except Exception as exc:  # noqa: BLE001 - fail only this chunk stage
        for flow in flows:
            flow.manager.active_stage = stage
            _emit_stage_progress(
                flow.manager, stage, scratch_root, status="start"
            )
            _finish_failed_video(
                flow,
                exc,
                result_slots=result_slots,
                scratch_root=scratch_root,
                release_id=release_id,
                batch_id=batch_id,
                video_count=video_count,
            )
        return []

    survivors: list[_VideoFlow] = []
    try:
        for flow in flows:
            try:
                yielded = flow.pipeline.send(client)
                if yielded != expected_yield:
                    raise RuntimeError(
                        "Phase01 video flow expected "
                        f"{expected_yield}, received {yielded!r}"
                    )
                survivors.append(flow)
            except Exception as exc:  # noqa: BLE001 - isolate failures per video
                _finish_failed_video(
                    flow,
                    exc,
                    result_slots=result_slots,
                    scratch_root=scratch_root,
                    release_id=release_id,
                    batch_id=batch_id,
                    video_count=video_count,
                )
    finally:
        _release_structured_client(client)
    return survivors


def _finish_failed_video(
    flow: _VideoFlow,
    exc: Exception,
    *,
    result_slots: list[dict[str, Any] | None],
    scratch_root: Path,
    release_id: str,
    batch_id: str,
    video_count: int,
) -> None:
    retryable = _retryable_video_error(exc)
    checkpoint_error: str | None = None
    failed_stage = flow.manager.active_stage
    _emit_stage_progress(
        flow.manager,
        failed_stage,
        scratch_root,
        status="failed_retryable" if retryable else "failed_terminal",
    )
    try:
        flow.manager.mark_failed(
            failed_stage,
            input_fingerprint=None,
            retryable=retryable,
            error={
                "error_type": type(exc).__name__,
                "message": str(exc),
                "failed_at": utc_now(),
            },
        )
    except Exception as checkpoint_exc:  # noqa: BLE001 - retain original error
        failed_stage = "unknown"
        checkpoint_error = str(checkpoint_exc)
    result = {
        "video_id": flow.video_id,
        "status": "failed_retryable" if retryable else "failed_terminal",
        "failed_stage": failed_stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "checkpoint_error": checkpoint_error,
    }
    flow.pipeline.close()
    _finish_video(
        flow,
        result,
        result_slots=result_slots,
        scratch_root=scratch_root,
        release_id=release_id,
        batch_id=batch_id,
        video_count=video_count,
    )


def _finish_video(
    flow: _VideoFlow,
    result: dict[str, Any],
    *,
    result_slots: list[dict[str, Any] | None],
    scratch_root: Path,
    release_id: str,
    batch_id: str,
    video_count: int,
) -> None:
    result_slots[flow.video_index - 1] = result
    shutil.rmtree(flow.scratch, ignore_errors=True)
    runtime_context = _MANAGER_RUNTIME_CONTEXT.get(flow.manager, {})
    _emit_progress(
        event="video_cache_cleanup",
        status="complete",
        scratch=scratch_root,
        release_id=release_id,
        batch_id=batch_id,
        video_id=flow.video_id,
        **runtime_context,
    )
    result_progress = {
        key: result[key]
        for key in ("failed_stage", "error_type")
        if result.get(key) is not None
    }
    _emit_progress(
        event="video",
        status=str(result["status"]),
        scratch=scratch_root,
        release_id=release_id,
        batch_id=batch_id,
        video_id=flow.video_id,
        video_index=flow.video_index,
        video_count=video_count,
        **runtime_context,
        **result_progress,
    )
    _MANAGER_RUNTIME_CONTEXT.pop(flow.manager, None)


def _empty_video_flow() -> Generator[str, Any, dict[str, Any]]:
    if False:  # pragma: no cover - typed empty generator
        yield ""
    return {}


def _process_video_flow(
    *,
    video_id: str,
    video_row: dict[str, Any],
    mapping: dict[str, Any],
    release_dir: Path,
    scratch: Path,
    manager: CheckpointManager,
    config: ResolvedPhase01Config,
    transnet_artifact_dir: Path,
    release_store: HuggingFaceDatasetArtifactStore,
    sync_release: bool,
) -> Generator[str, Any, dict[str, Any]]:
    manager.active_stage = "shots"
    _emit_stage_progress(manager, "shots", scratch, status="start")
    stage_dir = scratch / "stages"
    stage_dir.mkdir(exist_ok=True)
    video_path = _materialize_canonical(mapping, "canonical_video_path", scratch / "source")
    metadata_path = _materialize_canonical(mapping, "canonical_metadata_path", scratch / "source")
    timeline_path = _timeline_path(release_dir, video_id, video_row)
    timeline = pd.read_parquet(timeline_path).sort_values("frame_id").to_dict("records")
    if not timeline:
        raise ValueError(f"Phase00 frame timeline is empty for {video_id}")
    video_timeline_fingerprint = compute_fingerprint(
        _stable_video_identity(mapping),
        sha256_file(video_path),
        sha256_file(timeline_path),
        len(timeline),
    )
    metadata_fingerprint = compute_fingerprint(
        _stable_metadata_identity(mapping),
        sha256_file(metadata_path),
    )
    models = config.payload["models"]
    phase01 = config.payload["phase01"]
    scene_boundary_model = _semantic_model_config(models, "scene_boundary")
    scene_summary_model = _semantic_model_config(models, "scene_summary")
    media_config = config.payload["media"]

    shots_path = stage_dir / "shots.parquet"
    shots_fingerprint = compute_fingerprint(
        video_timeline_fingerprint, config.stage_config_hashes["shots"]
    )
    shots_reused = _restore_if_reusable(manager, "shots", shots_fingerprint, stage_dir)
    if not shots_reused:
        artifact = load_transnet_artifact(
            transnet_artifact_dir,
            expected_commit=str(models["shot_detection"]["model_revision"]),
            expected_source_sha256=str(
                models["shot_detection"]["source_sha256"]
            ),
            expected_weights_sha256=str(models["shot_detection"]["weights_sha256"]),
            expected_conversion_verified=bool(
                models["shot_detection"].get("conversion_verified", True)
            ),
        )
        prediction_path = stage_dir / "transnet_predictions.json"
        predictions = detect_shot_scenes(
            video_path,
            artifact=artifact,
            output_path=prediction_path,
            threshold=float(models["shot_detection"]["threshold"]),
            transition_run_boundary=str(
                models["shot_detection"]["transition_run_boundary"]
            ),
            expected_frame_count=len(timeline),
            total_attempts=int(phase01["retry"]["local_model_total_attempts"]),
        )
        shots = scenes_to_shot_rows(
            video_id=video_id,
            scenes_inclusive=predictions["scenes_inclusive"],
            frame_timeline=timeline,
        )
        _write_parquet(shots_path, shots)
        manager.promote_stage(
            "shots",
            input_fingerprint=shots_fingerprint,
            outputs=[shots_path, prediction_path],
            model=models["shot_detection"],
            schema_version=phase01["schemas"]["shots"],
        )
    _emit_stage_progress(
        manager, "shots", scratch, status="complete", reused=shots_reused
    )
    shots = pd.read_parquet(shots_path).to_dict("records")
    shots_output_fingerprint = manager.stage_output_fingerprint("shots")

    manager.active_stage = "keyframes"
    _emit_stage_progress(manager, "keyframes", scratch, status="start")
    keyframes_path = stage_dir / "keyframes.parquet"
    keyframes_bundle = stage_dir / "keyframes.zip"
    keyframes_fingerprint = _stage_fingerprint(
        manager, "keyframes", shots_output_fingerprint
    )
    keyframes_reused = _restore_keyframes_if_reusable(
        manager, keyframes_fingerprint, stage_dir
    )
    if not keyframes_reused:
        _build_keyframes(
            video_id=video_id,
            video_path=video_path,
            shots=shots,
            timeline=timeline,
            output_dir=stage_dir,
            config=media_config,
        )
        _write_directory_zip(stage_dir, keyframes_bundle, ("keyframes.parquet", "keyframes", "thumbnails", "keyframe_diagnostics.jsonl"))
        manager.promote_stage(
            "keyframes",
            input_fingerprint=keyframes_fingerprint,
            outputs=[keyframes_bundle],
            schema_version=phase01["schemas"]["keyframes"],
        )
    _emit_stage_progress(
        manager, "keyframes", scratch, status="complete", reused=keyframes_reused
    )
    keyframes = pd.read_parquet(keyframes_path).to_dict("records")
    keyframes_output_fingerprint = manager.stage_output_fingerprint("keyframes")

    manager.active_stage = "asr"
    _emit_stage_progress(manager, "asr", scratch, status="start")
    asr_path = stage_dir / "asr_segments.parquet"
    asr_status_path = stage_dir / "asr_status.json"
    asr_fingerprint = compute_fingerprint(
        video_timeline_fingerprint, config.stage_config_hashes["asr"]
    )
    asr_reused = _restore_if_reusable(manager, "asr", asr_fingerprint, stage_dir)
    if not asr_reused:
        asr_config = {**models["asr"], "total_attempts": phase01["retry"]["local_model_total_attempts"]}
        result = transcribe_video(
            video_path,
            video_id=video_id,
            frame_timeline=timeline,
            config=asr_config,
        )
        _write_parquet(asr_path, result.rows, empty_columns=PARQUET_COLUMNS["asr_segments"])
        _write_json(asr_status_path, {
            "status": result.status,
            "compute_type": result.compute_type,
            "attempts": result.attempts,
            "detected_language": result.detected_language,
        })
        manager.promote_stage(
            "asr",
            input_fingerprint=asr_fingerprint,
            outputs=[asr_path, asr_status_path],
            model=models["asr"],
            schema_version=phase01["schemas"]["asr_segments"],
        )
    _emit_stage_progress(
        manager, "asr", scratch, status="complete", reused=asr_reused
    )
    asr_rows = pd.read_parquet(asr_path).to_dict("records")
    asr_output_fingerprint = manager.stage_output_fingerprint("asr")

    # The decoded Phase00 timeline can be very large. Release it before this
    # prepared video waits for the other videos in its runtime chunk.
    timeline = []
    ocr_client = yield "ocr"
    manager.active_stage = "ocr"
    _emit_stage_progress(manager, "ocr", scratch, status="start")
    ocr_path = stage_dir / "ocr.parquet"
    ocr_status_path = stage_dir / "ocr_status.json"
    ocr_fingerprint = _stage_fingerprint(manager, "ocr", keyframes_output_fingerprint)
    ocr_reused = _restore_if_reusable(manager, "ocr", ocr_fingerprint, stage_dir)
    if not ocr_reused:
        if ocr_client is None:
            raise RuntimeError("Phase01 OCR stage requires a structured client")
        try:
            ocr_gate_counts: dict[str, int] = {}
            ocr_rows = _build_ocr(
                video_id=video_id,
                keyframes=keyframes,
                stage_dir=stage_dir,
                client=ocr_client,
                model_config=models["ocr"],
                ocr_config=phase01["ocr"],
                diagnostics=ocr_gate_counts,
            )
            _write_parquet(ocr_path, ocr_rows, empty_columns=PARQUET_COLUMNS["ocr"])
            status_counts: dict[str, int] = {}
            for row in ocr_rows:
                status = str(row["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
            ocr_status = "pass"
            if status_counts.get("failed") == len(ocr_rows) and ocr_rows:
                ocr_status = "failed"
            elif status_counts.get("failed"):
                ocr_status = "partial"
            _write_json(ocr_status_path, {
                "status": ocr_status,
                "provider": models["ocr"]["provider"],
                "model_id": models["ocr"]["model_id"],
                "status_counts": status_counts,
                **ocr_gate_counts,
            })
            _emit_progress(
                event="ocr_gate",
                status="complete",
                scratch=scratch,
                release_id=manager.release_id,
                video_id=video_id,
                **_MANAGER_RUNTIME_CONTEXT.get(manager, {}),
                **ocr_gate_counts,
            )
            manager.promote_stage(
                "ocr",
                input_fingerprint=ocr_fingerprint,
                outputs=[ocr_path, ocr_status_path],
                model=models["ocr"],
                prompt_version=models["ocr"]["prompt_version"],
                schema_version=phase01["schemas"]["ocr"],
            )
        finally:
            pass  # The chunk scheduler owns the shared client lifecycle.
    _emit_stage_progress(manager, "ocr", scratch, status="complete", reused=ocr_reused)
    ocr_rows = pd.read_parquet(ocr_path).to_dict("records")
    ocr_output_fingerprint = manager.stage_output_fingerprint("ocr")

    caption_client = yield "shot_captions"
    manager.active_stage = "shot_captions"
    _emit_stage_progress(manager, "shot_captions", scratch, status="start")
    captions_path = stage_dir / "shot_captions.parquet"
    captions_fingerprint = _stage_fingerprint(
        manager, "shot_captions", compute_fingerprint(keyframes_output_fingerprint, ocr_output_fingerprint)
    )
    captions_reused = _restore_if_reusable(
        manager, "shot_captions", captions_fingerprint, stage_dir
    )
    if not captions_reused:
        if caption_client is None:
            raise RuntimeError(
                "Phase01 shot_captions stage requires a structured client"
            )
        try:
            caption_rows = _build_captions(
                video_id=video_id,
                shots=shots,
                keyframes=keyframes,
                ocr_rows=ocr_rows,
                stage_dir=stage_dir,
                client=caption_client,
                model_config=models["shot_caption"],
                max_concurrency=int(phase01["api"]["max_concurrency_per_video"]),
                caption_config=phase01.get("shot_captions", {}),
            )
            _write_parquet(captions_path, caption_rows)
            manager.promote_stage(
                "shot_captions",
                input_fingerprint=captions_fingerprint,
                outputs=[captions_path],
                model=models["shot_caption"],
                prompt_version=models["shot_caption"]["prompt_version"],
                schema_version=phase01["schemas"]["shot_captions"],
            )
        finally:
            pass  # The chunk scheduler owns the shared client lifecycle.
    _emit_stage_progress(
        manager,
        "shot_captions",
        scratch,
        status="complete",
        reused=captions_reused,
    )
    captions = pd.read_parquet(captions_path).to_dict("records")
    captions_output_fingerprint = manager.stage_output_fingerprint("shot_captions")

    manager.active_stage = "shot_transcript_links"
    _emit_stage_progress(manager, "shot_transcript_links", scratch, status="start")
    links_path = stage_dir / "shot_transcript_links.parquet"
    links_fingerprint = compute_fingerprint(
        shots_output_fingerprint,
        asr_output_fingerprint,
        config.stage_config_hashes["shot_transcript_links"],
    )
    links_reused = _restore_if_reusable(
        manager, "shot_transcript_links", links_fingerprint, stage_dir
    )
    if not links_reused:
        links = build_shot_transcript_links(shots, asr_rows)
        _write_parquet(links_path, links, empty_columns=PARQUET_COLUMNS["shot_transcript_links"])
        manager.promote_stage(
            "shot_transcript_links",
            input_fingerprint=links_fingerprint,
            outputs=[links_path],
            schema_version=phase01["schemas"]["shot_transcript_links"],
        )
    _emit_stage_progress(
        manager,
        "shot_transcript_links",
        scratch,
        status="complete",
        reused=links_reused,
    )
    links = pd.read_parquet(links_path).to_dict("records")
    links_output_fingerprint = manager.stage_output_fingerprint(
        "shot_transcript_links"
    )

    manager.active_stage = "scenes"
    _emit_stage_progress(manager, "scenes", scratch, status="start")
    scenes_path = stage_dir / "scenes.parquet"
    scene_links_path = stage_dir / "scene_transcript_links.parquet"
    scene_diagnostics_path = stage_dir / "scene_boundary_diagnostics.jsonl"
    scenes_fingerprint = compute_fingerprint(
        shots_output_fingerprint,
        keyframes_output_fingerprint,
        ocr_output_fingerprint,
        captions_output_fingerprint,
        asr_output_fingerprint,
        links_output_fingerprint,
        config.stage_config_hashes["scenes"],
    )
    scenes_reused = _restore_if_reusable(
        manager, "scenes", scenes_fingerprint, stage_dir
    )
    if not scenes_reused:
        evidence = _build_scene_evidence(shots, keyframes, ocr_rows, captions, asr_rows, links, stage_dir)
        judge = StructuredSceneBoundaryJudge(
            caption_client,
            video_id=video_id,
            prompt_dir=_prompt_dir(),
            diagnostics_dir=stage_dir / "diagnostics" / "scene_requests",
            model_config=scene_boundary_model,
        )
        scenes, decisions = group_scenes(
            video_id=video_id,
            shots=shots,
            evidence=evidence,
            judge=judge,
            config=phase01["scene_grouping"],
        )
        _write_parquet(scenes_path, scenes)
        scene_links = _build_scene_transcript_links(scenes, asr_rows)
        _write_parquet(
            scene_links_path,
            scene_links,
            empty_columns=PARQUET_COLUMNS["scene_transcript_links"],
        )
        _write_jsonl(scene_diagnostics_path, [decision.__dict__ for decision in decisions])
        manager.promote_stage(
            "scenes",
            input_fingerprint=scenes_fingerprint,
            outputs=[scenes_path, scene_links_path, scene_diagnostics_path],
            model=scene_boundary_model,
            prompt_version=scene_boundary_model["prompt_version"],
            schema_version=phase01["schemas"]["scenes"],
        )
    _emit_stage_progress(
        manager, "scenes", scratch, status="complete", reused=scenes_reused
    )
    scenes = pd.read_parquet(scenes_path).to_dict("records")
    scene_links = pd.read_parquet(scene_links_path).to_dict("records")
    scenes_output_fingerprint = manager.stage_output_fingerprint("scenes")

    manager.active_stage = "scene_summaries"
    _emit_stage_progress(manager, "scene_summaries", scratch, status="start")
    summaries_path = stage_dir / "scene_summaries.parquet"
    summaries_fingerprint = compute_fingerprint(
        scenes_output_fingerprint,
        keyframes_output_fingerprint,
        ocr_output_fingerprint,
        asr_output_fingerprint,
        captions_output_fingerprint,
        links_output_fingerprint,
        config.stage_config_hashes["scene_summaries"],
    )
    summaries_reused = _restore_if_reusable(
        manager, "scene_summaries", summaries_fingerprint, stage_dir
    )
    if not summaries_reused:
        summary_rows = _build_scene_summaries(
            video_id=video_id,
            scenes=scenes,
            shots=shots,
            keyframes=keyframes,
            ocr_rows=ocr_rows,
            captions=captions,
            asr_rows=asr_rows,
            scene_links=scene_links,
            stage_dir=stage_dir,
            client=caption_client,
            model_config=scene_summary_model,
            summary_config=phase01["scene_summary"],
        )
        _write_parquet(summaries_path, summary_rows)
        manager.promote_stage(
            "scene_summaries",
            input_fingerprint=summaries_fingerprint,
            outputs=[summaries_path],
            model=scene_summary_model,
            prompt_version=scene_summary_model["prompt_version"],
            schema_version=phase01["schemas"]["scene_summaries"],
        )
    _emit_stage_progress(
        manager,
        "scene_summaries",
        scratch,
        status="complete",
        reused=summaries_reused,
    )
    summaries_output_fingerprint = manager.stage_output_fingerprint("scene_summaries")

    yield "finalize"

    manager.active_stage = "package"
    _emit_stage_progress(manager, "package", scratch, status="start")
    package_fingerprint = compute_fingerprint(
        metadata_fingerprint,
        shots_output_fingerprint,
        keyframes_output_fingerprint,
        asr_output_fingerprint,
        ocr_output_fingerprint,
        captions_output_fingerprint,
        links_output_fingerprint,
        scenes_output_fingerprint,
        summaries_output_fingerprint,
        config.stage_config_hashes["package"],
    )
    package_config = config.payload["artifact"]["package"]
    package_filename = str(package_config["filename"]).format(video_id=video_id)
    if Path(package_filename).name != package_filename:
        raise ValueError("Phase01 package filename must resolve to a basename")
    package_path = stage_dir / package_filename
    package_reused = _restore_if_reusable(
        manager, "package", package_fingerprint, stage_dir
    )
    if not package_reused:
        artifact_dir = stage_dir / "package" / video_id
        _assemble_package(
            artifact_dir=artifact_dir,
            video_id=video_id,
            metadata_path=metadata_path,
            stage_dir=stage_dir,
            config=config,
        )
        write_artifact_zip(
            artifact_dir=artifact_dir,
            zip_path=package_path,
            video_id=video_id,
            artifact_type="structure",
            batch_id=str(config.payload["runtime"]["batch_id"]),
            worker_id=str(config.payload["runtime"]["worker_id"]),
            status="complete",
            schema_version="phase01_structure_v2",
        )
        validate_artifact_zip(package_path)
        manager.promote_stage(
            "package",
            input_fingerprint=package_fingerprint,
            outputs=[package_path],
            schema_version="phase01_structure_v2",
        )
    _emit_stage_progress(
        manager, "package", scratch, status="complete", reused=package_reused
    )

    local_artifact = release_dir / "artifacts" / "structure" / package_filename
    local_artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_path, local_artifact)
    if not sync_release:
        return {"video_id": video_id, "status": "complete_local", "artifact": str(local_artifact)}

    manager.active_stage = "sync"
    _emit_stage_progress(manager, "sync", scratch, status="start")
    sync_fingerprint = compute_fingerprint(
        package_fingerprint, sha256_file(package_path), config.stage_config_hashes["sync"]
    )
    sync_reused = manager.is_reusable("sync", input_fingerprint=sync_fingerprint)
    if not sync_reused:
        remote_root = str(package_config["root"]).format(
            release_id=config.payload["runtime"]["release_id"],
            batch_id=config.payload["runtime"]["batch_id"],
            video_id=video_id,
        ).strip("/")
        remote_path = f"{remote_root}/{package_filename}"
        release_store.upload_file(package_path, remote_path)
        with tempfile.TemporaryDirectory(prefix="phase01_sync_verify_") as tmp:
            verified = Path(tmp) / package_path.name
            release_store.download_file(remote_path, verified)
            if sha256_file(verified) != sha256_file(package_path):
                raise ValueError(f"Release artifact remote checksum mismatch: {remote_path}")
        receipt_path = stage_dir / "sync_receipt.json"
        _write_json(receipt_path, {
            "repo_id": release_store.repo_id,
            "remote_path": remote_path,
            "sha256": sha256_file(package_path),
            "synced_at": utc_now(),
        })
        manager.promote_stage(
            "sync",
            input_fingerprint=sync_fingerprint,
            outputs=[receipt_path],
            schema_version="phase01_sync_receipt_v1",
        )
    _emit_stage_progress(
        manager, "sync", scratch, status="complete", reused=sync_reused
    )
    return {"video_id": video_id, "status": "complete", "artifact": str(local_artifact)}


def _build_keyframes(
    *,
    video_id: str,
    video_path: Path,
    shots: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    keyframe_config = config["keyframe"]
    candidate_groups = []
    for shot in shots:
        by_role = candidate_frame_ids_for_shot(shot, keyframe_config)
        candidate_groups.append(
            {frame_id for role_ids in by_role.values() for frame_id in role_ids}
        )
    timeline_by_frame = {int(row["frame_id"]): row for row in timeline}
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    decoded_groups = iter_decode_frame_groups(video_path, candidate_groups)
    for shot, decoded in zip(shots, decoded_groups, strict=True):
        selected, candidate_diagnostics = select_keyframes_for_shot(shot, decoded, keyframe_config)
        selected_roles = sorted(item.role for item in selected)
        selected_frame_ids = sorted(item.frame_id for item in selected)
        shot_frame_span = int(shot["end_frame"]) - int(shot["start_frame"])
        shot_duration_sec = float(shot["end_sec"]) - float(shot["start_sec"])
        diagnostics.extend(
            {
                "shot_id": shot["shot_id"],
                "shot_frame_span": shot_frame_span,
                "shot_duration_sec": shot_duration_sec,
                "selected_roles": selected_roles,
                "selected_frame_ids": selected_frame_ids,
                "candidate_count": len(candidate_diagnostics),
                "valid_candidate_count": sum(1 for item in candidate_diagnostics if item.valid),
                "long_shot_coverage_warning": shot_duration_sec > 10.0 and len(selected_frame_ids) < 3,
                **item.__dict__,
            }
            for item in candidate_diagnostics
        )
        for item in selected:
            frame = decoded[item.frame_id]
            keyframe_id = f"{video_id}:{item.frame_id}"
            media_stem = f"{video_id}_f{item.frame_id:07d}"
            filename = f"{media_stem}.jpg"
            thumbnail_name = f"{media_stem}.webp"
            write_keyframe_images(
                frame,
                keyframe_path=output_dir / "keyframes" / filename,
                thumbnail_path=output_dir / "thumbnails" / thumbnail_name,
                keyframe_long_side=int(keyframe_config["encoding"]["long_side"]),
                jpeg_quality=int(keyframe_config["encoding"]["jpeg_quality"]),
                thumbnail_width=int(config["thumbnail"]["width"]),
                webp_quality=int(config["thumbnail"]["encoding"]["webp_quality"]),
            )
            timeline_row = timeline_by_frame[item.frame_id]
            rows.append({
                "keyframe_id": keyframe_id,
                "video_id": video_id,
                "frame_id": item.frame_id,
                "timestamp_sec": float(timeline_row["pts_time"]),
                "shot_id": str(shot["shot_id"]),
                "scene_id": None,
                "keyframe_role": item.role,
                "quality_score": item.quality_score,
                "is_representative": item.is_representative,
                "selection_reason": item.selection_reason,
                "keyframe_ref": f"media://keyframes/{video_id}/{filename}",
                "thumbnail_ref": f"media://thumbnails/{video_id}/{thumbnail_name}",
                "status": "pass",
            })
    _validate_keyframe_rows(shots, rows)
    _write_parquet(output_dir / "keyframes.parquet", rows)
    _write_jsonl(output_dir / "keyframe_diagnostics.jsonl", diagnostics)


def _caption_client_for_model(
    model_config: Mapping[str, Any],
    *,
    phase01: Mapping[str, Any],
    cache: ArtifactStore,
    lifecycle_callback=None,
):
    clients = [
        _structured_client_for_model(
            model_config,
            phase01=phase01,
            cache=cache,
            cache_prefix=f"{model_config['provider']}/shot_caption",
            lifecycle_callback=lifecycle_callback,
        )
    ]
    for fallback in model_config.get("fallbacks", []):
        clients.append(
            _structured_client_for_model(
                fallback,
                phase01=phase01,
                cache=cache,
                cache_prefix=f"{fallback['provider']}/shot_caption",
                lifecycle_callback=lifecycle_callback,
            )
        )
    return (
        clients[0]
        if len(clients) == 1
        else FallbackStructuredClient(
            clients, telemetry_callback=lifecycle_callback
        )
    )


def _semantic_model_config(
    models: Mapping[str, Any], stage_key: str
) -> dict[str, Any]:
    stage_config = dict(models[stage_key])
    model_key = stage_config.pop("model_key", None)
    if not model_key:
        return stage_config
    if str(model_key) not in models:
        raise ValueError(
            f"Phase01 semantic model_key does not exist: {model_key}"
        )
    return {**dict(models[str(model_key)]), **stage_config}


def _structured_client_for_model(
    model_config: Mapping[str, Any],
    *,
    phase01: Mapping[str, Any],
    cache: ArtifactStore,
    cache_prefix: str,
    lifecycle_callback=None,
):
    provider = str(model_config["provider"])
    if provider == "gemini":
        client = GeminiStructuredClient(
            model_id=str(model_config["model_id"]),
            api_config={
                **phase01["api"],
                "schema_repair_attempts": phase01["retry"]["schema_repair_attempts"],
                "thinking_level": model_config.get("thinking_level", "medium"),
            },
            cache=cache,
            cache_prefix=f"gemini/{cache_prefix}",
        )
        return MetadataStructuredClient(
            client,
            provider_name="gemini",
            model_id=str(model_config["model_id"]),
            model_revision=str(model_config["model_revision"]),
        )
    if provider in {"qwen_local", "vintern_local"}:
        inference_stage = (
            "shot_captions" if "shot_caption" in cache_prefix else "ocr"
        )
        return LocalVisionStructuredClient(
            model_config={
                **model_config,
                "total_attempts": phase01["retry"]["local_model_total_attempts"],
                "inference_batch_size": phase01["execution"][
                    "inference_batch_size"
                ][inference_stage],
            },
            cache=cache,
            cache_prefix=f"local_vlm/{cache_prefix}",
            lifecycle_callback=lifecycle_callback,
        )
    raise RuntimeError(f"Unsupported structured provider: {provider}")


def _build_ocr(
    *,
    video_id: str,
    keyframes: list[dict[str, Any]],
    stage_dir: Path,
    client,
    model_config: Mapping[str, Any],
    ocr_config: Mapping[str, Any],
    diagnostics: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    prompt = (_prompt_dir() / f"{model_config['prompt_version']}.txt").read_text(encoding="utf-8")
    schema = {
        "type": "object",
        "properties": {
            "full_text": {"type": "string"},
            "ocr_blocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "confidence": {"type": ["number", "null"]},
                    },
                    "required": ["text"],
                    "additionalProperties": True,
                },
            },
            "language": {"type": "string"},
            "confidence": {"type": ["number", "null"]},
        },
        # Only full_text is consumed downstream; ocr_blocks is a fallback the
        # local model often omits. Demanding it throws away otherwise good reads.
        "required": ["full_text"],
        "additionalProperties": False,
    }
    allowed_roles = {str(role) for role in ocr_config.get("run_on_keyframe_roles", [])}
    selected_keyframes = sorted(
        [
        keyframe
        for keyframe in keyframes
        if not allowed_roles or str(keyframe.get("keyframe_role")) in allowed_roles
        ],
        key=lambda row: (str(row["shot_id"]), int(row["frame_id"])),
    )
    gate_config = ocr_config.get("text_presence_filter", {})
    gate_enabled = bool(
        isinstance(gate_config, Mapping) and gate_config.get("enabled", False)
    )
    counts = {
        "gate_checked": 0,
        "gate_no_text": 0,
        "gate_failures": 0,
        "vintern_processed": 0,
        "dedup_reused": 0,
    }

    def _image_for(keyframe: Mapping[str, Any]) -> Path:
        return stage_dir / "keyframes" / Path(str(keyframe["keyframe_ref"])).name

    dedup_config = ocr_config.get("dedup", {})
    representative_of: dict[str, str] = {}
    if isinstance(dedup_config, Mapping) and dedup_config.get("enabled", False):
        representative_of = _dedup_groups(
            selected_keyframes,
            _image_for,
            int(dedup_config.get("hamming_threshold", 4)),
        )
        counts["dedup_reused"] = len(representative_of)

    rows_by_keyframe: dict[str, dict[str, Any]] = {}
    requests: list[StructuredRequest] = []
    request_keyframes: list[dict[str, Any]] = []
    deduped_keyframes: list[dict[str, Any]] = []
    for keyframe in selected_keyframes:
        keyframe_id = str(keyframe["keyframe_id"])
        if keyframe_id in representative_of:
            deduped_keyframes.append(keyframe)
            continue
        image = _image_for(keyframe)
        gate_decision = "uncertain"
        if gate_enabled:
            counts["gate_checked"] += 1
            try:
                gate_decision = _text_presence_gate(image, gate_config)
            except Exception:  # noqa: BLE001 - gate failure must run Vintern
                gate_decision = "error"
                counts["gate_failures"] += 1
        if gate_decision == "no_text":
            counts["gate_no_text"] += 1
            rows_by_keyframe[keyframe_id] = _ocr_row(
                video_id=video_id,
                keyframe=keyframe,
                text="",
                provider="opencv_text_gate",
                model_name="opencv_mser_canny",
                model_version="text_presence_gate_v1",
                language="vi",
                confidence=None,
                status="empty",
            )
            continue
        requests.append(
            StructuredRequest(
                request_kind="keyframe_ocr",
                video_id=video_id,
                prompt=prompt,
                prompt_version=str(model_config["prompt_version"]),
                response_schema_version=str(model_config["response_schema_version"]),
                response_schema=schema,
                image_paths=(image,),
                identity={"keyframe_id": keyframe_id},
            )
        )
        request_keyframes.append(keyframe)

    counts["vintern_processed"] = len(requests)
    responses: list[dict[str, Any] | None] = [None] * len(requests)
    errors: dict[int, Exception] = {}
    if requests:
        try:
            responses = list(client.request_many(requests))
        except BatchRequestError as exc:
            responses = list(exc.results)
            errors = dict(exc.errors)
        except Exception as exc:  # noqa: BLE001 - preserve per-keyframe degradation
            errors = {index: exc for index in range(len(requests))}

    for index, keyframe in enumerate(request_keyframes):
        keyframe_id = str(keyframe["keyframe_id"])
        response = responses[index] if index < len(responses) else None
        try:
            if index in errors:
                raise errors[index]
            if response is None:
                raise RuntimeError("OCR request returned no response")
            text = _ocr_text(response)
            status = "pass" if text else "empty"
            provider = str(response.get("__provider", model_config["provider"]))
            model_name = str(response.get("__model_id", model_config["model_id"]))
            model_version = str(response.get("__model_revision", model_config["model_revision"]))
            confidence = _nullable_confidence(response.get("confidence"))
            language = str(response.get("language") or "vi")
        except Exception:  # noqa: BLE001 - preserve per-keyframe degradation
            text = ""
            status = "failed"
            provider = str(model_config["provider"])
            model_name = str(model_config["model_id"])
            model_version = str(model_config["model_revision"])
            confidence = None
            language = "vi"
        rows_by_keyframe[keyframe_id] = _ocr_row(
            video_id=video_id,
            keyframe=keyframe,
            text=text,
            provider=provider,
            model_name=model_name,
            model_version=model_version,
            language=language,
            confidence=confidence,
            status=status,
        )
    for keyframe in deduped_keyframes:
        keyframe_id = str(keyframe["keyframe_id"])
        source = rows_by_keyframe.get(representative_of[keyframe_id])
        if source is None:
            continue
        rows_by_keyframe[keyframe_id] = _ocr_row(
            video_id=video_id,
            keyframe=keyframe,
            text=str(source["text"]),
            provider=str(source["provider"]),
            model_name=str(source["model_name"]),
            model_version=str(source["model_version"]),
            language=str(source["language"]),
            confidence=source["confidence"],
            status=str(source["status"]),
        )

    if diagnostics is not None:
        diagnostics.update(counts)
    return [
        rows_by_keyframe[str(row["keyframe_id"])]
        for row in selected_keyframes
        if str(row["keyframe_id"]) in rows_by_keyframe
    ]


def _ocr_row(
    *,
    video_id: str,
    keyframe: Mapping[str, Any],
    text: str,
    provider: str,
    model_name: str,
    model_version: str,
    language: str,
    confidence: float | None,
    status: str,
) -> dict[str, Any]:
    keyframe_id = str(keyframe["keyframe_id"])
    return {
        "ocr_id": f"{keyframe_id}:ocr",
        "video_id": video_id,
        "keyframe_id": keyframe_id,
        "shot_id": str(keyframe["shot_id"]),
        "frame_id": int(keyframe["frame_id"]),
        "text": text,
        "raw_text": text,
        "provider": provider,
        "model_name": model_name,
        "model_version": model_version,
        "language": language,
        "confidence": confidence,
        "status": status,
    }


def _text_presence_gate(
    image_path: Path, config: Mapping[str, Any]
) -> str:
    import cv2

    if str(config.get("policy")) != "opencv_conservative_v1":
        raise ValueError(f"Unsupported OCR text gate policy: {config.get('policy')}")
    grayscale = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None or grayscale.size == 0:
        raise ValueError(f"OCR text gate could not decode {image_path}")
    max_long_side = int(config["max_long_side"])
    height, width = grayscale.shape[:2]
    if max(height, width) > max_long_side:
        scale = max_long_side / float(max(height, width))
        grayscale = cv2.resize(
            grayscale,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    edges = cv2.Canny(
        grayscale,
        int(config["canny_low"]),
        int(config["canny_high"]),
    )
    edge_density = float((edges > 0).mean())
    gray_std = float(grayscale.std())
    regions, boxes = cv2.MSER_create().detectRegions(grayscale)
    plausible_regions = 0
    image_area = float(grayscale.shape[0] * grayscale.shape[1])
    for region, box in zip(regions, boxes, strict=False):
        x, y, region_width, region_height = (int(value) for value in box)
        del x, y
        area = float(region_width * region_height)
        aspect = region_width / float(max(1, region_height))
        if (
            len(region) >= 10
            and 0.00002 <= area / image_area <= 0.08
            and 0.15 <= aspect <= 20.0
        ):
            plausible_regions += 1
            if plausible_regions >= 2:
                break
    if (
        plausible_regions == 0
        and edge_density <= float(config["max_no_text_edge_density"])
        and gray_std <= float(config["max_no_text_gray_std"])
    ):
        return "no_text"
    return "uncertain"


def _perceptual_hash(image_path: Path) -> int | None:
    """64-bit DCT hash. Returns None when the image cannot be read."""
    import cv2

    grayscale = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None or grayscale.size == 0:
        return None
    resized = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype("float32"))[:8, :8].flatten()
    # Skip the DC term: it tracks overall brightness, not structure.
    median = float(np.median(coefficients[1:]))
    bits = 0
    for index, value in enumerate(coefficients):
        if float(value) > median:
            bits |= 1 << index
    return bits


def _hamming_distance(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def _dedup_groups(
    keyframes: list[dict[str, Any]],
    image_for: Callable[[dict[str, Any]], Path],
    threshold: int,
) -> dict[str, str]:
    """Map keyframe_id -> representative keyframe_id within the same shot.

    The three role keyframes of one shot usually show the same on-screen text
    (subtitles, logos, signage hold still), so OCRing all three re-reads the same
    characters. Grouping only ever happens inside a shot — never across shots.
    """
    representative_of: dict[str, str] = {}
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for keyframe in keyframes:
        by_shot.setdefault(str(keyframe.get("shot_id")), []).append(keyframe)

    for shot_keyframes in by_shot.values():
        seen: list[tuple[int, str]] = []
        for keyframe in shot_keyframes:
            keyframe_id = str(keyframe["keyframe_id"])
            digest = _perceptual_hash(image_for(keyframe))
            if digest is None:
                continue
            match = next(
                (
                    other_id
                    for other_hash, other_id in seen
                    if _hamming_distance(digest, other_hash) <= threshold
                ),
                None,
            )
            if match is None:
                seen.append((digest, keyframe_id))
            else:
                representative_of[keyframe_id] = match
    return representative_of


def _ocr_text(payload: Mapping[str, Any]) -> str:
    full_text = str(payload.get("full_text", "")).strip()
    if full_text:
        return full_text
    blocks = payload.get("ocr_blocks", [])
    if not isinstance(blocks, list):
        return ""
    return " ".join(
        str(block.get("text", "")).strip()
        for block in blocks
        if isinstance(block, Mapping) and str(block.get("text", "")).strip()
    )


def _ocr_text_by_keyframe(rows: list[dict[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        if str(row.get("status")) == "failed":
            continue
        text = str(row.get("text") or row.get("raw_text") or "").strip()
        if text:
            output[str(row["keyframe_id"])] = text
    return output


def _nullable_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, number))


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        if hasattr(value, "tolist"):
            value = value.tolist()
        else:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _build_captions(
    *, video_id: str, shots: list[dict[str, Any]], keyframes: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]], stage_dir: Path, client, model_config: Mapping[str, Any],
    max_concurrency: int, caption_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if max_concurrency < 1:
        raise ValueError("caption max_concurrency must be positive")
    if str(model_config.get("provider")) != "gemini":
        max_concurrency = 1
    representative = {str(row["shot_id"]): row for row in keyframes if row["is_representative"]}
    prompt = (_prompt_dir() / f"{model_config['prompt_version']}.txt").read_text(encoding="utf-8")
    ocr_by_keyframe = _ocr_text_by_keyframe(ocr_rows)
    schema = {
        "type": "object",
        "properties": {
            "caption_vi": {"type": "string", "minLength": 1},
            "caption_en": {"type": "string", "minLength": 1},
            "objects_vi": {"type": "array", "items": {"type": "string"}},
            "objects_en": {"type": "array", "items": {"type": "string"}},
            "actions_vi": {"type": "array", "items": {"type": "string"}},
            "actions_en": {"type": "array", "items": {"type": "string"}},
            "scene_type": {"type": "string"},
        },
        "required": [
            "caption_vi",
            "caption_en",
            "objects_vi",
            "objects_en",
            "actions_vi",
            "actions_en",
            "scene_type",
        ],
        "additionalProperties": False,
    }
    ordered_shots = sorted(shots, key=lambda row: str(row["shot_id"]))
    requests: list[StructuredRequest] = []
    request_context: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for shot in ordered_shots:
        keyframe = representative[str(shot["shot_id"])]
        keyframe_id = str(keyframe["keyframe_id"])
        image = stage_dir / "keyframes" / Path(str(keyframe["keyframe_ref"])).name
        ocr_text = ocr_by_keyframe.get(keyframe_id, "")
        request_prompt = (
            prompt
            + "\n\nOCR TEXT DETECTED FOR THIS REPRESENTATIVE KEYFRAME:\n"
            + (ocr_text if ocr_text else "<none>")
        )
        requests.append(
            StructuredRequest(
                request_kind="shot_caption",
                video_id=video_id,
                prompt=request_prompt,
                prompt_version=str(model_config["prompt_version"]),
                response_schema_version=str(model_config["response_schema_version"]),
                response_schema=schema,
                image_paths=(image,),
                identity={"shot_id": shot["shot_id"]},
            )
        )
        request_context.append((shot, keyframe, keyframe_id))
    responses: list[dict[str, Any] | None] = [None] * len(requests)
    errors: dict[int, Exception] = {}
    if requests:
        try:
            responses = list(client.request_many(requests))
        except BatchRequestError as exc:
            responses = list(exc.results)
            errors = dict(exc.errors)
        except Exception as exc:  # noqa: BLE001 - preserve per-shot degradation
            errors = {index: exc for index in range(len(requests))}
    if len(responses) != len(request_context):
        raise ValueError(
            "caption client returned a different number of responses than requests"
        )
    rows: list[dict[str, Any]] = []
    for index, context in enumerate(request_context):
        shot, keyframe, keyframe_id = context
        response = responses[index] if index < len(responses) else None
        try:
            if index in errors:
                raise errors[index]
            if response is None:
                raise RuntimeError("caption request returned no response")
            caption_vi = _required_text(response, "caption_vi")
            caption_en = _required_text(response, "caption_en")
            objects_vi = _string_list(response, "objects_vi")
            objects_en = _string_list(response, "objects_en")
            actions_vi = _string_list(response, "actions_vi")
            actions_en = _string_list(response, "actions_en")
            # v4 dropped these fields from the prompt/schema — the model already has
            # the OCR text in its input, so re-emitting it just burns tokens. Reuse
            # the same OCR text the request was built from instead of asking again.
            shot_ocr_text = ocr_by_keyframe.get(keyframe_id, "")
            visible_text_summary_vi = shot_ocr_text
            visible_text_summary_en = shot_ocr_text
            scene_type = str(response.get("scene_type", ""))
            provider = str(response.get("__provider", model_config["provider"]))
            model_name = str(response.get("__model_id", model_config["model_id"]))
            model_version = str(response.get("__model_revision", model_config["model_revision"]))
            status = "pass"
        except Exception:  # noqa: BLE001 - preserve per-shot degradation
            caption_vi = ""
            caption_en = ""
            objects_vi = []
            objects_en = []
            actions_vi = []
            actions_en = []
            visible_text_summary_vi = ""
            visible_text_summary_en = ""
            scene_type = ""
            provider = str(model_config["provider"])
            model_name = str(model_config["model_id"])
            model_version = str(model_config["model_revision"])
            status = "failed"
        rows.append({
            "shot_caption_id": f"{shot['shot_id']}_caption", "video_id": video_id,
            "shot_id": shot["shot_id"], "representative_keyframe_id": keyframe_id,
            "representative_timestamp_sec": keyframe["timestamp_sec"],
            "caption_vi": caption_vi, "caption_en": caption_en,
            "objects_vi": objects_vi, "objects_en": objects_en,
            "actions_vi": actions_vi, "actions_en": actions_en,
            "visible_text_summary_vi": visible_text_summary_vi,
            "visible_text_summary_en": visible_text_summary_en,
            "scene_type": scene_type,
            "provider": provider, "model_name": model_name,
            "model_version": model_version, "prompt_version": model_config["prompt_version"],
            "schema_version": model_config["response_schema_version"], "confidence": None, "status": status,
        })
    failed_count = sum(1 for row in rows if row["status"] == "failed")
    if rows:
        max_failed_ratio = float((caption_config or {}).get("max_failed_ratio", 0.5))
        if failed_count / len(rows) > max_failed_ratio:
            raise RuntimeError(
                f"shot_captions stage exceeded max_failed_ratio: "
                f"{failed_count}/{len(rows)} shots failed"
            )
    return rows


def _build_scene_evidence(shots, keyframes, ocr_rows, captions, asr_rows, links, stage_dir):
    by_shot_keyframes: dict[str, list[dict[str, Any]]] = {}
    for row in keyframes:
        by_shot_keyframes.setdefault(str(row["shot_id"]), []).append(row)
    caption_by_shot = {str(row["shot_id"]): row for row in captions}
    ocr_by_keyframe = _ocr_text_by_keyframe(ocr_rows)
    asr_by_id = {str(row["asr_segment_id"]): row for row in asr_rows}
    links_by_shot: dict[str, list[dict[str, Any]]] = {}
    for row in links:
        links_by_shot.setdefault(str(row["shot_id"]), []).append(row)
    evidence = []
    for shot in shots:
        shot_id = str(shot["shot_id"])
        frames = by_shot_keyframes[shot_id]
        representative = next(row for row in frames if row["is_representative"])
        role_paths = {
            str(row["keyframe_role"]): stage_dir
            / "keyframes"
            / Path(str(row["keyframe_ref"])).name
            for row in frames
        }
        linked_ids = {
            str(link["asr_segment_id"])
            for link in links_by_shot.get(shot_id, [])
            if str(link["asr_segment_id"]) in asr_by_id
        }
        ordered_segments = sorted(
            (asr_by_id[segment_id] for segment_id in linked_ids),
            key=lambda row: (
                float(row["start_sec"]),
                float(row["end_sec"]),
                str(row["asr_segment_id"]),
            ),
        )
        transcript = " ".join(
            str(row["text"]).strip()
            for row in ordered_segments
            if str(row["text"]).strip()
        )
        caption = caption_by_shot.get(shot_id)
        caption_failed = caption is None or caption.get("status") == "failed"
        if caption_failed:
            # group_scenes indexes evidence by shot position, so a dropped row
            # would misalign every later shot. Emit an empty-text placeholder:
            # the keyframe image still carries evidence for boundary judging.
            caption = {}
        shot_ocr = [
            ocr_by_keyframe[str(row["keyframe_id"])]
            for row in frames
            if ocr_by_keyframe.get(str(row["keyframe_id"]))
        ]
        evidence.append({
            "shot_id": shot_id, "start_sec": shot["start_sec"], "end_sec": shot["end_sec"],
            "representative_path": stage_dir
            / "keyframes"
            / Path(str(representative["keyframe_ref"])).name,
            "early_path": role_paths.get("early"), "late_path": role_paths.get("late"),
            "caption_missing": caption_failed,
            "caption_vi": caption.get("caption_vi", ""), "caption_en": caption.get("caption_en", ""),
            "objects_vi": _string_list(caption, "objects_vi"), "objects_en": _string_list(caption, "objects_en"),
            "actions_vi": _string_list(caption, "actions_vi"), "actions_en": _string_list(caption, "actions_en"),
            "visible_text_summary_vi": caption.get("visible_text_summary_vi", ""),
            "visible_text_summary_en": caption.get("visible_text_summary_en", ""),
            "ocr_text": shot_ocr, "transcript": transcript,
        })
    return evidence


def _build_scene_transcript_links(scenes, asr_rows):
    rows = []
    for scene in scenes:
        for segment in asr_rows:
            overlap = min(float(scene["end_sec"]), float(segment["end_sec"])) - max(float(scene["start_sec"]), float(segment["start_sec"]))
            if overlap > 0:
                duration = float(segment["end_sec"]) - float(segment["start_sec"])
                if duration <= 0:
                    raise ValueError("ASR segment duration must be positive")
                rows.append({"video_id": scene["video_id"], "scene_id": scene["scene_id"], "asr_segment_id": segment["asr_segment_id"], "coverage": min(1.0, max(0.0, overlap / duration))})
    return rows


def _build_scene_summaries(*, video_id, scenes, shots, keyframes, ocr_rows, captions, asr_rows, scene_links, stage_dir, client, model_config, summary_config):
    prompt_base = (_prompt_dir() / f"{model_config['prompt_version']}.txt").read_text(encoding="utf-8")
    schema = {"type": "object", "properties": {"summary_vi": {"type": "string", "minLength": 1}, "summary_en": {"type": "string", "minLength": 1}}, "required": ["summary_vi", "summary_en"], "additionalProperties": False}
    captions_by_shot = {str(row["shot_id"]): row for row in captions}
    ocr_by_keyframe = _ocr_text_by_keyframe(ocr_rows)
    representative = {str(row["shot_id"]): row for row in keyframes if row["is_representative"]}
    keyframes_by_shot: dict[str, list[dict[str, Any]]] = {}
    for row in keyframes:
        keyframes_by_shot.setdefault(str(row["shot_id"]), []).append(row)
    asr_by_id = {str(row["asr_segment_id"]): row for row in asr_rows}
    links_by_scene: dict[str, list[dict[str, Any]]] = {}
    for link in scene_links: links_by_scene.setdefault(str(link["scene_id"]), []).append(link)
    rows = []
    for scene in scenes:
        scene_shots = [shot for shot in shots if float(shot["start_sec"]) >= float(scene["start_sec"]) and float(shot["end_sec"]) <= float(scene["end_sec"]) + 1e-6]
        sampled_shots = _evenly_sample(
            scene_shots, int(summary_config["max_representative_images"])
        )
        sheet_tiles = [
            (
                stage_dir
                / "keyframes"
                / Path(str(representative[str(shot["shot_id"])]["keyframe_ref"])).name,
                str(shot["shot_id"]),
            )
            for shot in sampled_shots
        ]
        sheet_path = (
            stage_dir
            / "diagnostics"
            / "scene_summaries"
            / f"{scene['scene_id']}.jpg"
        )
        contact_sheet = write_contact_sheet(sheet_tiles, sheet_path)
        image_paths = (contact_sheet,) if contact_sheet is not None else ()
        shot_evidence = []
        for shot in scene_shots:
            shot_id = str(shot["shot_id"])
            caption = captions_by_shot.get(shot_id)
            if caption is None or caption.get("status") == "failed":
                # Shot's caption request failed the batch — skip it from
                # summary evidence instead of raising.
                continue
            shot_evidence.append({
                "shot_id": shot_id,
                "caption_vi": caption["caption_vi"],
                "caption_en": caption["caption_en"],
                "objects_vi": _string_list(caption, "objects_vi"),
                "objects_en": _string_list(caption, "objects_en"),
                "actions_vi": _string_list(caption, "actions_vi"),
                "actions_en": _string_list(caption, "actions_en"),
                "visible_text_summary_vi": caption.get("visible_text_summary_vi", ""),
                "visible_text_summary_en": caption.get("visible_text_summary_en", ""),
                "ocr_text": [
                    ocr_by_keyframe[str(row["keyframe_id"])]
                    for row in keyframes_by_shot.get(shot_id, [])
                    if ocr_by_keyframe.get(str(row["keyframe_id"]))
                ],
            })
        evidence = {"shots": shot_evidence, "transcript": [asr_by_id[str(link["asr_segment_id"])]["text"] for link in links_by_scene.get(str(scene["scene_id"]), [])], "timeline": [scene["start_sec"], scene["end_sec"]]}
        try:
            response = client.request(
                StructuredRequest(
                    request_kind="scene_summary",
                    video_id=video_id,
                    prompt=prompt_base
                    + "\n\nSCENE EVIDENCE:\n"
                    + json.dumps(evidence, ensure_ascii=False),
                    prompt_version=str(model_config["prompt_version"]),
                    response_schema_version=str(
                        model_config["response_schema_version"]
                    ),
                    response_schema=schema,
                    image_paths=image_paths,
                    identity={"scene_id": scene["scene_id"]},
                )
            )
            rows.append({
                "scene_id": scene["scene_id"],
                "video_id": video_id,
                "summary_vi": _required_text(response, "summary_vi"),
                "summary_en": _required_text(response, "summary_en"),
                "provider": str(response.get("__provider", model_config["provider"])),
                "model_name": str(response.get("__model_id", model_config["model_id"])),
                "model_version": str(
                    response.get("__model_revision", model_config["model_revision"])
                ),
                "prompt_version": model_config["prompt_version"],
                "schema_version": model_config["response_schema_version"],
                "confidence": None,
                "status": "pass",
            })
        except Exception as exc:  # noqa: BLE001 - preserve per-scene degradation
            if _retryable_video_error(exc):
                # Infrastructure faults (OOM, timeouts, 5xx) are systemic, not
                # scene-specific — surface them instead of masking as "failed".
                raise
            logging.getLogger(__name__).warning(
                "scene_summary failed for scene_id=%s video_id=%s: %s",
                scene["scene_id"], video_id, str(exc)[:200],
            )
            rows.append({
                "scene_id": scene["scene_id"],
                "video_id": video_id,
                "summary_vi": "",
                "summary_en": "",
                "provider": str(model_config["provider"]),
                "model_name": str(model_config["model_id"]),
                "model_version": str(model_config["model_revision"]),
                "prompt_version": model_config["prompt_version"],
                "schema_version": model_config["response_schema_version"],
                "confidence": None,
                "status": "failed",
            })
    failed_count = sum(1 for row in rows if row["status"] == "failed")
    if rows:
        max_failed_ratio = float((summary_config or {}).get("max_failed_ratio", 0.5))
        if failed_count / len(rows) > max_failed_ratio:
            raise RuntimeError(
                f"scene_summaries stage exceeded max_failed_ratio: "
                f"{failed_count}/{len(rows)} scenes failed"
            )
    return rows


def _evenly_sample(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("Scene summary image limit must be positive")
    if len(rows) <= maximum:
        return rows
    indices = sorted(
        {
            round(position * (len(rows) - 1) / (maximum - 1))
            for position in range(maximum)
        }
    ) if maximum > 1 else [len(rows) // 2]
    return [rows[index] for index in indices]


def _assemble_package(*, artifact_dir: Path, video_id: str, metadata_path: Path, stage_dir: Path, config: ResolvedPhase01Config):
    if artifact_dir.exists(): shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    metadata = read_metadata(metadata_path)
    _write_json(artifact_dir / "metadata_normalized.json", metadata)
    for name in ("shots.parquet", "keyframes.parquet", "asr_segments.parquet", "ocr.parquet", "shot_captions.parquet", "shot_transcript_links.parquet", "scenes.parquet", "scene_transcript_links.parquet", "scene_summaries.parquet"):
        shutil.copy2(stage_dir / name, artifact_dir / name)
    shutil.copytree(stage_dir / "keyframes", artifact_dir / "keyframes")
    shutil.copytree(stage_dir / "thumbnails", artifact_dir / "thumbnails")
    diagnostics = artifact_dir / "diagnostics"; diagnostics.mkdir()
    for name in ("keyframe_diagnostics.jsonl", "scene_boundary_diagnostics.jsonl", "transnet_predictions.json", "asr_status.json", "ocr_status.json"):
        source = stage_dir / name
        if source.exists(): shutil.copy2(source, diagnostics / name)
    # A complete per-video package has no item-level errors, but retains the
    # canonical file so downstream readers never need layout-specific logic.
    _write_jsonl(artifact_dir / "errors.jsonl", [])
    _backfill_scene_ids(artifact_dir)
    persist_resolved_phase01_config(config, artifact_dir / "resolved_config.json")
    counts = {path.stem: len(pd.read_parquet(path)) for path in artifact_dir.glob("*.parquet")}
    _validate_package_invariants(artifact_dir, counts)
    _write_json(artifact_dir / "manifest.json", {"schema_version": "phase01_video_manifest_v2", "video_id": video_id, "status": "complete", "config_hash": config.config_hash, "stage_config_hashes": config.stage_config_hashes, "counts": counts, "created_at": utc_now()})


def _backfill_scene_ids(artifact_dir: Path):
    shots = pd.read_parquet(artifact_dir / "shots.parquet")
    keyframes = pd.read_parquet(artifact_dir / "keyframes.parquet")
    scenes = pd.read_parquet(artifact_dir / "scenes.parquet")
    mapping = {}
    shot_rows = shots.sort_values("shot_index").to_dict("records")
    for scene in scenes.to_dict("records"):
        start = next(i for i,row in enumerate(shot_rows) if row["shot_id"] == scene["start_shot_id"])
        end = next(i for i,row in enumerate(shot_rows) if row["shot_id"] == scene["end_shot_id"])
        for row in shot_rows[start:end+1]: mapping[str(row["shot_id"])] = str(scene["scene_id"])
    shots["scene_id"] = shots["shot_id"].astype(str).map(mapping)
    keyframes["scene_id"] = keyframes["shot_id"].astype(str).map(mapping)
    keyframe_counts = keyframes.groupby("scene_id").size().to_dict()
    scenes["keyframe_count"] = scenes["scene_id"].map(keyframe_counts).fillna(0).astype(int)
    shots.to_parquet(artifact_dir / "shots.parquet", index=False)
    keyframes.to_parquet(artifact_dir / "keyframes.parquet", index=False)
    scenes.to_parquet(artifact_dir / "scenes.parquet", index=False)


def _validate_package_invariants(artifact_dir: Path, counts: Mapping[str, int]):
    validate_phase01_package(artifact_dir)


def _restore_if_reusable(manager, stage, fingerprint, target_dir):
    return manager.is_reusable(
        stage,
        input_fingerprint=fingerprint,
        restore_dir=target_dir,
    )


def _restore_keyframes_if_reusable(manager, fingerprint, target_dir):
    if not manager.is_reusable(
        "keyframes", input_fingerprint=fingerprint, restore_dir=target_dir
    ):
        return False
    bundle = target_dir / "keyframes.zip"
    _safe_extract_zip(bundle, target_dir)
    return True


def _stage_fingerprint(manager, stage, upstream): return compute_fingerprint(upstream, manager.stage_config_hashes[stage])


def _write_directory_zip(root, output, members):
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            path = root / member
            if path.is_file(): archive.write(path, path.relative_to(root))
            elif path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file(): archive.write(child, child.relative_to(root))


def _safe_extract_zip(bundle: Path, target_dir: Path) -> None:
    root = target_dir.resolve()
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (root / member.filename).resolve()
            target.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _emit_progress(
    *,
    event: str,
    status: str,
    scratch: Path,
    **details: Any,
) -> None:
    payload = {
        "event": event,
        "status": status,
        **_resource_snapshot(scratch),
        **details,
    }
    print(f"[phase01] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def _emit_stage_progress(
    manager: CheckpointManager,
    stage: str,
    scratch: Path,
    *,
    status: str,
    reused: bool | None = None,
) -> None:
    timer_key = (id(manager), stage)
    details: dict[str, Any] = {
        "release_id": manager.release_id,
        "video_id": manager.video_id,
        "stage": stage,
        **_MANAGER_RUNTIME_CONTEXT.get(manager, {}),
    }
    if status == "start":
        _STAGE_TIMERS[timer_key] = time.monotonic()
    else:
        started = _STAGE_TIMERS.pop(timer_key, None)
        if started is not None:
            details["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if reused is not None:
        details["source"] = "checkpoint" if reused else "computed"
    _emit_progress(event="stage", status=status, scratch=scratch, **details)


def _model_lifecycle_callback(
    *,
    scratch_root: Path,
    release_id: str,
    batch_id: str,
    chunk_index: int,
    chunk_size: int,
    stage: str,
):
    def emit(payload: Mapping[str, Any]) -> None:
        details = dict(payload)
        status = str(details.pop("status"))
        if "load_seconds" in details:
            details.setdefault("elapsed_seconds", details["load_seconds"])
        _emit_progress(
            event="model",
            status=status,
            scratch=scratch_root,
            release_id=release_id,
            batch_id=batch_id,
            stage=stage,
            chunk_index=chunk_index,
            chunk_size=chunk_size,
            **details,
        )

    return emit


def _resource_snapshot(scratch: Path) -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {
        "scratch_free_gb": None,
        "ram_used_gb": None,
        "ram_available_gb": None,
        "gpu_allocated_gb": None,
        "gpu_reserved_gb": None,
        "gpu_peak_allocated_gb": None,
    }
    try:
        snapshot["scratch_free_gb"] = _bytes_to_gb(shutil.disk_usage(scratch).free)
    except (FileNotFoundError, OSError):
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_bytes = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        available_bytes = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        snapshot["ram_available_gb"] = _bytes_to_gb(available_bytes)
        snapshot["ram_used_gb"] = _bytes_to_gb(total_bytes - available_bytes)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        import torch
    except ImportError:
        return snapshot
    try:
        if torch.cuda.is_available():
            snapshot["gpu_allocated_gb"] = _bytes_to_gb(
                torch.cuda.memory_allocated()
            )
            snapshot["gpu_reserved_gb"] = _bytes_to_gb(
                torch.cuda.memory_reserved()
            )
            snapshot["gpu_peak_allocated_gb"] = _bytes_to_gb(
                torch.cuda.max_memory_allocated()
            )
    except (RuntimeError, TypeError, ValueError):
        pass
    return snapshot


def _bytes_to_gb(value: float) -> float:
    return round(float(value) / (1024**3), 3)


def _scratch_free_gb(scratch: Path) -> float:
    return float(shutil.disk_usage(scratch).free) / (1024**3)


def _mapping_raw_bytes(mapping: Mapping[str, Any]) -> int | None:
    value = mapping.get("video_size_bytes")
    if _present_scalar(value):
        try:
            size = int(value)
        except (TypeError, ValueError):
            size = -1
        if size >= 0:
            return size
    for key in ("video_local_path", "debug_video_local_path", "source_video_path"):
        path_value = mapping.get(key)
        if _present_scalar(path_value):
            path = Path(str(path_value))
            if path.is_file():
                return path.stat().st_size
    return None


def _hf_store(config, *, cache_dir: Path | str | None = None):
    return HuggingFaceDatasetArtifactStore(
        repo_id=str(config["repo_id"]),
        repo_type=str(config.get("repo_type", "dataset")),
        revision=str(config.get("revision", "main")),
        token=os.environ.get("AIC_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        prefix=str(config.get("prefix", "")),
        cache_dir=cache_dir,
    )


def _release_structured_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
    # Third-party wrappers do not consistently release cyclic references or
    # CUDA caches even after their close hook returns.
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError:
        pass


def _materialize_canonical(mapping, key, target_dir):
    local_keys = ("video_local_path", "debug_video_local_path", "source_video_path") if "video" in key else ("metadata_local_path", "debug_metadata_local_path", "source_metadata_path")
    for local_key in local_keys:
        value = mapping.get(local_key)
        if _present_scalar(value) and Path(str(value)).is_file(): return Path(str(value))
    target_dir.mkdir(parents=True, exist_ok=True)
    remote_path = str(mapping[key]); target = target_dir / Path(remote_path).name
    store = HuggingFaceDatasetArtifactStore(
        repo_id=str(mapping["canonical_repo_id"]),
        repo_type=(
            str(mapping.get("canonical_repo_type"))
            if _present_scalar(mapping.get("canonical_repo_type"))
            else "dataset"
        ),
        revision=(
            str(mapping.get("canonical_revision"))
            if _present_scalar(mapping.get("canonical_revision"))
            else "main"
        ),
        token=os.environ.get("AIC_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        prefix=(
            str(mapping.get("canonical_prefix"))
            if _present_scalar(mapping.get("canonical_prefix"))
            else ""
        ),
        cache_dir=target_dir / ".hf_cache",
    )
    return store.download_file(remote_path, target)


def _timeline_path(release_dir, video_id, video_row):
    candidate = video_row.get("frame_timeline_ref")
    ref = candidate if _present_scalar(candidate) else f"frame_timeline/{video_id}.parquet"
    path = release_dir / str(ref)
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def _stable_video_identity(mapping):
    return {key: _json_scalar(mapping.get(key)) for key in ("canonical_repo_id", "canonical_revision", "canonical_prefix", "canonical_video_path", "video_size_bytes")}


def _stable_metadata_identity(mapping):
    return {
        key: _json_scalar(mapping.get(key))
        for key in (
            "canonical_repo_id",
            "canonical_revision",
            "canonical_prefix",
            "canonical_metadata_path",
            "metadata_size_bytes",
            "metadata_schema_version",
        )
    }


def _present_scalar(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _json_scalar(value: Any) -> Any:
    if not _present_scalar(value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _validate_keyframe_rows(shots, rows):
    for shot in shots:
        members = [row for row in rows if row["shot_id"] == shot["shot_id"]]
        if not members or sum(row["is_representative"] for row in members) != 1: raise ValueError(f"Invalid keyframe selection for {shot['shot_id']}")
        if any(not (shot["start_frame"] <= row["frame_id"] < shot["end_frame"]) for row in members): raise ValueError("Keyframe lies outside its shot")


def _write_parquet(path, rows, empty_columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / f"{path.stem}.schema.json"
    if schema_path.is_file():
        validate_rows(path.stem, rows)
    pd.DataFrame(rows, columns=empty_columns if not rows and empty_columns else None).to_parquet(path, index=False)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prompt_dir(): return Path(__file__).resolve().parents[3] / "prompts"


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload[key]).strip()
    if not value:
        raise ValueError(
            f"Structured field must contain non-whitespace text: {key}"
        )
    return value


def _retryable_video_error(exc):
    # "decode" is here for ffmpeg, not for JSON. A malformed reply says nothing
    # about the machine, and treating one as infrastructure re-raises past the
    # per-scene degradation below — three good videos died on 27/08 because one
    # summary in thirteen came back with the wrong closing bracket.
    if _carries(exc, (SystemicProviderError, MemoryError)):
        return True
    if _carries(exc, (json.JSONDecodeError, ValidationError)):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in ("timeout", "timed out", "429", "500", "502", "503", "504", "out of memory", "temporarily unavailable", "connection reset", "decode", "i/o"))


def _carries(exc, types: tuple[type, ...]) -> bool:
    """True when exc is one of `types`, or wraps one.

    A batch failure reports as BatchRequestError and keeps the real cause in
    `.errors`, so the type that decides how to react is never the outer one.
    """
    seen: set[int] = set()
    queue = [exc]
    while queue:
        current = queue.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, types):
            return True
        queue.extend((getattr(current, "errors", None) or {}).values())
        queue.append(current.__cause__)
        queue.append(current.__context__)
    return False
