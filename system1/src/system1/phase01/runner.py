from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from system1.artifacts.checkpoint import sha256_file
from system1.artifacts.hf_store import HuggingFaceDatasetArtifactStore
from system1.config import (
    load_configs,
    persist_resolved_phase01_config,
    require_phase01_production_ready,
    resolve_phase01_config,
)
from system1.phase01.checkpoint import compute_fingerprint
from system1.phase01.model_artifacts import materialize_transnet_artifact
from system1.phase01.phase00 import discover_phase00_candidates, resolve_phase00_release
from system1.phase01.preflight import (
    PreflightResult,
    run_phase01_preflight,
    run_phase01_storage_preflight,
)
from system1.phase01.production import process_production_batch
from system1.release.sync import phase00_ingestion_remote_prefix


@dataclass(frozen=True)
class Phase01RunResult:
    release_id: str
    release_dir: Path
    resolved_config_path: Path
    worker_report_path: Path
    preflight: PreflightResult


def run_phase01_pipeline(
    *,
    config_dir: Path,
    output_root: Path,
    user_settings: dict[str, Any],
    restore_phase00: bool = True,
    sync_release: bool = True,
    validate_remote: bool = True,
) -> Phase01RunResult:
    """Resolve, restore, preflight, resume, process, package, and sync Phase01."""

    configs = load_configs(config_dir)
    release_storage = dict(configs["storage"]["release"])
    for key, setting in (
        ("repo_id", "hf_release_repo"),
        ("repo_type", "hf_repo_type"),
        ("revision", "hf_release_revision"),
        ("prefix", "hf_release_prefix"),
    ):
        if user_settings.get(setting) is not None:
            release_storage[key] = user_settings[setting]
    discovery_cache = output_root.resolve().parent / ".phase01_hf_cache" / f"discovery-{os.getpid()}"
    try:
        release_store = _hf_store(release_storage, cache_dir=discovery_cache)
        selected = resolve_phase00_release(
            discover_phase00_candidates(release_store),
            release_id_override=str(user_settings.get("release_id_override") or "") or None,
        )
    finally:
        shutil.rmtree(discovery_cache, ignore_errors=True)
    resolved = resolve_phase01_config(
        config_dir,
        user_settings=user_settings,
        phase00_release_id=selected.release_id,
    )
    require_phase01_production_ready(resolved)
    scratch_root = _scratch_root(resolved.payload["storage"], output_root)
    if validate_remote:
        preflight_cache = scratch_root / ".hf_cache" / f"storage_preflight-{os.getpid()}"
        try:
            run_phase01_storage_preflight(resolved, cache_dir=preflight_cache)
        finally:
            shutil.rmtree(preflight_cache, ignore_errors=True)
    release_id = str(resolved.payload["runtime"]["release_id"])
    release_dir = output_root.resolve() / release_id
    if restore_phase00:
        restore_cache = scratch_root / ".hf_cache" / f"phase00_restore-{os.getpid()}"
        try:
            _restore_phase00_if_needed(
                output_root=output_root,
                release_id=release_id,
                batch_id=str(resolved.payload["runtime"]["batch_id"]),
                storage=resolved.payload["storage"]["release"],
                selected_manifest=selected.manifest,
                cache_dir=restore_cache,
            )
        finally:
            shutil.rmtree(restore_cache, ignore_errors=True)
    resolved_path = persist_resolved_phase01_config(
        resolved, release_dir / "manifests" / "phase01" / "resolved_config.json"
    )
    transnet = materialize_transnet_artifact(
        model_config=resolved.payload["models"]["shot_detection"],
        storage_config=resolved.payload["storage"]["model_artifacts"],
        cache_root=scratch_root / "model_artifacts",
    )
    preflight = run_phase01_preflight(
        resolved,
        release_dir=release_dir,
        transnet_artifact_dir=transnet.root,
        scratch_root=scratch_root,
        validate_remote=False,
    )
    report = process_production_batch(
        release_dir=release_dir,
        config=resolved,
        scratch_root=scratch_root,
        transnet_artifact_dir=transnet.root,
        sync_release=sync_release,
    )
    last_run = output_root.resolve() / "phase01_last_run.json"
    temporary = last_run.with_name(f".{last_run.name}.partial")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "phase01_last_run_v1",
                "release_id": release_id,
                "release_dir": str(release_dir),
                "resolved_config_path": str(resolved_path),
                "worker_report_path": str(report),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(last_run)
    return Phase01RunResult(release_id, release_dir, resolved_path, report, preflight)


def _restore_phase00_if_needed(
    *,
    output_root: Path,
    release_id: str,
    batch_id: str,
    storage: dict[str, Any],
    selected_manifest: dict[str, Any],
    cache_dir: Path | str | None = None,
) -> None:
    release_dir = output_root.resolve() / release_id
    marker_path = (
        release_dir
        / "manifests"
        / "phase01"
        / f"phase00_restore_{batch_id}.json"
    )
    source_manifest_fingerprint = compute_fingerprint(selected_manifest)
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            marker = {}
        if (
            marker.get("status") == "complete"
            and marker.get("release_id") == release_id
            and marker.get("batch_id") == batch_id
            and marker.get("source_manifest_fingerprint")
            == source_manifest_fingerprint
            and _restored_phase00_files_are_valid(release_dir, marker.get("files"))
        ):
            return

    store = _hf_store(storage, cache_dir=cache_dir)
    remote_root = phase00_ingestion_remote_prefix(release_id)
    expected_checksums = {
        str(row["relative_path"]): str(row["sha256"])
        for row in selected_manifest.get("files", [])
        if isinstance(row, dict) and row.get("relative_path") and row.get("sha256")
    }
    core_files = [
        "tables/videos.parquet",
        "raw_mapping/media_store_manifest.parquet",
        f"manifests/{batch_id}.txt",
    ]
    for relative in core_files:
        _restore_phase00_file(
            store=store,
            remote_root=remote_root,
            release_dir=release_dir,
            relative_path=relative,
            expected_checksum=expected_checksums.get(relative),
        )

    batch_path = release_dir / "manifests" / f"{batch_id}.txt"
    video_ids = [
        line.strip()
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    videos = pd.read_parquet(release_dir / "tables" / "videos.parquet")
    videos_by_id = {
        str(row["video_id"]): row for row in videos.to_dict("records")
    }
    timeline_files: list[str] = []
    for video_id in video_ids:
        if video_id not in videos_by_id:
            raise RuntimeError(
                f"Phase00 batch references video missing from videos.parquet: {video_id}"
            )
        value = videos_by_id[video_id].get("frame_timeline_ref")
        relative = (
            f"frame_timeline/{video_id}.parquet"
            if not _present_phase00_ref(value)
            else str(value)
        )
        normalized = _safe_phase00_relative_path(relative)
        if not normalized.startswith("frame_timeline/") or not normalized.endswith(
            ".parquet"
        ):
            raise ValueError(f"Unsafe Phase00 frame timeline ref: {relative}")
        timeline_files.append(normalized)

    def restore_timeline(relative: str) -> None:
        _restore_phase00_file(
            store=store,
            remote_root=remote_root,
            release_dir=release_dir,
            relative_path=relative,
            expected_checksum=expected_checksums.get(relative),
        )

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(timeline_files)))) as executor:
        list(executor.map(restore_timeline, timeline_files))

    restored_files = sorted(set(core_files + timeline_files))
    marker_payload = {
        "schema_version": "phase01_phase00_batch_restore_v1",
        "status": "complete",
        "release_id": release_id,
        "batch_id": batch_id,
        "source_completed_at": selected_manifest.get("completed_at"),
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "files": [
            {
                "relative_path": relative,
                "sha256": sha256_file(release_dir / relative),
            }
            for relative in restored_files
        ],
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_name(f".{marker_path.name}.partial")
    temporary.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker_path)


def _restore_phase00_file(
    *,
    store: HuggingFaceDatasetArtifactStore,
    remote_root: str,
    release_dir: Path,
    relative_path: str,
    expected_checksum: str | None,
) -> Path:
    relative = _safe_phase00_relative_path(relative_path)
    target = release_dir / relative
    if (
        target.is_file()
        and expected_checksum
        and sha256_file(target) == expected_checksum
    ):
        return target
    store.download_file(f"{remote_root}/{relative}", target)
    if expected_checksum and sha256_file(target) != expected_checksum:
        raise ValueError(f"Phase00 restored file checksum mismatch: {relative}")
    return target


def _restored_phase00_files_are_valid(
    release_dir: Path, files: Any
) -> bool:
    if not isinstance(files, list) or not files:
        return False
    for row in files:
        if not isinstance(row, dict):
            return False
        relative = row.get("relative_path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        try:
            target = release_dir / _safe_phase00_relative_path(relative)
        except ValueError:
            return False
        if not target.is_file() or sha256_file(target) != expected:
            return False
    return True


def _safe_phase00_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe Phase00 relative path: {value}")
    return path.as_posix()


def _present_phase00_ref(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _scratch_root(storage: dict[str, Any], output_root: Path) -> Path:
    configured = storage["scratch"].get("root_override")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return (output_root.resolve().parent / "phase01_scratch").resolve()


def _hf_store(
    config: dict[str, Any],
    *,
    cache_dir: Path | str | None = None,
) -> HuggingFaceDatasetArtifactStore:
    return HuggingFaceDatasetArtifactStore(
        repo_id=str(config["repo_id"]),
        repo_type=str(config.get("repo_type", "dataset")),
        revision=str(config.get("revision", "main")),
        token=os.environ.get("AIC_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        prefix=str(config.get("prefix", "")),
        cache_dir=cache_dir,
    )
