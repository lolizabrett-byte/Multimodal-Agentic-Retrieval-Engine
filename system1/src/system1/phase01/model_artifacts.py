from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from system1.artifacts.hf_store import HuggingFaceDatasetArtifactStore
from system1.shots import TransNetArtifact, load_transnet_artifact


def materialize_transnet_artifact(
    *,
    model_config: Mapping[str, Any],
    storage_config: Mapping[str, Any],
    cache_root: Path,
) -> TransNetArtifact:
    """Restore and validate the pinned, project-owned TransNet bundle."""

    expected_commit = str(model_config["model_revision"])
    expected_source_sha256 = str(model_config["source_sha256"])
    expected_weights_sha256 = str(model_config["weights_sha256"])
    expected_conversion_verified = bool(model_config.get("conversion_verified", True))
    artifact_subdir = str(
        model_config.get("artifact_subdir") or f"transnetv2/{expected_commit}"
    ).strip("/")
    target = cache_root / artifact_subdir
    if target.is_dir():
        try:
            return load_transnet_artifact(
                target,
                expected_commit=expected_commit,
                expected_source_sha256=expected_source_sha256,
                expected_weights_sha256=expected_weights_sha256,
                expected_conversion_verified=expected_conversion_verified,
            )
        except (FileNotFoundError, ValueError):
            shutil.rmtree(target)

    download_cache = cache_root / f".hf_download_cache-{os.getpid()}"
    store = HuggingFaceDatasetArtifactStore(
        repo_id=str(storage_config["repo_id"]),
        repo_type=str(storage_config.get("repo_type", "dataset")),
        revision=str(storage_config.get("revision", "main")),
        token=os.environ.get("AIC_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        prefix=str(storage_config.get("prefix", "")),
        cache_dir=download_cache,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".transnet_restore_", dir=cache_root) as tmp:
            staged = Path(tmp) / "artifact"
            staged.mkdir()
            manifest_path = store.download_file(
                f"{artifact_subdir}/manifest.json", staged / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("source_file", "weights_file"):
                filename = str(manifest.get(key, ""))
                if not filename or Path(filename).name != filename:
                    raise ValueError(f"Unsafe or missing TransNet manifest field: {key}")
                store.download_file(f"{artifact_subdir}/{filename}", staged / filename)
            load_transnet_artifact(
                staged,
                expected_commit=expected_commit,
                expected_source_sha256=expected_source_sha256,
                expected_weights_sha256=expected_weights_sha256,
                expected_conversion_verified=expected_conversion_verified,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(staged, target)
            except OSError:
                # Another worker on this machine finished the same download while
                # this one was still fetching. os.replace refuses a non-empty
                # directory, and the bundle is content-addressed by commit, so
                # whatever landed there is the same bundle — validated below.
                if not target.is_dir():
                    raise
    finally:
        shutil.rmtree(download_cache, ignore_errors=True)
    return load_transnet_artifact(
        target,
        expected_commit=expected_commit,
        expected_source_sha256=expected_source_sha256,
        expected_weights_sha256=expected_weights_sha256,
        expected_conversion_verified=expected_conversion_verified,
    )
