from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from system1.artifacts.hf_store import HuggingFaceDatasetArtifactStore
from system1.artifacts.reports import utc_now
from system1.config import ResolvedPhase01Config, require_phase01_production_ready
from system1.shots import load_transnet_artifact


@dataclass(frozen=True)
class PreflightResult:
    environment: str
    release_id: str
    batch_id: str
    cuda_available: bool
    scratch_free_gb: float
    model_cache_free_gb: float | None
    versions: dict[str, str]


def run_phase01_preflight(
    config: ResolvedPhase01Config,
    *,
    release_dir: Path,
    transnet_artifact_dir: Path,
    scratch_root: Path,
    validate_remote: bool = True,
) -> PreflightResult:
    require_phase01_production_ready(config)
    runtime = config.payload["runtime"]
    if config.payload["phase01"]["api"].get("request_cache_backend") != "stage_local":
        raise RuntimeError("Phase01 Gemini request cache backend must be stage_local")
    batch_path = release_dir / "manifests" / f"{runtime['batch_id']}.txt"
    required = [
        release_dir / "tables" / "videos.parquet",
        release_dir / "raw_mapping" / "media_store_manifest.parquet",
        batch_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase00 local handoff is incomplete: " + ", ".join(missing))
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is unavailable: {executable}")
    models = config.payload["models"]
    if _requires_gemini(models) and not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    if not os.environ.get("AIC_HF_TOKEN") and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("AIC_HF_TOKEN or HF_TOKEN is required")
    _validate_phase00_batch(release_dir, batch_path)
    _validate_prompt_files(config)

    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_free_gb = shutil.disk_usage(scratch_root).free / (1024**3)
    required_free_gb = float(config.payload["phase01"]["execution"]["min_scratch_free_gb"])
    if scratch_free_gb < required_free_gb:
        raise RuntimeError(
            f"Scratch free space is too low: {scratch_free_gb:.2f} GiB "
            f"< {required_free_gb:.2f} GiB"
        )
    model_cache_free_gb = _validate_model_cache_free_space(config, models)

    shot_model = config.payload["models"]["shot_detection"]
    load_transnet_artifact(
        transnet_artifact_dir,
        expected_commit=str(shot_model["model_revision"]),
        expected_source_sha256=str(shot_model["source_sha256"]),
        expected_weights_sha256=str(shot_model["weights_sha256"]),
        expected_conversion_verified=bool(shot_model.get("conversion_verified", True)),
    )
    if validate_remote:
        storage_cache = scratch_root / ".hf_cache" / f"storage_preflight-{os.getpid()}"
        try:
            run_phase01_storage_preflight(config, cache_dir=storage_cache)
        finally:
            shutil.rmtree(storage_cache, ignore_errors=True)

    versions: dict[str, str] = {}
    for package in (
        "bitsandbytes",
        "faster-whisper",
        "google-genai",
        "huggingface-hub",
        "nemo-toolkit",
        "onnx",
        "opencv-python-headless",
        "pillow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    cuda_available = False
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        versions["torch"] = "missing"
    expected = config.payload["models"]
    asr_model = expected["asr"]
    asr_provider = str(asr_model.get("provider", "nemo"))
    if asr_provider == "faster_whisper":
        if versions["faster-whisper"] != str(asr_model["package_version"]):
            raise RuntimeError("Installed faster-whisper version differs from resolved config")
    elif asr_provider == "nemo":
        if versions["nemo-toolkit"] != str(asr_model["package_version"]):
            raise RuntimeError("Installed nemo-toolkit version differs from resolved config")
        try:
            importlib.import_module("nemo.collections.asr")
        except (ImportError, AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "nemo_toolkit[asr] could not initialize for configured NeMo ASR"
            ) from exc
    else:
        raise RuntimeError(f"Unsupported Phase01 ASR provider: {asr_provider}")
    semantic_models = [expected.get(key, {}) for key in ("shot_caption", "scene_boundary", "scene_summary")]
    gemini_versions = [
        model.get("sdk_version")
        for configured in semantic_models
        for model in [configured, *configured.get("fallbacks", [])]
        if model.get("provider") == "gemini" and model.get("sdk_version") is not None
    ]
    if gemini_versions and versions["google-genai"] not in {str(value) for value in gemini_versions}:
        raise RuntimeError("Installed google-genai version differs from resolved config")
    _validate_local_vlm_dependencies(expected, versions=versions)
    if versions["torch"] == "missing":
        raise RuntimeError("PyTorch is required for TransNet V2")
    return PreflightResult(
        environment=str(runtime["environment"]),
        release_id=str(runtime["release_id"]),
        batch_id=str(runtime["batch_id"]),
        cuda_available=cuda_available,
        scratch_free_gb=scratch_free_gb,
        model_cache_free_gb=model_cache_free_gb,
        versions=versions,
    )


def run_phase01_storage_preflight(
    config: ResolvedPhase01Config,
    *,
    cache_dir: Path | str | None = None,
) -> None:
    """Prove release access and checkpoint write/read before heavy work."""

    storage = config.payload["storage"]
    token = os.environ.get("AIC_HF_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    release = storage["release"]
    checkpoint = storage["checkpoint"]
    api.repo_info(
        repo_id=release["repo_id"],
        repo_type=release["repo_type"],
        revision=release.get("revision", "main"),
    )
    checkpoint_info = api.repo_info(
        repo_id=checkpoint["repo_id"],
        repo_type=checkpoint["repo_type"],
        revision=checkpoint.get("revision", "main"),
    )
    if checkpoint.get("require_private") and not checkpoint_info.private:
        raise RuntimeError(
            "Phase01 checkpoint repository is public but require_private=true"
        )

    store = HuggingFaceDatasetArtifactStore(
        repo_id=str(checkpoint["repo_id"]),
        repo_type=str(checkpoint.get("repo_type", "dataset")),
        revision=str(checkpoint.get("revision", "main")),
        token=token,
        prefix=str(checkpoint.get("prefix", "")),
        cache_dir=cache_dir,
    )
    runtime = config.payload["runtime"]
    proof_path = (
        Path("phase01_checkpoints")
        / "_preflight"
        / f"{runtime['release_id']}_{runtime['worker_id']}.json"
    )
    payload = {
        "schema_version": "phase01_preflight_write_v1",
        "release_id": runtime["release_id"],
        "worker_id": runtime["worker_id"],
        "config_hash": config.config_hash,
        "checked_at": utc_now(),
    }
    store.write_json(proof_path, payload)
    restored = store.read_json(proof_path)
    if json.dumps(restored, sort_keys=True) != json.dumps(payload, sort_keys=True):
        raise RuntimeError("Checkpoint repository write/read proof did not round-trip")


def _validate_phase00_batch(release_dir: Path, batch_path: Path) -> None:
    import pandas as pd

    video_ids = [
        line.strip()
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not video_ids:
        raise RuntimeError(f"Phase00 batch manifest is empty: {batch_path}")
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError(f"Phase00 batch manifest contains duplicate video IDs: {batch_path}")
    videos = pd.read_parquet(release_dir / "tables" / "videos.parquet")
    media = pd.read_parquet(release_dir / "raw_mapping" / "media_store_manifest.parquet")
    duplicate_videos = sorted(
        videos.loc[videos["video_id"].astype(str).duplicated(), "video_id"].astype(str).unique()
    )
    duplicate_media = sorted(
        media.loc[media["video_id"].astype(str).duplicated(), "video_id"].astype(str).unique()
    )
    if duplicate_videos or duplicate_media:
        raise RuntimeError(
            "Phase00 handoff contains duplicate IDs: "
            f"videos={duplicate_videos[:10]}, media={duplicate_media[:10]}"
        )
    missing_videos = sorted(set(video_ids) - set(videos["video_id"].astype(str)))
    missing_media = sorted(set(video_ids) - set(media["video_id"].astype(str)))
    video_rows = {
        str(row["video_id"]): row for row in videos.to_dict("records")
    }
    missing_timelines = []
    for video_id in video_ids:
        row = video_rows.get(video_id, {})
        ref = row.get("frame_timeline_ref")
        if ref is None or pd.isna(ref) or not str(ref).strip():
            ref = f"frame_timeline/{video_id}.parquet"
        if not (release_dir / str(ref)).is_file():
            missing_timelines.append(video_id)
    if missing_videos or missing_media or missing_timelines:
        raise RuntimeError(
            "Phase00 batch handoff is inconsistent: "
            f"missing_videos={missing_videos[:10]}, "
            f"missing_media={missing_media[:10]}, "
            f"missing_timelines={missing_timelines[:10]}"
        )
    canonical_required = {
        "canonical_repo_id",
        "canonical_repo_type",
        "canonical_revision",
        "canonical_prefix",
        "canonical_video_path",
        "canonical_metadata_path",
    }
    missing_columns = sorted(canonical_required - set(media.columns))
    if missing_columns:
        raise RuntimeError(
            "Phase00 media_store_manifest is missing canonical columns: "
            + ", ".join(missing_columns)
        )
    selected_media = media[media["video_id"].astype(str).isin(video_ids)]
    null_canonical: dict[str, list[str]] = {}
    for column in sorted(canonical_required):
        missing_mask = selected_media[column].isna() | (
            selected_media[column].astype(str).str.strip() == ""
        )
        if missing_mask.any():
            null_canonical[column] = sorted(
                selected_media.loc[missing_mask, "video_id"].astype(str).tolist()
            )[:10]
    if null_canonical:
        raise RuntimeError(
            f"Phase00 media_store_manifest has null canonical values: {null_canonical}"
        )


def _validate_prompt_files(config: ResolvedPhase01Config) -> None:
    models = config.payload["models"]
    versions = {
        str(config.payload["phase01"]["api"]["schema_repair_prompt_version"]),
        str(models["ocr"]["prompt_version"]),
        str(models["shot_caption"]["prompt_version"]),
        str(models["scene_boundary"]["prompt_version"]),
        str(models["scene_boundary"]["focused_prompt_version"]),
        str(models["scene_boundary"]["consistency_prompt_version"]),
        str(models["scene_summary"]["prompt_version"]),
    }
    prompt_root = Path(__file__).resolve().parents[3] / "prompts"
    missing = []
    for version in sorted(versions):
        if Path(version).name != version:
            raise RuntimeError(f"Unsafe Phase01 prompt version: {version}")
        path = prompt_root / f"{version}.txt"
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            missing.append(str(path))
    if missing:
        raise RuntimeError("Phase01 prompt files are missing or empty: " + ", ".join(missing))


def _requires_gemini(models: dict[str, Any]) -> bool:
    return any(
        str(models.get(key, {}).get("provider")) == "gemini"
        for key in ("shot_caption", "scene_boundary", "scene_summary")
    )


def _validate_local_vlm_dependencies(
    models: dict[str, Any], *, versions: dict[str, str]
) -> None:
    local_models = [
        models.get("ocr", {}),
        models.get("shot_caption", {}),
        *models.get("shot_caption", {}).get("fallbacks", []),
    ]
    if not any(str(model.get("provider")) in {"qwen_local", "vintern_local"} for model in local_models):
        return
    required_modules = {
        "transformers": "transformers",
        "accelerate": "accelerate",
        "torch": "torch",
    }
    if any(str(model.get("provider")) == "qwen_local" for model in local_models):
        required_modules["qwen_vl_utils"] = "qwen-vl-utils"
        required_modules["bitsandbytes"] = "bitsandbytes"
    missing = [
        package
        for module, package in required_modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Local VLM dependencies are missing: "
            + ", ".join(sorted(missing))
            + ". Install system1[phase01-production]."
        )
    qwen_models = [
        model for model in local_models if str(model.get("provider")) == "qwen_local"
    ]
    if not qwen_models:
        return
    configured_versions = {
        str(model.get("quantization", {}).get("package_version", ""))
        for model in qwen_models
    }
    configured_versions.discard("")
    if configured_versions and versions["bitsandbytes"] not in configured_versions:
        raise RuntimeError(
            "Installed bitsandbytes version differs from resolved Qwen quantization config"
        )
    try:
        torch = importlib.import_module("torch")
        cextension = importlib.import_module("bitsandbytes.cextension")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("bitsandbytes could not initialize for Qwen 4-bit") from exc
    if bool(torch.cuda.is_available()) and not bool(
        getattr(getattr(cextension, "lib", None), "compiled_with_cuda", False)
    ):
        raise RuntimeError(
            "bitsandbytes has no CUDA native backend for the installed PyTorch CUDA build"
        )


def _validate_model_cache_free_space(
    config: ResolvedPhase01Config, models: dict[str, Any]
) -> float | None:
    if not _uses_local_vlm(models):
        return None
    required_free_gb = float(
        config.payload["phase01"]["execution"].get("min_model_cache_free_gb", 0)
    )
    if required_free_gb <= 0:
        return None
    cache_root = _hf_model_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache_root).free / (1024**3)
    if free_gb < required_free_gb:
        raise RuntimeError(
            f"Model cache free space is too low: {free_gb:.2f} GiB "
            f"< {required_free_gb:.2f} GiB. Set HF_HOME to a larger runtime disk."
        )
    return free_gb


def _uses_local_vlm(models: dict[str, Any]) -> bool:
    local_models = [
        models.get("ocr", {}),
        models.get("shot_caption", {}),
        *models.get("shot_caption", {}).get("fallbacks", []),
    ]
    return any(
        str(model.get("provider")) in {"qwen_local", "vintern_local"}
        for model in local_models
    )


def _hf_model_cache_root() -> Path:
    explicit = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path("~/.cache/huggingface").expanduser()
