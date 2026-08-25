from __future__ import annotations

import json
import hashlib
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from system1.artifacts.package import discover_artifact_zip, extract_artifact_zip
from system1.release.types import write_json

STRUCTURE_TABLES = [
    "asr_segments",
    "shots",
    "scenes",
    "keyframes",
    "ocr",
    "shot_captions",
    "shot_transcript_links",
    "scene_transcript_links",
    "scene_summaries",
]
FEATURE_TABLES = [
    "embeddings_meta",
    "objects",
    "text_sources",
]


def merge_worker_outputs(release_dir: Path | str) -> Path:
    """Merge worker artifacts.

    Artifact ZIPs are the primary source. Extracted per-video folders are kept
    as a local cache/debug fallback for developer workflows.
    """
    release_path = Path(release_dir)
    structure_root = release_path / "artifacts" / "structure"
    feature_root = release_path / "artifacts" / "features"
    if not structure_root.exists():
        raise FileNotFoundError(f"missing structure artifacts root: {structure_root}")
    if not feature_root.exists():
        raise FileNotFoundError(f"missing feature artifacts root: {feature_root}")

    videos_df = pd.read_parquet(release_path / "tables" / "videos.parquet")
    merged: dict[str, list[pd.DataFrame]] = {name: [] for name in STRUCTURE_TABLES + FEATURE_TABLES}
    quality_rows: list[dict[str, Any]] = []
    artifact_manifest_rows: list[dict[str, Any]] = []
    video_status_rows: list[dict[str, Any]] = []
    feature_manifests: list[dict[str, Any]] = []

    for video_id in videos_df["video_id"].astype(str).tolist():
        structure_dir = _resolve_artifact_dir_for_merge(
            release_path,
            artifact_root=structure_root,
            video_id=video_id,
            artifact_type="structure",
        )
        feature_dir = _resolve_artifact_dir_for_merge(
            release_path,
            artifact_root=feature_root,
            video_id=video_id,
            artifact_type="features",
        )
        structure_has_ocr = True
        for table in STRUCTURE_TABLES:
            path = structure_dir / f"{table}.parquet"
            if not path.exists():
                if table == "ocr":
                    structure_has_ocr = False
                    continue
                raise FileNotFoundError(f"missing structure parquet for {video_id}: {path}")
            merged[table].append(pd.read_parquet(path))
        for table in FEATURE_TABLES:
            path = feature_dir / f"{table}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing feature parquet for {video_id}: {path}")
            merged[table].append(pd.read_parquet(path))
        if not structure_has_ocr:
            feature_ocr = feature_dir / "ocr.parquet"
            if not feature_ocr.exists():
                raise FileNotFoundError(
                    f"missing OCR parquet for {video_id}: {structure_dir / 'ocr.parquet'} or {feature_ocr}"
                )
            merged["ocr"].append(pd.read_parquet(feature_ocr))
        structure_manifest = json.loads((structure_dir / "manifest.json").read_text(encoding="utf-8"))
        feature_manifest = json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        feature_manifests.append(feature_manifest)
        _materialize_runtime_media(release_path, video_id, structure_dir)
        video_status_rows.append({
            "video_id": video_id,
            "structure_status": structure_manifest.get("status", "unknown"),
            "feature_status": feature_manifest.get("status", "unknown"),
            "status": "complete" if structure_manifest.get("status") == "pass" and feature_manifest.get("status") == "pass" else "partial",
        })
        quality_rows.append({
            "video_id": video_id,
            "structure_errors": len(structure_manifest.get("errors", [])),
            "feature_errors": len(feature_manifest.get("errors", [])),
        })
        for root in (structure_dir, feature_dir):
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    artifact_manifest_rows.append({
                        "video_id": video_id,
                        "artifact_path": str(path.relative_to(release_path)),
                        "artifact_type": path.suffix.lstrip(".") or "file",
                        "phase": "structure" if root == structure_dir else "features",
                    })

    tables_dir = release_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table_name, frames in merged.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined.to_parquet(tables_dir / f"{table_name}.parquet", index=False)
        counts[table_name] = len(combined)

    structure_text_sources = _build_structure_text_sources(
        pd.concat(merged["asr_segments"], ignore_index=True),
        pd.concat(merged["ocr"], ignore_index=True),
        pd.concat(merged["shot_captions"], ignore_index=True),
        pd.concat(merged["scene_summaries"], ignore_index=True),
    )
    feature_text_sources = pd.concat(merged["text_sources"], ignore_index=True)
    text_sources = pd.concat([feature_text_sources, structure_text_sources], ignore_index=True)
    text_sources.to_parquet(tables_dir / "text_sources.parquet", index=False)
    counts["text_sources"] = len(text_sources)

    text_documents = _build_text_documents(text_sources)
    text_documents.to_parquet(tables_dir / "text_documents.parquet", index=False)
    counts["text_documents"] = len(text_documents)

    feature_availability = _build_feature_availability(
        pd.concat(merged["keyframes"], ignore_index=True),
        pd.concat(merged["asr_segments"], ignore_index=True),
        pd.concat(merged["ocr"], ignore_index=True),
        pd.concat(merged["shot_captions"], ignore_index=True),
        pd.concat(merged["embeddings_meta"], ignore_index=True),
    )
    feature_availability.to_parquet(tables_dir / "feature_availability.parquet", index=False)
    counts["feature_availability"] = len(feature_availability)

    release_capabilities = _build_release_capabilities(
        feature_availability,
        counts,
        feature_manifests,
        pd.concat(merged["asr_segments"], ignore_index=True),
        pd.concat(merged["ocr"], ignore_index=True),
    )
    release_capabilities.to_parquet(tables_dir / "release_capabilities.parquet", index=False)
    capabilities = {str(row["capability"]): str(row["status"]) for row in release_capabilities.to_dict("records")}

    manifests_dir = release_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(artifact_manifest_rows).to_parquet(manifests_dir / "artifact_manifest.parquet", index=False)
    pd.DataFrame(video_status_rows).to_parquet(manifests_dir / "video_processing_status.parquet", index=False)
    write_json(manifests_dir / "quality_report.json", {"videos": quality_rows, "status": "pass"})
    write_json(
        manifests_dir / "dataset_manifest.json",
        {
            "release_id": release_path.name,
            "counts": counts,
            "app_sqlite": "db/app.sqlite",
            "fts5": "app.sqlite:text_documents_fts",
            "visual_index": "indexes/visual.faiss",
            "vector_map": "indexes/vector_map.parquet",
            "capabilities": capabilities,
            "release_usable": False,
        },
    )
    report_path = manifests_dir / "merge_report.json"
    write_json(report_path, {"status": "pass", "video_count": len(video_status_rows), "counts": counts})
    return report_path


def _build_structure_text_sources(
    asr: pd.DataFrame,
    ocr: pd.DataFrame,
    shot_captions: pd.DataFrame,
    scene_summaries: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in asr.to_dict("records"):
        rows.append(_text_source(
            str(row.get("video_id", "")),
            "video",
            str(row.get("video_id", "")),
            "asr",
            str(row.get("text", "")),
            str(row.get("provider", row.get("asr_provider", "asr"))),
            str(row.get("status", "pass")),
            str(row.get("language", "und")),
        ))
    for row in ocr.to_dict("records"):
        text = str(row.get("text", row.get("raw_text", ""))).strip()
        if not text:
            continue
        rows.append(_text_source(
            str(row.get("video_id", "")),
            "keyframe",
            str(row.get("keyframe_id", "")),
            "ocr",
            text,
            str(row.get("provider", "ocr")),
            str(row.get("status", "pass")),
            str(row.get("language", "vi")),
        ))
    for row in shot_captions.to_dict("records"):
        for language, column in (("vi", "caption_vi"), ("en", "caption_en")):
            rows.append(_text_source(
                str(row.get("video_id", "")),
                "shot",
                str(row.get("shot_id", "")),
                "shot_caption",
                str(row.get(column, "")),
                str(row.get("provider", row.get("caption_model", "shot_caption"))),
                str(row.get("status", "pass")),
                language,
            ))
        for language, column in (
            ("vi", "visible_text_summary_vi"),
            ("en", "visible_text_summary_en"),
        ):
            text = str(row.get(column, "")).strip()
            if text:
                rows.append(_text_source(
                    str(row.get("video_id", "")),
                    "shot",
                    str(row.get("shot_id", "")),
                    "visible_text_summary",
                    text,
                    str(row.get("provider", row.get("caption_model", "shot_caption"))),
                    str(row.get("status", "pass")),
                    language,
                ))
        for language, columns in (
            ("vi", ("objects_vi", "actions_vi")),
            ("en", ("objects_en", "actions_en")),
        ):
            text = " ".join(
                item
                for column in columns
                for item in _string_values(row.get(column, []))
            )
            if text:
                rows.append(_text_source(
                    str(row.get("video_id", "")),
                    "shot",
                    str(row.get("shot_id", "")),
                    "shot_visual_terms",
                    text,
                    str(row.get("provider", row.get("caption_model", "shot_caption"))),
                    str(row.get("status", "pass")),
                    language,
                ))
    for row in scene_summaries.to_dict("records"):
        for language, column in (("vi", "summary_vi"), ("en", "summary_en")):
            text = str(row.get(column, "")).strip()
            if not text:
                continue
            rows.append(_text_source(
                str(row.get("video_id", "")),
                "scene",
                str(row.get("scene_id", "")),
                "scene_summary",
                text,
                str(row.get("provider", row.get("model_name", "scene_summary"))),
                str(row.get("status", "pass")),
                language,
            ))
    return pd.DataFrame(rows)


def _text_source(
    video_id: str,
    entity_type: str,
    entity_id: str,
    source_type: str,
    raw_text: str,
    provider: str,
    status: str,
    language: str,
) -> dict[str, Any]:
    normalized_text = raw_text or ""
    normalized_no_diacritics = "".join(
        char for char in unicodedata.normalize("NFD", normalized_text) if unicodedata.category(char) != "Mn"
    )
    digest = hashlib.sha256(
        f"{entity_type}|{entity_id}|{source_type}|{language}|{provider}|{raw_text}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "source_id": f"{video_id}:{entity_type}:{source_type}:{provider}:{digest}",
        "video_id": video_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source_type": source_type,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "normalized_no_diacritics": normalized_no_diacritics,
        "language": language,
        "provider": provider,
        "status": status,
    }


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _resolve_artifact_dir_for_merge(
    release_path: Path,
    *,
    artifact_root: Path,
    video_id: str,
    artifact_type: str,
) -> Path:
    zip_path = discover_artifact_zip(artifact_root, video_id=video_id, artifact_type=artifact_type)
    if zip_path is not None:
        return extract_artifact_zip(
            zip_path,
            release_path / "staging" / "extracted_artifacts" / "merge" / artifact_type,
            expected_video_id=video_id,
            expected_artifact_type=artifact_type,
        )

    local_dir = artifact_root / video_id
    if local_dir.exists():
        return local_dir
    raise FileNotFoundError(f"missing {artifact_type} artifact zip or folder for video_id={video_id}: {artifact_root}")


def _build_text_documents(text_sources: pd.DataFrame) -> pd.DataFrame:
    grouped_rows = []
    for (video_id, entity_type, entity_id), group in text_sources.groupby(["video_id", "entity_type", "entity_id"], dropna=False):
        normalized_text = "\n".join(str(value) for value in group["normalized_text"].fillna("") if str(value))
        source_types = sorted(set(str(value) for value in group["source_type"].fillna("") if str(value)))
        normalized_no_diacritics = "".join(
            ch for ch in unicodedata.normalize("NFD", normalized_text) if unicodedata.category(ch) != "Mn"
        )
        document_id = f"doc:{video_id}:{entity_type}:{entity_id}"
        languages = sorted(set(str(value) for value in group["language"].dropna() if str(value)))
        grouped_rows.append({
            "doc_id": document_id,
            "document_id": document_id,
            "video_id": video_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "source_types": ",".join(source_types),
            "level": entity_type,
            "normalized_text": normalized_text,
            "normalized_no_diacritics": normalized_no_diacritics,
            "language": languages[0] if len(languages) == 1 else ("mul" if languages else "und"),
            "text": normalized_text,
        })
    return pd.DataFrame(grouped_rows)


def _build_feature_availability(keyframes: pd.DataFrame, asr: pd.DataFrame, ocr: pd.DataFrame, shot_captions: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    asr_video_ids = set(asr["video_id"].astype(str)) if not asr.empty else set()
    ocr_keyframes = set(ocr[ocr["status"] != "failed"]["keyframe_id"].astype(str)) if not ocr.empty else set()
    caption_shots = set(shot_captions[shot_captions["status"] != "failed"]["shot_id"].astype(str)) if not shot_captions.empty else set()
    embedding_keyframes = set(embeddings[embeddings["status"] != "failed"]["keyframe_id"].astype(str)) if not embeddings.empty else set()
    for row in keyframes.to_dict("records"):
        keyframe_id = str(row["keyframe_id"])
        video_id = str(row["video_id"])
        shot_id = str(row.get("shot_id", ""))
        has_embedding = keyframe_id in embedding_keyframes
        has_ocr = keyframe_id in ocr_keyframes
        has_caption = shot_id in caption_shots
        has_asr = video_id in asr_video_ids
        status = "pass" if all([has_embedding, has_caption]) else "degraded"
        rows.append({
            "entity_type": "keyframe",
            "entity_id": keyframe_id,
            "video_id": video_id,
            "has_embedding": has_embedding,
            "has_ocr": has_ocr,
            "has_caption": has_caption,
            "has_asr": has_asr,
            "status": status,
        })
    return pd.DataFrame(rows)


def _build_release_capabilities(
    feature_availability: pd.DataFrame,
    counts: dict[str, int],
    feature_manifests: list[dict[str, Any]],
    asr: pd.DataFrame,
    ocr: pd.DataFrame,
) -> pd.DataFrame:
    has_embeddings = bool(counts.get("embeddings_meta"))
    has_text = bool(counts.get("text_documents"))
    has_context = bool(counts.get("keyframes")) and bool(counts.get("shots")) and bool(counts.get("scenes"))
    provider_plan = _first_manifest_value(feature_manifests, "provider_plan", {})
    availability_complete = bool(
        not feature_availability.empty
        and (feature_availability["status"] == "pass").all()
    )
    uses_mock_provider = bool(
        isinstance(provider_plan, dict)
        and any(str(provider) == "mock" for provider in provider_plan.values())
    )
    enrichment_status = (
        "pass" if availability_complete and not uses_mock_provider else "degraded"
    )
    asr_provider = _first_provider(asr, "provider") or (
        str(provider_plan.get("asr", "mock")) if isinstance(provider_plan, dict) else "mock"
    )
    ocr_provider = _first_provider(ocr, "provider") or (
        str(provider_plan.get("ocr", "mock")) if isinstance(provider_plan, dict) else "mock"
    )
    asr_status = _provider_capability_status(
        asr,
        provider=asr_provider,
        row_count_optional=True,
        implemented_providers={"faster_whisper", "nemo"},
    )
    ocr_status = _provider_capability_status(
        ocr,
        provider=ocr_provider,
        row_count_optional=False,
        implemented_providers={"vintern_local", "qwen_local", "gemini"},
    )
    rows = [
        {"capability": "core_runtime", "status": "pass", "reason": "merged release tables available"},
        {"capability": "visual_search", "status": "degraded" if has_embeddings else "fail", "reason": "index built later"},
        {"capability": "text_search", "status": "pass" if has_text else "fail", "reason": "text_documents merged"},
        {"capability": "inspection_context", "status": "pass" if has_context else "fail", "reason": "structure tables merged"},
        {
            "capability": "asr",
            "status": asr_status,
            "reason": _capability_reason(asr_provider, asr_status, "ASR"),
        },
        {
            "capability": "ocr",
            "status": ocr_status,
            "reason": _capability_reason(ocr_provider, ocr_status, "OCR"),
        },
        {"capability": "enrichment_overall", "status": enrichment_status, "reason": "feature availability merged"},
        {
            "capability": "incremental_reuse",
            "status": "degraded",
            "reason": "batch checkpoints exist; per-video content-addressed reuse is not implemented",
        },
    ]
    return pd.DataFrame(rows)


def _first_provider(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    for value in frame[column].dropna().astype(str):
        provider = value.strip()
        if provider:
            return provider
    return None


def _provider_capability_status(
    frame: pd.DataFrame,
    *,
    provider: str,
    row_count_optional: bool,
    implemented_providers: set[str],
) -> str:
    if provider in {"mock", "unconfigured", "unavailable"}:
        return "degraded"
    if provider not in implemented_providers:
        return "degraded"
    if frame.empty:
        return "pass" if row_count_optional else "degraded"
    if "status" not in frame.columns:
        return "degraded"
    statuses = {str(value) for value in frame["status"].dropna()}
    return "degraded" if "failed" in statuses or not statuses else "pass"


def _capability_reason(provider: str, status: str, label: str) -> str:
    if status == "pass":
        return f"{provider} {label} provider emitted schema-valid rows"
    if provider == "mock":
        return f"mock empty {label} provider"
    return f"{provider} {label} provider emitted incomplete or failed rows"


def _first_manifest_value(feature_manifests: list[dict[str, Any]], key: str, default: Any) -> Any:
    for manifest in feature_manifests:
        value = manifest.get(key)
        if value is not None:
            return value
    return default


def _materialize_runtime_media(release_path: Path, video_id: str, structure_dir: Path) -> None:
    target_keyframes = release_path / "media" / "keyframes" / video_id
    target_thumbnails = release_path / "media" / "thumbnails" / video_id
    target_keyframes.mkdir(parents=True, exist_ok=True)
    target_thumbnails.mkdir(parents=True, exist_ok=True)
    for source in sorted((structure_dir / "keyframes").glob("*.jpg")):
        shutil.copy2(source, target_keyframes / source.name)
    for source in sorted((structure_dir / "thumbnails").glob("*.webp")):
        shutil.copy2(source, target_thumbnails / source.name)
