from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from system1.artifacts.reports import utc_now


def write_manual_review_report(
    *,
    release_dir: Path,
    batch_id: str,
    worker_id: str,
    video_results: Iterable[dict[str, Any]],
    sample_size: int,
) -> Path:
    candidates: dict[str, list[dict[str, Any]]] = {
        "shot_caption": [],
        "scene_boundary": [],
        "scene_summary": [],
    }
    for result in video_results:
        if not str(result.get("status", "")).startswith("complete"):
            continue
        artifact = Path(str(result["artifact"]))
        _collect_artifact_candidates(artifact, candidates)
    selected = _stratified_sample(candidates, sample_size)
    payload = {
        "schema_version": "phase01_manual_review_v1",
        "batch_id": batch_id,
        "worker_id": worker_id,
        "status": "pending_manual_review" if selected else "not_available",
        "sample_size_requested": sample_size,
        "sample_size_actual": len(selected),
        "created_at": utc_now(),
        "instructions": (
            "Review visual/text agreement, timeline correctness, boundary consistency, "
            "and bilingual fidelity. Fill reviewer fields without changing canonical artifacts."
        ),
        "samples": selected,
    }
    path = (
        release_dir
        / "manifests"
        / "phase01"
        / f"manual_review_{batch_id}_{worker_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _collect_artifact_candidates(
    artifact: Path, candidates: dict[str, list[dict[str, Any]]]
) -> None:
    with zipfile.ZipFile(artifact) as archive:
        roots = {name.split("/", 1)[0] for name in archive.namelist() if "/" in name}
        if len(roots) != 1:
            raise ValueError(f"Unexpected Phase01 artifact root: {artifact}")
        video_id = next(iter(roots))
        keyframes = _read_parquet(archive, f"{video_id}/keyframes.parquet")
        ocr = _read_parquet_optional(archive, f"{video_id}/ocr.parquet")
        captions = _read_parquet(archive, f"{video_id}/shot_captions.parquet")
        scenes = _read_parquet(archive, f"{video_id}/scenes.parquet")
        summaries = _read_parquet(archive, f"{video_id}/scene_summaries.parquet")
        representative = {
            str(row["shot_id"]): row
            for row in keyframes.to_dict("records")
            if bool(row["is_representative"])
        }
        ocr_by_keyframe = {
            str(row.get("keyframe_id")): str(row.get("text") or row.get("raw_text") or "")
            for row in ocr.to_dict("records")
            if str(row.get("status", "")) != "failed"
        }
        for row in captions.to_dict("records"):
            frame = representative[str(row["shot_id"])]
            candidates["shot_caption"].append(
                _review_row(
                    kind="shot_caption",
                    video_id=video_id,
                    entity_id=str(row["shot_id"]),
                    artifact=artifact,
                    evidence={
                        "keyframe_ref": frame["keyframe_ref"],
                        "caption_vi": row["caption_vi"],
                        "caption_en": row["caption_en"],
                        "objects_vi": _string_values(row.get("objects_vi", [])),
                        "objects_en": _string_values(row.get("objects_en", [])),
                        "actions_vi": _string_values(row.get("actions_vi", [])),
                        "actions_en": _string_values(row.get("actions_en", [])),
                        "visible_text_summary_vi": row.get("visible_text_summary_vi", ""),
                        "visible_text_summary_en": row.get("visible_text_summary_en", ""),
                        "ocr_text": ocr_by_keyframe.get(str(frame["keyframe_id"]), ""),
                        "quality_score": float(frame["quality_score"]),
                    },
                )
            )
        summary_by_scene = {
            str(row["scene_id"]): row for row in summaries.to_dict("records")
        }
        scene_rows = scenes.sort_values("scene_index").to_dict("records")
        for row in scene_rows:
            scene_id = str(row["scene_id"])
            summary = summary_by_scene[scene_id]
            if summary.get("status") == "failed":
                continue
            candidates["scene_summary"].append(
                _review_row(
                    kind="scene_summary",
                    video_id=video_id,
                    entity_id=scene_id,
                    artifact=artifact,
                    evidence={
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "summary_vi": summary["summary_vi"],
                        "summary_en": summary["summary_en"],
                    },
                )
            )
        diagnostics_name = f"{video_id}/diagnostics/scene_boundary_diagnostics.jsonl"
        if diagnostics_name in archive.namelist():
            diagnostics = [
                json.loads(line)
                for line in archive.read(diagnostics_name).decode("utf-8").splitlines()
                if line.strip()
            ]
            for row in diagnostics:
                candidates["scene_boundary"].append(
                    _review_row(
                        kind="scene_boundary",
                        video_id=video_id,
                        entity_id=str(row["after_shot_id"]),
                        artifact=artifact,
                        evidence={
                            "is_scene_boundary": bool(row["is_boundary"]),
                            "primary_boundary_score": float(
                                row["primary_boundary_score"]
                            ),
                            "review_route": row["review_route"],
                            "reason": row.get("reason"),
                            "confidence": row.get("confidence"),
                            "evidence_used": row.get("evidence_used", []),
                        },
                    )
                )


def _read_parquet(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(archive.read(name)))


def _read_parquet_optional(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    if name not in archive.namelist():
        return pd.DataFrame()
    return _read_parquet(archive, name)


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


def _review_row(
    *, kind: str, video_id: str, entity_id: str, artifact: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "review_kind": kind,
        "video_id": video_id,
        "entity_id": entity_id,
        "artifact_path": str(artifact),
        "evidence": evidence,
        "reviewer": None,
        "reviewed_at": None,
        "decision": None,
        "notes": None,
    }


def _stratified_sample(
    candidates: dict[str, list[dict[str, Any]]], sample_size: int
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("Manual review sample size must be positive")
    kinds = tuple(candidates)
    ordered = {
        kind: sorted(rows, key=lambda row: _stable_rank(kind, row))
        for kind, rows in candidates.items()
    }
    selected: list[dict[str, Any]] = []
    cursor = {kind: 0 for kind in kinds}
    while len(selected) < sample_size:
        progressed = False
        for kind in kinds:
            index = cursor[kind]
            if index >= len(ordered[kind]):
                continue
            selected.append(ordered[kind][index])
            cursor[kind] += 1
            progressed = True
            if len(selected) == sample_size:
                break
        if not progressed:
            break
    return selected


def _stable_rank(kind: str, row: dict[str, Any]) -> str:
    identity = f"{kind}|{row['video_id']}|{row['entity_id']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
