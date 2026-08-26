from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from system1.gemini import StructuredRequest
from system1.media.contact_sheet import write_contact_sheet
from system1.vlm import StructuredClient


class StructuredSceneBoundaryJudge:
    def __init__(
        self,
        client: StructuredClient,
        *,
        video_id: str,
        prompt_dir: Path,
        diagnostics_dir: Path,
        model_config: Mapping[str, Any],
    ) -> None:
        self.client = client
        self.video_id = video_id
        self.prompt_dir = prompt_dir
        self.diagnostics_dir = diagnostics_dir
        self.model_config = model_config
        self.request_index = 0
        self._diagnostics: dict[str, dict[str, Any]] = {}

    def judge(
        self,
        *,
        request_kind: str,
        focus_gap_ids: tuple[str, ...],
        context: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, bool]:
        prompt_version = {
            "primary": self.model_config["prompt_version"],
            "focused_review": self.model_config["focused_prompt_version"],
            "consistency_review": self.model_config["consistency_prompt_version"],
        }[request_kind]
        base_prompt = (self.prompt_dir / f"{prompt_version}.txt").read_text(encoding="utf-8")
        evidence_payload = [_json_safe_evidence(item) for item in context]
        prompt = (
            base_prompt
            + "\n\nFOCUS GAPS:\n"
            + json.dumps(focus_gap_ids, ensure_ascii=False)
            + "\n\nORDERED SHOT EVIDENCE:\n"
            + json.dumps(evidence_payload, ensure_ascii=False)
        )
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet = self.diagnostics_dir / f"{self.request_index:05d}_{request_kind}.jpg"
        _write_contact_sheet(context, contact_sheet)
        image_paths = [contact_sheet]
        if request_kind != "primary":
            role_sheet = (
                self.diagnostics_dir
                / f"{self.request_index:05d}_{request_kind}_early_late.jpg"
            )
            # Vintern chấp nhận đúng 1 ảnh mỗi request — chỉ gửi sheet vai
            # trò khi có, thay cho sheet chính, thay vì gửi cả hai.
            if _write_role_contact_sheet(context, focus_gap_ids, role_sheet):
                image_paths = [role_sheet]
        response_schema = {
            "type": "object",
            "properties": {
                "boundaries": {
                    "type": "array",
                    "minItems": len(focus_gap_ids),
                    "maxItems": len(focus_gap_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            # A position, not a shot id. The model kept inventing
                            # ids from other videos; an integer it cannot spell
                            # wrong is checkable arithmetically instead.
                            "gap_index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": len(focus_gap_ids) - 1,
                            },
                            "is_scene_boundary": {"type": "boolean"},
                            "reason": {"type": "string"},
                            "confidence": {"type": ["number", "null"]},
                            "evidence_used": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["gap_index", "is_scene_boundary"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["boundaries"],
            "additionalProperties": False,
        }
        response = self.client.request(
            StructuredRequest(
                request_kind=f"scene_boundary_{request_kind}",
                video_id=self.video_id,
                prompt=prompt,
                prompt_version=prompt_version,
                response_schema_version=str(self.model_config["response_schema_version"]),
                response_schema=response_schema,
                image_paths=tuple(image_paths),
                identity={"focus_gap_ids": focus_gap_ids},
            )
        )
        self.request_index += 1
        boundaries = response["boundaries"]
        result: dict[str, bool] = {}
        for item in boundaries:
            index = int(item["gap_index"])
            if not 0 <= index < len(focus_gap_ids):
                raise ValueError(
                    f"Structured judge returned out-of-range gap_index: {index}"
                )
            gap_id = focus_gap_ids[index]
            if gap_id in result:
                raise ValueError(f"Structured judge duplicated scene gap: {gap_id}")
            result[gap_id] = item["is_scene_boundary"]
            self._diagnostics[gap_id] = {
                "reason": str(item.get("reason", "")),
                "confidence": item.get("confidence"),
                "evidence_used": item.get("evidence_used", []),
            }
        if len(result) != len(focus_gap_ids):
            raise ValueError(
                "Structured judge did not cover every requested gap: "
                f"expected {len(focus_gap_ids)}, got {len(result)}"
            )
        return result

    def diagnostics_for(self, gap_id: str) -> Mapping[str, Any]:
        return self._diagnostics.get(gap_id, {})


def _write_contact_sheet(context: Sequence[Mapping[str, Any]], output: Path) -> None:
    tiles = [
        (
            Path(str(item["representative_path"])),
            f"{item['shot_id']} {float(item['start_sec']):.2f}-{float(item['end_sec']):.2f}s",
        )
        for item in context
    ]
    write_contact_sheet(tiles, output)


def _write_role_contact_sheet(
    context: Sequence[Mapping[str, Any]],
    focus_gap_ids: tuple[str, ...],
    output: Path,
) -> bool:
    relevant_ids = set(focus_gap_ids)
    for previous, current in pairwise(context):
        if str(previous["shot_id"]) in relevant_ids:
            relevant_ids.add(str(current["shot_id"]))
    tiles: list[tuple[str, str, Path]] = []
    for item in context:
        shot_id = str(item["shot_id"])
        if shot_id not in relevant_ids:
            continue
        for role in ("early", "late"):
            value = item.get(f"{role}_path")
            if value:
                tiles.append((shot_id, role, Path(str(value))))
    if not tiles:
        return False
    labeled_tiles = [
        (image_path, f"{shot_id} {role}") for shot_id, role, image_path in tiles
    ]
    write_contact_sheet(labeled_tiles, output)
    return True


def _json_safe_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shot_id": str(item["shot_id"]),
        "start_sec": float(item["start_sec"]),
        "end_sec": float(item["end_sec"]),
        "caption_vi": str(item.get("caption_vi", "")),
        "caption_en": str(item.get("caption_en", "")),
        "objects_vi": _string_list(item.get("objects_vi", [])),
        "objects_en": _string_list(item.get("objects_en", [])),
        "actions_vi": _string_list(item.get("actions_vi", [])),
        "actions_en": _string_list(item.get("actions_en", [])),
        "visible_text_summary_vi": str(item.get("visible_text_summary_vi", "")),
        "visible_text_summary_en": str(item.get("visible_text_summary_en", "")),
        "ocr_text": _string_list(item.get("ocr_text", [])),
        "transcript": str(item.get("transcript", "")),
        "has_early_frame": bool(item.get("early_path")),
        "has_late_frame": bool(item.get("late_path")),
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []


# Compatibility for existing imports while production call sites use the
# provider-neutral name.
GeminiSceneBoundaryJudge = StructuredSceneBoundaryJudge
