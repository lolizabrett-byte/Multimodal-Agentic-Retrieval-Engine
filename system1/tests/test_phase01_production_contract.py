from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from system1.cli import app
from system1.config import (
    load_configs,
    persist_resolved_phase01_config,
    require_phase01_production_ready,
    resolve_phase01_config,
)
from system1.config.loader import _stage_config_hashes
from system1.phase01.production import (
    PARQUET_COLUMNS,
    _assemble_package,
    _build_captions,
    _required_text,
)
from system1.phase01.validation import validate_rows

SYSTEM1_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = SYSTEM1_ROOT / "configs"
SCHEMA_DIR = SYSTEM1_ROOT / "schemas"


def user_settings(**overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "batch_id": "batch_000",
        "worker_id": "worker_001",
        "hf_release_repo": "org/release",
    }
    settings.update(overrides)
    return settings


def test_phase01_config_encodes_one_fixed_production_pipeline() -> None:
    configs = load_configs(CONFIG_DIR)
    phase01 = configs["phase01"]
    models = configs["models"]
    storage = configs["storage"]

    assert phase01["pipeline_id"] == "phase01_production_v1_2"
    assert phase01["execution"]["max_concurrent_videos"] == 1
    assert phase01["execution"]["gpu_heavy_models_resident"] == 1
    assert phase01["execution"]["min_model_cache_free_gb"] == 25
    assert phase01["execution"]["chunk_scheduler"] == {
        "max_chunk_videos": 4,
        "max_chunk_raw_bytes": 1610612736,
        "min_free_disk_gb": 20,
        "medium_free_disk_gb": 35,
        "medium_max_chunk_videos": 2,
        "low_disk_max_chunk_videos": 1,
    }
    assert phase01["execution"]["inference_batch_size"] == {
        "ocr": 4,
        "shot_captions": 2,
    }
    assert phase01["api"]["max_concurrency_per_video"] == 2
    assert phase01["api"]["request_cache_backend"] == "stage_local"
    assert phase01["stages"]["order"] == [
        "shots",
        "keyframes",
        "asr",
        "ocr",
        "shot_captions",
        "shot_transcript_links",
        "scenes",
        "scene_summaries",
        "package",
        "sync",
    ]
    assert "providers" not in models
    assert storage["checkpoint"]["require_private"] is False
    assert set(models["phase01"]) == {
        "shot_detection",
        "asr",
        "asr_providers",
        "ocr",
        "shot_caption",
        "scene_boundary",
        "scene_summary",
    }
    assert models["phase01"]["asr"]["provider"] == "nemo"
    assert models["phase01"]["asr"]["model_id"] == "nvidia/parakeet-ctc-0.6b-vi"
    assert models["phase01"]["ocr"]["provider"] == "vintern_local"
    assert models["phase01"]["ocr"]["model_id"] == "5CD-AI/Vintern-1B-v3_5"
    assert models["phase01"]["shot_caption"]["provider"] == "vintern_local"
    assert models["phase01"]["shot_caption"]["model_id"] == "5CD-AI/Vintern-3B-R-beta"
    # Vintern-3B fits a 16 GB T4 at fp16, so captioning runs unquantised.
    assert "quantization" not in models["phase01"]["shot_caption"]
    assert models["phase01"]["shot_caption"]["torch_dtype"] == "float16"
    # scene_summary inherits the captioning model via model_key. Its declared
    # provider must track it — a stale one would load Vintern weights through
    # the Qwen code path.
    assert models["phase01"]["scene_summary"]["model_key"] == "shot_caption"
    assert (
        models["phase01"]["scene_summary"]["provider"]
        == models["phase01"]["shot_caption"]["provider"]
    )
    # scene_boundary runs Qwen-7B on its own weights: Vintern-3B could not hold
    # the boundary schema (bare arrays, misspelled keys, invented shot ids).
    boundary = models["phase01"]["scene_boundary"]
    assert "model_key" not in boundary
    assert boundary["provider"] == "qwen_local"
    assert boundary["model_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    # qwen_local refuses to load without an explicit 4bit block, and 7B only
    # fits a T4 quantised.
    assert boundary["quantization"]["mode"] == "4bit"
    assert set(models["phase01"]["asr_providers"]) == {"faster_whisper", "nemo"}


def test_canonical_structured_text_rejects_whitespace_only_values() -> None:
    assert _required_text({"caption_vi": "  nội dung  "}, "caption_vi") == "nội dung"
    with pytest.raises(ValueError, match="non-whitespace"):
        _required_text({"caption_vi": "   "}, "caption_vi")


def test_build_captions_submits_all_shots_through_request_many(
    tmp_path: Path,
) -> None:
    class RequestManyOnlyClient:
        def __init__(self) -> None:
            self.batches = []

        def request_many(self, requests):
            self.batches.append(requests)
            return [
                {
                    "caption_vi": f"Cảnh {index}",
                    "caption_en": f"Scene {index}",
                    "objects_vi": [],
                    "objects_en": [],
                    "actions_vi": [],
                    "actions_en": [],
                    "visible_text_summary_vi": "",
                    "visible_text_summary_en": "",
                    "scene_type": "unknown",
                    "__provider": "qwen_local",
                    "__model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "__model_revision": "revision",
                }
                for index, _request in enumerate(requests)
            ]

    client = RequestManyOnlyClient()
    shots = [{"shot_id": f"shot_{index:03d}"} for index in range(2)]
    keyframes = [
        {
            "shot_id": shot["shot_id"],
            "keyframe_id": f"keyframe_{index:03d}",
            "keyframe_ref": f"media://keyframes/frame_{index:03d}.jpg",
            "timestamp_sec": float(index),
            "is_representative": True,
        }
        for index, shot in enumerate(shots)
    ]
    model_config = load_configs(CONFIG_DIR)["models"]["phase01"]["shot_caption"]

    rows = _build_captions(
        video_id="L21_V001",
        shots=shots,
        keyframes=keyframes,
        ocr_rows=[],
        stage_dir=tmp_path,
        client=client,
        model_config=model_config,
        max_concurrency=2,
    )

    assert len(client.batches) == 1
    assert len(client.batches[0]) == 2
    assert [row["caption_en"] for row in rows] == ["Scene 0", "Scene 1"]


@pytest.mark.parametrize("provider", ["qwen_local", "gemini"])
def test_scene_summaries_v2_accepts_local_and_fallback_provenance(
    provider: str,
) -> None:
    validate_rows(
        "scene_summaries",
        [{
            "scene_id": "L21_V001_SC00000",
            "video_id": "L21_V001",
            "summary_vi": "Một cảnh",
            "summary_en": "A scene",
            "provider": provider,
            "model_name": "model",
            "model_version": "revision",
            "prompt_version": "scene_summary_v1",
            "schema_version": "scene_summary_response_v1",
            "confidence": None,
            "status": "pass",
        }],
    )


def test_keyframe_config_uses_search_bands_and_relative_representative_rule() -> None:
    media = load_configs(CONFIG_DIR)["media"]
    keyframe = media["keyframe"]

    assert keyframe["roles"] == {
        "early": {
            "search_start_ratio": 0.10,
            "target_ratio": 0.20,
            "search_end_ratio": 0.30,
        },
        "middle": {
            "search_start_ratio": 0.40,
            "target_ratio": 0.50,
            "search_end_ratio": 0.60,
        },
        "late": {
            "search_start_ratio": 0.70,
            "target_ratio": 0.80,
            "search_end_ratio": 0.90,
        },
    }
    assert keyframe["selection"]["max_candidates_per_band"] == 5
    assert keyframe["selection"]["deduplicate_frame_id"] is True
    assert keyframe["quality"]["metric"] == "variance_of_laplacian"
    assert keyframe["quality"]["absolute_blur_threshold"] is None
    assert keyframe["representative"]["preferred_role"] == "middle"
    assert keyframe["representative"]["preferred_min_ratio_of_best"] == 0.85


def test_phase01_config_encodes_oom_and_dependency_invalidation_policy() -> None:
    configs = load_configs(CONFIG_DIR)
    phase01 = configs["phase01"]
    models = configs["models"]["phase01"]
    dependencies = phase01["stages"]["dependencies"]

    assert models["asr"]["model_id"] == "nvidia/parakeet-ctc-0.6b-vi"
    assert models["asr"]["model_file"] == "parakeet-ctc-0.6b-vi.nemo"
    assert phase01["asr"]["exhausted_oom_status"] == "failed_retryable"
    assert dependencies["keyframes"] == ["shots"]
    assert dependencies["ocr"] == ["keyframes"]
    assert set(dependencies["shot_captions"]) == {"keyframes", "ocr"}
    assert set(dependencies["shot_transcript_links"]) == {"shots", "asr"}
    assert set(dependencies["scenes"]) == {
        "shots",
        "keyframes",
        "ocr",
        "shot_captions",
        "shot_transcript_links",
    }
    assert dependencies["sync"] == ["package"]


def test_runtime_chunk_policy_does_not_change_stage_fingerprints() -> None:
    resolved = resolve_phase01_config(
        CONFIG_DIR,
        user_settings=user_settings(),
        phase00_release_id="canonical_release_v001",
        environment="local",
    )
    modified = copy.deepcopy(resolved.payload)
    modified["phase01"]["execution"]["chunk_scheduler"]["max_chunk_videos"] = 1
    modified["phase01"]["execution"]["inference_batch_size"] = {
        "ocr": 1,
        "shot_captions": 1,
    }

    assert _stage_config_hashes(modified) == resolved.stage_config_hashes


def test_semantic_policies_change_only_relevant_stage_hashes() -> None:
    resolved = resolve_phase01_config(
        CONFIG_DIR,
        user_settings=user_settings(),
        phase00_release_id="canonical_release_v001",
        environment="local",
    )

    ocr_changed = copy.deepcopy(resolved.payload)
    ocr_changed["phase01"]["ocr"]["text_presence_filter"][
        "max_no_text_gray_std"
    ] = 11
    ocr_hashes = _stage_config_hashes(ocr_changed)
    assert ocr_hashes["ocr"] != resolved.stage_config_hashes["ocr"]
    assert ocr_hashes["shots"] == resolved.stage_config_hashes["shots"]

    caption_runtime_changed = copy.deepcopy(resolved.payload)
    caption_runtime_changed["models"]["shot_caption"]["max_dynamic_patch"] = 1
    caption_hashes = _stage_config_hashes(caption_runtime_changed)
    # `scenes` runs its own model now, so caption runtime settings no longer
    # reach it. `scene_summaries` still inherits the caption model via model_key.
    for stage in ("shot_captions", "scene_summaries"):
        assert caption_hashes[stage] != resolved.stage_config_hashes[stage]
    assert caption_hashes["scenes"] == resolved.stage_config_hashes["scenes"]

    boundary_runtime_changed = copy.deepcopy(resolved.payload)
    boundary_runtime_changed["models"]["scene_boundary"]["max_new_tokens"] = 123
    boundary_runtime_hashes = _stage_config_hashes(boundary_runtime_changed)
    assert (
        boundary_runtime_hashes["scenes"] != resolved.stage_config_hashes["scenes"]
    )

    boundary_changed = copy.deepcopy(resolved.payload)
    boundary_changed["models"]["scene_boundary"]["provider"] = "gemini"
    boundary_hashes = _stage_config_hashes(boundary_changed)
    assert boundary_hashes["scenes"] != resolved.stage_config_hashes["scenes"]
    assert (
        boundary_hashes["scene_summaries"]
        == resolved.stage_config_hashes["scene_summaries"]
    )


def test_phase01_config_can_select_nemo_asr_provider() -> None:
    resolved = resolve_phase01_config(
        CONFIG_DIR,
        user_settings={
            **user_settings(),
            "asr_provider": "nemo",
        },
        phase00_release_id="canonical_release_v001",
        environment="local",
    )
    models = resolved.payload["models"]
    assert models["asr"]["provider"] == "nemo"
    assert models["asr"]["model_id"] == "nvidia/parakeet-ctc-0.6b-vi"
    assert models["asr"]["model_revision"] == "b0493142b49458810324e3db8be9e8e07b4ebc17"
    assert models["asr"]["model_file"] == "parakeet-ctc-0.6b-vi.nemo"
    assert models["asr"]["segmentation"] == "ffmpeg_silence"
    assert models["asr"]["max_segment_seconds"] == 12


def test_phase01_config_can_select_faster_whisper_asr_provider() -> None:
    resolved = resolve_phase01_config(
        CONFIG_DIR,
        user_settings={**user_settings(), "asr_provider": "faster_whisper"},
        phase00_release_id="canonical_release_v001",
        environment="local",
    )

    asr = resolved.payload["models"]["asr"]
    assert asr["provider"] == "faster_whisper"
    assert asr["model_id"] == "Systran/faster-whisper-large-v3"
    assert asr["model_revision"] == "edaa852ec7e145841d8ffdb056a99866b5f0a478"


def test_resolved_config_is_stable_secret_free_and_auto_resolves_release(tmp_path: Path) -> None:
    first = resolve_phase01_config(
        CONFIG_DIR,
        user_settings=user_settings(),
        phase00_release_id="canonical_release_v001",
        environment="colab",
    )
    second = resolve_phase01_config(
        CONFIG_DIR,
        user_settings={
            "hf_release_repo": "org/release",
            "worker_id": "worker_001",
            "batch_id": "batch_000",
        },
        phase00_release_id="canonical_release_v001",
        environment="colab",
    )

    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64
    assert first.payload["runtime"]["release_id"] == "canonical_release_v001"
    assert first.payload["runtime"]["release_id_source"] == "phase00_auto_resolve"
    assert first.production_ready is True
    assert first.unresolved_required_fields == ()

    output = persist_resolved_phase01_config(first, tmp_path / "resolved_config.json")
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["config_hash"] == first.config_hash
    assert persisted["production_ready"] is True
    assert persisted["storage"]["release"]["repo_id"] == "org/release"
    assert set(persisted["stage_config_hashes"]) == set(
        first.payload["phase01"]["stages"]["order"]
    )
    assert "secret-value" not in output.read_text(encoding="utf-8")
    assert not (tmp_path / ".resolved_config.json.partial").exists()


def test_release_override_wins_over_auto_resolved_phase00_release() -> None:
    resolved = resolve_phase01_config(
        CONFIG_DIR,
        user_settings=user_settings(release_id_override="canonical_release_v009"),
        phase00_release_id="canonical_release_v001",
        environment="kaggle",
    )

    assert resolved.payload["runtime"]["release_id"] == "canonical_release_v009"
    assert resolved.payload["runtime"]["release_id_source"] == "user_override"


def test_checkpoint_repository_override_also_moves_model_artifact_store() -> None:
    resolved = resolve_phase01_config(
        CONFIG_DIR,
        user_settings=user_settings(
            hf_checkpoint_repo="org/checkpoints",
            checkpoint_revision="artifacts-v2",
        ),
        phase00_release_id="canonical_release_v001",
        environment="colab",
    )

    assert resolved.payload["storage"]["checkpoint"]["repo_id"] == "org/checkpoints"
    assert resolved.payload["storage"]["model_artifacts"]["repo_id"] == "org/checkpoints"
    assert resolved.payload["storage"]["model_artifacts"]["revision"] == "artifacts-v2"


def test_process_batch_does_not_override_versioned_storage_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict = {}

    def run_pipeline(**kwargs):
        captured.update(kwargs["user_settings"])
        return SimpleNamespace(
            release_dir=tmp_path / "release",
            worker_report_path=tmp_path / "report.json",
        )

    monkeypatch.setattr(
        "system1.commands.pipeline._phase01_test_provider_profile", lambda: "config"
    )
    monkeypatch.setattr("system1.commands.pipeline.run_phase01_pipeline", run_pipeline)
    result = CliRunner().invoke(
        app,
        ["process-batch", "--batch-id", "batch_000", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "hf_release_repo" not in captured
    assert "hf_repo_type" not in captured
    assert "hf_release_revision" not in captured
    assert "hf_release_prefix" not in captured


@pytest.mark.parametrize("secret_key", ["HF_TOKEN", "AIC_HF_TOKEN", "GEMINI_API_KEY"])
def test_resolved_config_rejects_secret_values(secret_key: str) -> None:
    with pytest.raises(ValueError, match="secret values"):
        resolve_phase01_config(
            CONFIG_DIR,
            user_settings=user_settings(**{secret_key: "secret-value"}),
            phase00_release_id="canonical_release_v001",
            environment="local",
        )


@pytest.mark.parametrize("forbidden_key", ["mode", "providers", "provider_profile"])
def test_resolved_config_rejects_pipeline_or_provider_selectors(forbidden_key: str) -> None:
    with pytest.raises(ValueError, match="unsupported Phase01 user settings"):
        resolve_phase01_config(
            CONFIG_DIR,
            user_settings=user_settings(**{forbidden_key: "mock"}),
            phase00_release_id="canonical_release_v001",
            environment="local",
        )


def test_production_readiness_lists_missing_authority_instead_of_guessing(
    tmp_path: Path,
) -> None:
    temp_config_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, temp_config_dir)
    models_yaml = temp_config_dir / "models.yaml"
    models_yaml.write_text(
        models_yaml.read_text(encoding="utf-8").replace(
            "weights_sha256: 834b10f25ae9e1b4e4f2652fe2843bd2b1388057a435d68b7c52635578fcc04d",
            "weights_sha256: null",
        ),
        encoding="utf-8",
    )
    resolved = resolve_phase01_config(
        temp_config_dir,
        user_settings=user_settings(),
        phase00_release_id="canonical_release_v001",
        environment="local",
    )

    assert resolved.unresolved_required_fields == ("models.shot_detection.weights_sha256",)
    with pytest.raises(ValueError, match="unresolved required fields"):
        require_phase01_production_ready(resolved)


def test_phase01_json_schemas_lock_checkpoint_and_keyframe_contracts() -> None:
    checkpoint = json.loads(
        (SCHEMA_DIR / "phase01_checkpoint_state.schema.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (SCHEMA_DIR / "resolved_config.schema.json").read_text(encoding="utf-8")
    )
    keyframes = json.loads(
        (SCHEMA_DIR / "keyframes.schema.json").read_text(encoding="utf-8")
    )

    expected_stages = {
        "shots",
        "keyframes",
        "asr",
        "ocr",
        "shot_captions",
        "shot_transcript_links",
        "scenes",
        "scene_summaries",
        "package",
        "sync",
    }
    assert set(checkpoint["properties"]["stages"]["required"]) == expected_stages
    assert {
        "status",
        "input_fingerprint",
        "config_hash",
        "model",
        "prompt_version",
        "schema_version",
        "output_checksums",
        "completed_at",
    }.issubset(checkpoint["$defs"]["stage"]["required"])
    assert resolved["properties"]["config_hash"]["pattern"] == "^[0-9a-f]{64}$"
    assert {"keyframe_role", "quality_score", "is_representative", "selection_reason"}.issubset(
        keyframes["required"]
    )
    assert keyframes["properties"]["keyframe_role"]["enum"] == ["early", "middle", "late"]


def test_package_assembly_backfills_scene_ids_and_passes_strict_validation(
    tmp_path: Path,
) -> None:
    video_id = "L21_V001"
    shot_id = f"{video_id}_SH00000"
    scene_id = f"{video_id}_SC00000"
    stage = tmp_path / "stage"
    (stage / "keyframes").mkdir(parents=True)
    (stage / "thumbnails").mkdir()
    (stage / "keyframes" / f"{video_id}_f0000000.jpg").write_bytes(b"jpg")
    (stage / "thumbnails" / f"{video_id}_f0000000.webp").write_bytes(b"webp")
    rows = {
        "shots": [{
            "shot_id": shot_id, "video_id": video_id, "scene_id": None,
            "shot_index": 0, "start_frame": 0, "end_frame": 1,
            "start_sec": 0.0, "end_sec": 0.04, "duration_sec": 0.04,
            "frame_count": 1, "boundary_convention": "[start_frame, end_frame)",
            "detection_method": "transnet_v2", "status": "transnet_v2_no_cut",
        }],
        "keyframes": [{
            "keyframe_id": f"{video_id}:0", "video_id": video_id, "frame_id": 0,
            "timestamp_sec": 0.0, "shot_id": shot_id, "scene_id": None,
            "keyframe_role": "middle", "quality_score": 10.0,
            "is_representative": True, "selection_reason": "middle_within_quality_ratio",
            "keyframe_ref": f"media://keyframes/{video_id}/{video_id}_f0000000.jpg",
            "thumbnail_ref": f"media://thumbnails/{video_id}/{video_id}_f0000000.webp",
            "status": "pass",
        }],
        "ocr": [{
            "ocr_id": f"{video_id}:0:ocr", "video_id": video_id,
            "keyframe_id": f"{video_id}:0", "shot_id": shot_id,
            "frame_id": 0, "text": "", "raw_text": "",
            "provider": "vintern_local", "model_name": "5CD-AI/Vintern-1B-v3_5",
            "model_version": "b98f263eab246eb5269ade64edbdca8a887dc44d",
            "language": "vi", "confidence": None, "status": "empty",
        }],
        "shot_captions": [{
            "shot_caption_id": f"{shot_id}_caption", "video_id": video_id,
            "shot_id": shot_id, "representative_keyframe_id": f"{video_id}:0",
            "representative_timestamp_sec": 0.0, "caption_vi": "Một cảnh",
            "caption_en": "A scene", "objects_vi": ["cảnh"], "objects_en": ["scene"],
            "actions_vi": [], "actions_en": [], "visible_text_summary_vi": "",
            "visible_text_summary_en": "", "scene_type": "unknown",
            "provider": "qwen_local",
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_version": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "prompt_version": "shot_caption_v2",
            "schema_version": "shot_caption_response_v2", "confidence": None,
            "status": "pass",
        }],
        "scenes": [{
            "scene_id": scene_id, "video_id": video_id, "scene_index": 0,
            "start_shot_id": shot_id, "end_shot_id": shot_id,
            "start_frame": 0, "end_frame": 1, "start_sec": 0.0,
            "end_sec": 0.04, "duration_sec": 0.04, "frame_count": 1,
            "shot_count": 1, "keyframe_count": 0, "scene_type": "semantic",
            "grouping_method": "multimodal_context_focus",
            "grouping_version": "scene_grouping_v1", "confidence": None,
            "boundary_convention": "[start_frame, end_frame)", "status": "pass",
        }],
        "scene_summaries": [{
            "scene_id": scene_id, "video_id": video_id, "summary_vi": "Một cảnh",
            "summary_en": "A scene", "provider": "gemini",
            "model_name": "gemini-3.6-flash", "model_version": "gemini-3.6-flash",
            "prompt_version": "scene_summary_v1",
            "schema_version": "scene_summary_response_v1", "confidence": None,
            "status": "pass",
        }],
    }
    for name, values in rows.items():
        pd.DataFrame(values).to_parquet(stage / f"{name}.parquet", index=False)
    for name in ("asr_segments", "shot_transcript_links", "scene_transcript_links"):
        pd.DataFrame(columns=PARQUET_COLUMNS[name]).to_parquet(
            stage / f"{name}.parquet", index=False
        )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"video_id": video_id}), encoding="utf-8")
    resolved = resolve_phase01_config(
        CONFIG_DIR,
        user_settings=user_settings(),
        phase00_release_id="canonical_release_v001",
        environment="local",
    )

    artifact = tmp_path / "artifact" / video_id
    _assemble_package(
        artifact_dir=artifact,
        video_id=video_id,
        metadata_path=metadata,
        stage_dir=stage,
        config=resolved,
    )

    assert (artifact / "errors.jsonl").read_text(encoding="utf-8") == ""
    assert pd.read_parquet(artifact / "shots.parquet").iloc[0]["scene_id"] == scene_id
    assert pd.read_parquet(artifact / "scenes.parquet").iloc[0]["keyframe_count"] == 1


def test_semantic_stages_resolve_to_one_consistent_local_provider() -> None:
    """A stage's provider must match the model it actually loads.

    Overriding `provider` on a stage that inherits its model via `model_key`
    silently pairs one architecture's loader with another's weights.
    """
    models = load_configs(CONFIG_DIR)["models"]["phase01"]

    for stage_name in ("scene_boundary", "scene_summary"):
        stage = copy.deepcopy(models[stage_name])
        model_key = stage.pop("model_key", None)
        if model_key is None:
            # Stage declares its own weights, so there is nothing to inherit.
            # It still has to name both halves itself.
            assert stage["provider"]
            assert stage["model_id"]
            continue
        resolved = {**copy.deepcopy(models[model_key]), **stage}

        assert resolved["provider"] == models[model_key]["provider"]
        assert resolved["model_id"] == models[model_key]["model_id"]


def test_local_vlm_stages_share_the_same_runtime_family() -> None:
    models = load_configs(CONFIG_DIR)["models"]["phase01"]

    providers = {
        models["ocr"]["provider"],
        models["shot_caption"]["provider"],
    }

    assert providers == {"vintern_local"}
