from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_dual_gpu.py"
spec = importlib.util.spec_from_file_location("run_dual_gpu", SCRIPT)
run_dual_gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_dual_gpu)

split_evenly = run_dual_gpu.split_evenly
read_manifest = run_dual_gpu.read_manifest


def test_even_split_across_two_gpus():
    shards = split_evenly([f"V{index:03d}" for index in range(88)], 2)

    assert [len(shard) for shard in shards] == [44, 44]


def test_odd_count_loses_no_video():
    videos = [f"V{index:03d}" for index in range(87)]

    shards = split_evenly(videos, 2)

    assert [len(shard) for shard in shards] == [44, 43]
    assert sorted(video for shard in shards for video in shard) == sorted(videos)


def test_single_gpu_keeps_one_shard():
    shards = split_evenly(["a", "b", "c"], 1)

    assert shards == [["a", "b", "c"]]


def test_more_gpus_than_videos_yields_empty_shards():
    shards = split_evenly(["only"], 2)

    assert shards == [["only"], []]


def test_empty_manifest_produces_empty_shards():
    assert split_evenly([], 2) == [[], []]


def test_zero_parts_is_rejected():
    with pytest.raises(ValueError):
        split_evenly(["a"], 0)


def test_shards_never_share_a_video():
    videos = [f"V{index:03d}" for index in range(51)]

    shards = split_evenly(videos, 3)

    seen = [video for shard in shards for video in shard]
    assert len(seen) == len(set(seen)) == len(videos)


def test_manifest_reader_skips_blank_lines(tmp_path: Path):
    manifest = tmp_path / "batch_001.txt"
    manifest.write_text("L21_V009\n\n  \nL22_V008\n", encoding="utf-8")

    assert read_manifest(manifest) == ["L21_V009", "L22_V008"]


def test_manifest_reader_reports_missing_file(tmp_path: Path):
    with pytest.raises(SystemExit, match="batch manifest not found"):
        read_manifest(tmp_path / "nope.txt")
