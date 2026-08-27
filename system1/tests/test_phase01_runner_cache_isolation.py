"""Two workers on one machine must not share a download cache.

scripts/run_dual_gpu.py runs one process per GPU against the same --output, so
every cache path derived from it was the same directory. Each process deletes
its cache in a finally block, so the first one to finish removed the partial
downloads the other was still writing: the 27/08 dual-GPU run died with
"No such file or directory: ...blobs/9d24d42d....incomplete" while the other
shard finished all its videos cleanly.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "system1" / "phase01" / "runner.py"
).read_text(encoding="utf-8")

CACHE_LINES = [
    line.strip()
    for line in SOURCE.splitlines()
    if re.search(r"_cache\s*=.*(\.hf_cache|_hf_cache)", line)
]


def test_the_runner_still_builds_the_three_caches() -> None:
    """Guards the test itself: a rename must not silently empty the checks below."""
    assert len(CACHE_LINES) == 3, CACHE_LINES


def test_every_cache_path_is_unique_per_process() -> None:
    for line in CACHE_LINES:
        assert "os.getpid()" in line, line


def test_each_cache_is_still_removed_after_use() -> None:
    """Per-process paths must not turn into per-process leaks on a small disk."""
    assert SOURCE.count("shutil.rmtree(discovery_cache") == 1
    assert SOURCE.count("shutil.rmtree(preflight_cache") == 1
    assert SOURCE.count("shutil.rmtree(restore_cache") == 1
