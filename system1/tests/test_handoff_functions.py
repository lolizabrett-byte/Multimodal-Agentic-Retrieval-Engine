from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from system1.handoff import extract_audio_text, extract_image_text
from system1.handoff import image_text


def _image(path: Path, size: tuple[int, int] = (320, 180)) -> Path:
    Image.new("RGB", size, color=(90, 120, 150)).save(path)
    return path


class _StubClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests: list = []

    def request(self, request):
        self.requests.append(request)
        return self.payload

    def close(self) -> None:
        return None


@pytest.fixture
def stub_ocr(monkeypatch):
    def _install(payload: dict) -> _StubClient:
        client = _StubClient(payload)
        monkeypatch.setattr(image_text, "_client_for", lambda _config: client)
        return client

    return _install


def test_extract_image_text_accepts_a_path(tmp_path: Path, stub_ocr) -> None:
    stub_ocr({"full_text": "UY BAN NHAN DAN", "ocr_blocks": []})

    assert extract_image_text(_image(tmp_path / "frame.jpg")) == "UY BAN NHAN DAN"


def test_extract_image_text_accepts_a_pil_image(tmp_path: Path, stub_ocr) -> None:
    stub_ocr({"full_text": "Trường Sa", "ocr_blocks": []})

    with Image.open(_image(tmp_path / "frame.jpg")) as opened:
        assert extract_image_text(opened) == "Trường Sa"


def test_extract_image_text_accepts_a_numpy_array(tmp_path: Path, stub_ocr) -> None:
    numpy = pytest.importorskip("numpy")
    stub_ocr({"full_text": "Hà Nội", "ocr_blocks": []})

    array = numpy.zeros((180, 320, 3), dtype=numpy.uint8)

    assert extract_image_text(array) == "Hà Nội"


def test_extract_image_text_falls_back_to_blocks(tmp_path: Path, stub_ocr) -> None:
    stub_ocr(
        {"full_text": "", "ocr_blocks": [{"text": "Km 12"}, {"text": "Quốc lộ 1A"}]}
    )

    assert extract_image_text(_image(tmp_path / "frame.jpg")) == "Km 12 Quốc lộ 1A"


def test_extract_image_text_returns_empty_when_no_text(tmp_path: Path, stub_ocr) -> None:
    stub_ocr({"full_text": "", "ocr_blocks": []})

    assert extract_image_text(_image(tmp_path / "frame.jpg")) == ""


def test_extract_image_text_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_image_text(tmp_path / "absent.jpg")


def test_extract_image_text_rejects_unsupported_input(stub_ocr) -> None:
    stub_ocr({"full_text": "", "ocr_blocks": []})

    with pytest.raises(TypeError):
        extract_image_text(12345)


def test_max_num_overrides_tiling_budget(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_client_for(config):
        captured.update(config)
        return _StubClient({"full_text": "ok", "ocr_blocks": []})

    monkeypatch.setattr(image_text, "_client_for", fake_client_for)

    extract_image_text(_image(tmp_path / "frame.jpg"), max_num=1)

    assert captured["max_dynamic_patch"] == 1


def test_preprocess_leaves_no_temp_files_behind(tmp_path: Path, stub_ocr) -> None:
    pytest.importorskip("cv2")
    stub_ocr({"full_text": "ok", "ocr_blocks": []})
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("*.png"))

    extract_image_text(_image(tmp_path / "frame.jpg"), preprocess="clahe")

    assert set(Path(tempfile.gettempdir()).glob("*.png")) - before == set()


def test_extract_audio_text_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio_text(tmp_path / "absent.mp4")


def test_extract_audio_text_returns_empty_for_silent_media(
    tmp_path: Path, monkeypatch
) -> None:
    from system1.handoff import audio_text

    media = tmp_path / "silent.mp4"
    media.write_bytes(b"stub")
    monkeypatch.setattr(audio_text, "has_audio_stream", lambda _path: False)

    assert extract_audio_text(media) == []


def test_extract_audio_text_returns_timestamped_segments(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from system1.handoff import audio_text

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"stub")
    monkeypatch.setattr(audio_text, "has_audio_stream", lambda _path: True)
    monkeypatch.setattr(
        audio_text,
        "transcribe_video",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="pass",
            rows=[
                {"start_sec": 0.0, "end_sec": 1.5, "text": "xin chào", "language": "vi"},
                {"start_sec": 1.5, "end_sec": 2.0, "text": "   ", "language": "vi"},
                {"start_sec": 2.0, "end_sec": 4.0, "text": "tạm biệt", "language": "vi"},
            ],
        ),
    )

    segments = extract_audio_text(media)

    assert [segment["text"] for segment in segments] == ["xin chào", "tạm biệt"]
    assert all(segment["start"] < segment["end"] for segment in segments)
    assert segments[0]["start"] == 0.0


def test_extract_audio_text_rejects_unknown_provider(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"stub")

    with pytest.raises(ValueError, match="unknown ASR provider"):
        extract_audio_text(media, provider="not_a_provider")
