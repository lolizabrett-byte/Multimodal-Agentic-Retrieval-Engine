"""Tiền xử lý ảnh trước OCR — dùng để THỬ NGHIỆM và ĐO, không bật mặc định.

Các kỹ thuật kinh điển (nhị phân hoá, morphology, chuyển xám) sinh ra cho OCR đời
cũ như Tesseract/PaddleOCR. Vintern là mô hình thị giác-ngôn ngữ học trên ảnh màu
tự nhiên, nên ép ảnh về dạng nó chưa từng thấy có thể làm giảm chất lượng.

Vì vậy mọi kỹ thuật ở đây đều phải đo CER trước/sau bằng
`research/ocr_asr/preprocess_experiment.py` rồi mới quyết định dùng. Không áp
dụng mù.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

# Nhóm giữ nguyên ảnh màu — ít rủi ro với VLM
SAFER = ("clahe", "denoise", "sharpen", "upscale")
# Nhóm đổi bản chất ảnh — rủi ro cao với VLM, chỉ để đo và chứng minh
RISKY = ("grayscale", "threshold", "adaptive_threshold", "morphology")
AVAILABLE = SAFER + RISKY


def _clahe(image: np.ndarray) -> np.ndarray:
    """Cân bằng sáng cục bộ trên kênh độ sáng, giữ nguyên màu."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((equalized, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def _sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def _upscale(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)


def _grayscale(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def _threshold(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _adaptive_threshold(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _morphology(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)


_OPERATIONS = {
    "clahe": _clahe,
    "denoise": _denoise,
    "sharpen": _sharpen,
    "upscale": _upscale,
    "grayscale": _grayscale,
    "threshold": _threshold,
    "adaptive_threshold": _adaptive_threshold,
    "morphology": _morphology,
}


def apply_array(image: np.ndarray, name: str) -> np.ndarray:
    """Áp một kỹ thuật. `name` có thể ghép bằng '+' (vd 'clahe+sharpen')."""
    result = image
    for step in name.split("+"):
        step = step.strip()
        if step not in _OPERATIONS:
            raise ValueError(f"unknown preprocess {step!r}; available: {sorted(_OPERATIONS)}")
        result = _OPERATIONS[step](result)
    return result


def apply_preprocess(image_path: Path, name: str, stack: list[Path]) -> Path:
    """Áp kỹ thuật rồi ghi ra file tạm; đường dẫn tạm được thêm vào `stack`."""
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise ValueError(f"could not decode image: {image_path}")

    processed = apply_array(image, name)

    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    handle.close()
    output = Path(handle.name)
    cv2.imwrite(str(output), processed)
    stack.append(output)
    return output
