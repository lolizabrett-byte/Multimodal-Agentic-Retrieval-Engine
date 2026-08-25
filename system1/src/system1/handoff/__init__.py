"""Standalone entry points for the OCR & ASR deliverables.

Thin wrappers over the Phase01 pipeline so the extraction functions can be called
on a single file without running a whole batch. No logic lives here.
"""

from .audio_text import extract_audio_text
from .image_text import extract_image_text

__all__ = ["extract_audio_text", "extract_image_text"]
