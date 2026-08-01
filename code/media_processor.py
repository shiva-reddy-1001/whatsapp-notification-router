"""Best-effort local media extraction; failures are explicit and non-fatal."""
from pathlib import Path
from typing import Tuple

from .config import Settings


class MediaProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._whisper = None

    def extract(self, media_type: str, path: Path) -> Tuple[str, float]:
        if not path or not path.exists() or self.settings.media_mode == "off":
            return "", 0.0
        try:
            if media_type == "image":
                return self._image_text(path)
            if media_type == "voice":
                return self._voice_text(path)
        except Exception:
            # The decision layer receives low media quality instead of a fake claim.
            return "", 0.0
        return "", 0.0

    def _image_text(self, path: Path) -> Tuple[str, float]:
        try:
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(path)).strip()
            return text, 0.75 if text else 0.25
        except Exception:
            return "", 0.0

    def _voice_text(self, path: Path) -> Tuple[str, float]:
        try:
            from faster_whisper import WhisperModel
            if self._whisper is None:
                self._whisper = WhisperModel(self.settings.whisper_model,
                                             device="cpu",
                                             compute_type=self.settings.whisper_compute_type)
            segments, info = self._whisper.transcribe(str(path), beam_size=1)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text, 0.72 if text else 0.0
        except Exception:
            return "", 0.0
