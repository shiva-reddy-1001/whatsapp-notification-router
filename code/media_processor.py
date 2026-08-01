"""Best-effort local media extraction; failures are explicit and non-fatal."""
from pathlib import Path
from typing import Tuple

from .config import Settings
from .cache import SQLiteCache


class MediaProcessor:
    def __init__(self, settings: Settings, cache: SQLiteCache = None):
        self.settings = settings
        self.cache = cache
        self._whisper = None

    def extract(self, media_type: str, path: Path) -> Tuple[str, float]:
        if not path or not path.exists() or self.settings.media_mode == "off":
            return "", 0.0
        cache_key = "%s:%s:%s:%s" % (media_type, path, path.stat().st_mtime_ns,
                                      self.settings.whisper_model if media_type == "voice" else "ocr-v1")
        cached = self.cache.get("media", cache_key) if self.cache else None
        if cached:
            return cached["text"], cached["quality"]
        try:
            if media_type == "image":
                result = self._image_text(path)
            elif media_type == "voice":
                result = self._voice_text(path)
            else:
                result = ("", 0.0)
        except Exception:
            # The decision layer receives low media quality instead of a fake claim.
            result = ("", 0.0)
        if self.cache:
            self.cache.put("media", cache_key, {"text": result[0], "quality": result[1]})
        return result

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
