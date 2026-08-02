"""Cached OCR, Qwen vision analysis, and local voice transcription."""
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Tuple

from .config import Settings
from .cache import SQLiteCache
from .reliability import InvalidModelResponse, RetryPolicy, run_with_retry


class MediaProcessor:
    def __init__(self, settings: Settings, cache: SQLiteCache = None):
        self.settings = settings
        self.cache = cache
        self._whisper = None

    def check(self) -> str:
        if self.settings.resolved_vision_provider() == "none":
            return "vision disabled; OCR remains active"
        from ollama import Client
        Client(host=self.settings.ollama_base_url,
               timeout=self.settings.timeout_seconds).show(self.settings.vision_model)
        return "Qwen vision configured for model %s" % self.settings.vision_model

    def extract(self, media_type: str, path: Path) -> Tuple[str, float]:
        if not path or not path.exists() or self.settings.media_mode == "off":
            return "", 0.0
        cache_key = "%s:%s:%s:%s" % (media_type, path, path.stat().st_mtime_ns,
                                      self.settings.whisper_model if media_type == "voice" else
                                      "ocr-v3:%s:%s" % (self.settings.resolved_vision_provider(),
                                                        self.settings.vision_model))
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
        except Exception as error:
            # The decision layer receives low media quality instead of a fake claim.
            logging.warning("media extraction failed type=%s file=%s category=%s",
                            media_type, path.name, type(error).__name__)
            result = ("", 0.0)
        if self.cache:
            self.cache.put("media", cache_key, {"text": result[0], "quality": result[1]})
        return result

    def _image_text(self, path: Path) -> Tuple[str, float]:
        ocr = ""
        try:
            import pytesseract
            from PIL import Image
            ocr = pytesseract.image_to_string(Image.open(path)).strip()
        except Exception:
            pass
        vision = self._vision_text(path) if self.settings.resolved_vision_provider() == "ollama" else ""
        parts = []
        if ocr:
            parts.append("OCR text:\n%s" % ocr)
        if vision:
            parts.append("Vision analysis:\n%s" % vision)
        text = "\n\n".join(parts)
        quality = 0.90 if ocr and vision else (0.78 if vision else (0.70 if ocr else 0.0))
        return text, quality

    def _vision_text(self, path: Path) -> str:
        from ollama import Client
        from PIL import Image
        client = Client(host=self.settings.ollama_base_url,
                        timeout=self.settings.timeout_seconds)
        schema = {"type": "object", "required": ["document_type", "visible_text", "facts", "risk_cues"],
                  "properties": {"document_type": {"type": "string"},
                                 "visible_text": {"type": "string"},
                                 "facts": {"type": "array", "items": {"type": "string"}},
                                 "risk_cues": {"type": "array", "items": {"type": "string"}}}}
        prompt = ("Analyze this WhatsApp image factually for downstream notification routing. "
                  "Extract visible text, document/poster type, dates and deadlines, amounts, "
                  "payment or QR cues, URLs, and urgency or scam cues. Do not decide notify/digest/mute.")
        # Dataset extensions are not always the actual encoding (some .jpg files
        # are AVIF/WEBP). Canonical bytes also cap huge phone photos before Qwen.
        image = Image.open(path).convert("RGB")
        image.thumbnail((1600, 1600))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        image_bytes = buffer.getvalue()

        def operation(attempt: int) -> str:
            response = client.chat(model=self.settings.vision_model,
                                   messages=[{"role": "user", "content": prompt,
                                              "images": [image_bytes]}],
                                   format=schema, options={"temperature": 0})
            raw = response["message"]["content"].strip()
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            try:
                data = json.loads(match.group(0) if match else raw)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                if attempt < self.settings.max_retries:
                    raise InvalidModelResponse("vision provider returned invalid JSON") from error
                # Vision evidence is free text downstream; retain a final factual
                # answer instead of discarding it solely because JSON was malformed.
                return "unstructured_vision=%s" % raw[:2000]
            facts = "; ".join(str(value) for value in data.get("facts", []))
            risks = "; ".join(str(value) for value in data.get("risk_cues", []))
            return ("type=%s; visible_text=%s; facts=%s; risk_cues=%s" %
                    (data.get("document_type", "unknown"), data.get("visible_text", ""),
                     facts, risks)).strip()

        policy = RetryPolicy(self.settings.max_retries, self.settings.retry_mode,
                             self.settings.retry_base_seconds, self.settings.retry_max_seconds,
                             self.settings.retry_jitter_seconds,
                             self.settings.message_deadline_seconds)
        return run_with_retry(operation, policy)

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
