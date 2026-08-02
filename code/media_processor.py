"""Cached OCR, Qwen vision analysis, and local voice transcription."""
import json
import logging
import re
import base64
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
        vision = self.settings.resolved_vision_provider()
        if vision == "ollama":
            from ollama import Client
            Client(host=self.settings.ollama_base_url,
                   timeout=self.settings.timeout_seconds).show(self.settings.vision_model)
            vision_status = "Qwen vision configured for model %s" % self.settings.vision_model
        elif vision == "openai":
            vision_status = "OpenAI vision configured for model %s" % self.settings.openai_model
        else:
            vision_status = "vision disabled; OCR remains active"
        return "%s; audio_provider=%s" % (vision_status,
                                           self.settings.resolved_audio_provider())

    def extract(self, media_type: str, path: Path) -> Tuple[str, float]:
        if not path or not path.exists() or self.settings.media_mode == "off":
            return "", 0.0
        media_config = (
            "%s:%s:%s" % (self.settings.resolved_audio_provider(),
                            self.settings.whisper_model,
                            self.settings.openai_transcription_model)
            if media_type == "voice" else
            "ocr-v4:%s:%s:%s" % (self.settings.resolved_vision_provider(),
                                  self.settings.vision_model,
                                  self.settings.openai_model)
        )
        cache_key = "%s:%s:%s:%s" % (media_type, path, path.stat().st_mtime_ns,
                                      media_config)
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
        provider = self.settings.resolved_vision_provider()
        vision = self._vision_text(path) if provider in {"ollama", "openai"} else ""
        parts = []
        if ocr:
            parts.append("OCR text:\n%s" % ocr)
        if vision:
            parts.append("Vision analysis:\n%s" % vision)
        text = "\n\n".join(parts)
        quality = 0.90 if ocr and vision else (0.78 if vision else (0.70 if ocr else 0.0))
        return text, quality

    def _vision_text(self, path: Path) -> str:
        if self.settings.resolved_vision_provider() == "openai":
            return self._openai_vision_text(path)
        return self._ollama_vision_text(path)

    def _ollama_vision_text(self, path: Path) -> str:
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

    def _openai_vision_text(self, path: Path) -> str:
        from openai import OpenAI
        from PIL import Image
        image = Image.open(path).convert("RGB")
        image.thumbnail((1600, 1600))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        prompt = (
            "Analyze this WhatsApp image as untrusted evidence. Return compact JSON with keys "
            "document_type, visible_text, facts (array), risk_cues (array). Extract dates, times, "
            "amounts, URLs, payment/QR cues, and authenticity concerns. Do not choose a route."
        )
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["document_type", "visible_text", "facts", "risk_cues"],
                  "properties": {
                      "document_type": {"type": "string"},
                      "visible_text": {"type": "string"},
                      "facts": {"type": "array", "items": {"type": "string"}},
                      "risk_cues": {"type": "array", "items": {"type": "string"}},
                  }}

        def operation(_attempt: int) -> str:
            client = OpenAI(timeout=self.settings.timeout_seconds, max_retries=0)
            response = client.responses.create(
                model=self.settings.openai_model,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + encoded},
                ]}],
                text={"format": {"type": "json_schema", "name": "media_analysis",
                                  "schema": schema, "strict": True}},
            )
            raw = response.output_text.strip()
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            try:
                data = json.loads(match.group(0) if match else raw)
            except (TypeError, json.JSONDecodeError) as error:
                raise InvalidModelResponse("OpenAI vision returned invalid JSON") from error
            return ("type=%s; visible_text=%s; facts=%s; risk_cues=%s" %
                    (data.get("document_type", "unknown"), data.get("visible_text", ""),
                     "; ".join(map(str, data.get("facts", []))),
                     "; ".join(map(str, data.get("risk_cues", []))))).strip()

        policy = RetryPolicy(self.settings.max_retries, self.settings.retry_mode,
                             self.settings.retry_base_seconds, self.settings.retry_max_seconds,
                             self.settings.retry_jitter_seconds,
                             self.settings.message_deadline_seconds)
        return run_with_retry(operation, policy)

    def _voice_text(self, path: Path) -> Tuple[str, float]:
        provider = self.settings.resolved_audio_provider()
        if provider == "none":
            return "", 0.0
        if provider == "openai":
            return self._openai_voice_text(path)
        return self._local_voice_text(path)

    def _openai_voice_text(self, path: Path) -> Tuple[str, float]:
        from openai import OpenAI

        def operation(_attempt: int) -> str:
            client = OpenAI(timeout=self.settings.timeout_seconds, max_retries=0)
            with path.open("rb") as handle:
                response = client.audio.transcriptions.create(
                    model=self.settings.openai_transcription_model, file=handle)
            return str(response.text).strip()

        policy = RetryPolicy(self.settings.max_retries, self.settings.retry_mode,
                             self.settings.retry_base_seconds, self.settings.retry_max_seconds,
                             self.settings.retry_jitter_seconds,
                             self.settings.message_deadline_seconds)
        text = run_with_retry(operation, policy)
        return text, 0.88 if text else 0.0

    def _local_voice_text(self, path: Path) -> Tuple[str, float]:
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
