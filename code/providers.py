"""Provider adapters with strict structured-output validation."""
import json
import re
import hashlib
from typing import Optional

from .config import Settings
from .cache import SQLiteCache
from .models import ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES, CaseFile, Prediction
from .prompting import PROMPT_VERSION, build_casefile_prompt


def _parse(case: CaseFile, raw: str) -> Optional[Prediction]:
    try:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        action, kind = str(data["action"]).strip().lower(), str(data["message_type"]).strip().lower()
        confidence = float(data["confidence"])
        # Some small local models emit an otherwise valid 1-10 or percentage
        # score. Normalize those documented scales without clamping arbitrary
        # invalid values.
        if 1 < confidence <= 10:
            confidence /= 10
        elif 10 < confidence <= 100:
            confidence /= 100
        available = {item.message_id for item in case.evidence}
        raw_evidence = data.get("evidence_message_ids", [])
        if raw_evidence in (None, "none"):
            raw_evidence = []
        if not isinstance(raw_evidence, list):
            return None
        evidence = list(dict.fromkeys(item for item in raw_evidence if item in available))[:3]
        if action not in ALLOWED_ACTIONS or kind not in ALLOWED_MESSAGE_TYPES or not 0 <= confidence <= 1:
            return None
        reason = str(data["reason"]).strip().replace("\n", " ")[:280]
        if not reason:
            return None
        return Prediction(case.message.message_id, action, kind, reason, confidence, evidence)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class Classifier:
    def __init__(self, settings: Settings, cache: SQLiteCache = None):
        self.settings = settings
        self.name = settings.resolved_provider()
        self.cache = cache

    def classify(self, case: CaseFile) -> Prediction:
        prompt = build_casefile_prompt(case)
        key = hashlib.sha256((self.name + self.settings.openai_model + self.settings.ollama_model + PROMPT_VERSION + prompt).encode()).hexdigest()
        cached = self.cache.get("predictions", key) if self.cache else None
        if cached:
            return Prediction(**cached)
        last_error = None
        repair = ""
        for attempt in range(self.settings.max_retries + 1):
            try:
                raw = self._call(prompt + repair)
                result = _parse(case, raw)
                if not result:
                    raise RuntimeError("provider returned invalid structured classification")
                if self.cache:
                    self.cache.put("predictions", key, {"message_id": result.message_id, "action": result.action,
                                   "message_type": result.message_type, "reason": result.reason,
                                   "confidence": result.confidence, "evidence_message_ids": result.evidence_message_ids})
                return result
            except Exception as error:
                last_error = error
                repair = "\nYour previous response was invalid. Return exactly the requested JSON schema; confidence must be 0.0 to 1.0."
        raise RuntimeError("classification failed for %s: %s" % (case.message.message_id, last_error))

    def _call(self, prompt: str) -> str:
        if self.name == "openai":
            from openai import OpenAI
            client = OpenAI(timeout=self.settings.timeout_seconds, max_retries=0)
            response = client.responses.create(model=self.settings.openai_model, input=prompt,
                                               temperature=self.settings.temperature)
            return response.output_text
        if self.name == "ollama":
            from ollama import Client
            client = Client(host=self.settings.ollama_base_url, timeout=self.settings.timeout_seconds)
            schema = {"type": "object", "required": ["action", "message_type", "reason", "confidence", "evidence_message_ids"],
                      "properties": {"action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                                     "message_type": {"type": "string", "enum": sorted(ALLOWED_MESSAGE_TYPES)},
                                     "reason": {"type": "string"},
                                     "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                     "evidence_message_ids": {"type": "array", "items": {"type": "string"}}}}
            response = client.chat(model=self.settings.ollama_model,
                                   messages=[{"role": "user", "content": prompt}],
                                   format=schema, options={"temperature": self.settings.temperature})
            return response["message"]["content"]
        raise RuntimeError("unsupported provider")

    def check(self) -> str:
        if self.name == "openai":
            import os
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required when provider=openai")
            return "OpenAI configured for model %s" % self.settings.openai_model
        from ollama import Client
        Client(host=self.settings.ollama_base_url, timeout=self.settings.timeout_seconds).show(self.settings.ollama_model)
        return "Ollama configured for model %s" % self.settings.ollama_model
