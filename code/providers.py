"""Optional LLM adapters. Invalid responses always fall back to deterministic routing."""
import json
import re
import hashlib
from typing import Optional

from .config import Settings
from .cache import SQLiteCache
from .models import ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES, CaseFile, Prediction
from .prompting import PROMPT_VERSION, build_casefile_prompt
from .router import deterministic_route


def _parse(case: CaseFile, raw: str) -> Optional[Prediction]:
    try:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        action, kind = data["action"], data["message_type"]
        confidence = float(data["confidence"])
        available = {item.message_id for item in case.evidence}
        evidence = [item for item in data.get("evidence_message_ids", []) if item in available][:3]
        if action not in ALLOWED_ACTIONS or kind not in ALLOWED_MESSAGE_TYPES or not 0 <= confidence <= 1:
            return None
        reason = str(data["reason"]).strip().replace("\n", " ")[:280]
        return Prediction(case.message.message_id, action, kind, reason, confidence, evidence)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class Classifier:
    def __init__(self, settings: Settings, cache: SQLiteCache = None):
        self.settings = settings
        self.name = settings.resolved_provider()
        self.cache = cache

    def classify(self, case: CaseFile) -> Prediction:
        # Do not spend a model call on clear safety cases; deterministic output is auditable.
        fallback = deterministic_route(case)
        if fallback.confidence >= 0.78 or self.name == "rules":
            return fallback
        prompt = build_casefile_prompt(case)
        key = hashlib.sha256((self.name + self.settings.openai_model + self.settings.ollama_model + PROMPT_VERSION + prompt).encode()).hexdigest()
        cached = self.cache.get("predictions", key) if self.cache else None
        if cached:
            return Prediction(**cached)
        try:
            raw = self._call(prompt)
            parsed = _parse(case, raw)
            result = parsed or fallback
            if self.cache:
                self.cache.put("predictions", key, {"message_id": result.message_id, "action": result.action,
                               "message_type": result.message_type, "reason": result.reason,
                               "confidence": result.confidence, "evidence_message_ids": result.evidence_message_ids})
            return result
        except Exception:
            return fallback

    def _call(self, prompt: str) -> str:
        if self.name == "openai":
            from openai import OpenAI
            client = OpenAI(timeout=self.settings.timeout_seconds)
            response = client.responses.create(model=self.settings.openai_model, input=prompt,
                                               temperature=self.settings.temperature)
            return response.output_text
        if self.name == "ollama":
            from ollama import Client
            client = Client(host=self.settings.ollama_base_url, timeout=self.settings.timeout_seconds)
            response = client.chat(model=self.settings.ollama_model,
                                   messages=[{"role": "user", "content": prompt}],
                                   format="json", options={"temperature": self.settings.temperature})
            return response["message"]["content"]
        raise RuntimeError("unsupported provider")

    def check(self) -> str:
        if self.name == "rules": return "rules-only mode: ready"
        if self.name == "openai":
            import os
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required when provider=openai")
            return "OpenAI configured for model %s" % self.settings.openai_model
        from ollama import Client
        Client(host=self.settings.ollama_base_url, timeout=self.settings.timeout_seconds).show(self.settings.ollama_model)
        return "Ollama configured for model %s" % self.settings.ollama_model
