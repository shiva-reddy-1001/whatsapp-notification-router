"""Provider adapters with strict structured-output validation."""
import json
import re
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Optional

from .config import Settings
from .cache import SQLiteCache
from .models import (ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES, CaseFile,
                     Prediction, TypeDecision)
from .prompting import (PROMPT_VERSION, TYPE_PROMPT_VERSION,
                        build_casefile_prompt, build_type_prompt)
from .reliability import InvalidModelResponse, RetryPolicy, run_with_retry
from .decision_policy import DecisionPolicy

POLICY_VERSION = "decision-policy-v7-trusted-deadlines-and-reason-consistency"


def _cached_prediction(case: CaseFile, cached: dict) -> Prediction:
    """Reuse the decision content, never another incoming message's identity."""
    return Prediction(message_id=case.message.message_id,
                      action=cached["action"], message_type=cached["message_type"],
                      reason=cached["reason"], confidence=float(cached["confidence"]),
                      evidence_message_ids=list(cached.get("evidence_message_ids", [])))


def _parse(case: CaseFile, raw: str) -> Optional[Prediction]:
    try:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        action = str(data["action"]).strip().lower()
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
        if action not in ALLOWED_ACTIONS or not 0 <= confidence <= 1:
            return None
        reason = str(data["reason"]).strip().replace("\n", " ")[:280]
        if not reason:
            return None
        return Prediction(case.message.message_id, action, "unknown", reason, confidence, evidence)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _parse_type(raw: str) -> Optional[TypeDecision]:
    try:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        kind = str(data["message_type"]).strip().lower()
        confidence = float(data["confidence"])
        reason = str(data["reason"]).strip().replace("\n", " ")[:240]
        if kind not in ALLOWED_MESSAGE_TYPES or not 0 <= confidence <= 1 or not reason:
            return None
        return TypeDecision(kind, reason, confidence)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _authoritative_type(prediction: Prediction, fixed_type: str) -> Prediction:
    """Compose the specialist type with the joint stage's routing decision."""
    prediction.message_type = fixed_type
    return prediction


class ProviderStrategy(ABC):
    """Vendor-neutral structured generation strategy."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def call(self, prompt: str, schema: Optional[dict] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def check(self) -> str:
        raise NotImplementedError


class OpenAIProvider(ProviderStrategy):
    name = "openai"

    def call(self, prompt: str, schema: Optional[dict] = None) -> str:
        from openai import OpenAI
        client = OpenAI(timeout=self.settings.timeout_seconds, max_retries=0)
        structured = ({"format": {"type": "json_schema", "name": "router_response",
                                   "schema": schema, "strict": True}}
                      if schema else None)
        response = client.responses.create(model=self.settings.openai_model, input=prompt,
                                           temperature=self.settings.temperature,
                                           **({"text": structured} if structured else {}))
        return response.output_text

    def check(self) -> str:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when provider=openai")
        return "OpenAI configured for model %s" % self.settings.openai_model


class OllamaProvider(ProviderStrategy):
    name = "ollama"

    def call(self, prompt: str, schema: Optional[dict] = None) -> str:
        from ollama import Client
        client = Client(host=self.settings.ollama_base_url,
                        timeout=self.settings.timeout_seconds)
        response = client.chat(model=self.settings.ollama_model,
                               messages=[{"role": "user", "content": prompt}],
                               format=schema or Classifier._classification_schema(),
                               options={"temperature": self.settings.temperature})
        return response["message"]["content"]

    def check(self) -> str:
        from ollama import Client
        Client(host=self.settings.ollama_base_url,
               timeout=self.settings.timeout_seconds).show(self.settings.ollama_model)
        return "Ollama configured for model %s" % self.settings.ollama_model


def provider_strategy(settings: Settings) -> ProviderStrategy:
    name = settings.resolved_provider()
    if name == "openai":
        return OpenAIProvider(settings)
    if name == "ollama":
        return OllamaProvider(settings)
    raise RuntimeError("unsupported provider: %s" % name)


class Classifier:
    def __init__(self, settings: Settings, cache: SQLiteCache = None,
                 provider: ProviderStrategy = None,
                 policy: DecisionPolicy = None):
        self.settings = settings
        self.provider = provider or provider_strategy(settings)
        self.name = self.provider.name
        self.cache = cache
        self.policy = policy or DecisionPolicy()

    def classify(self, case: CaseFile) -> Prediction:
        fixed_type = self.policy.refine_type(case, self.classify_type(case))
        prompt = build_casefile_prompt(case)
        key = hashlib.sha256((self.name + self.settings.openai_model + self.settings.ollama_model +
                              PROMPT_VERSION + TYPE_PROMPT_VERSION + POLICY_VERSION + prompt +
                              fixed_type.message_type + str(fixed_type.confidence)).encode()).hexdigest()
        cached = self.cache.get("predictions", key) if self.cache else None
        if cached:
            return _cached_prediction(case, cached)
        def operation(attempt: int) -> Prediction:
            repair = "" if not attempt else (
                "\nA previous response was invalid. Return exactly the requested JSON schema; "
                "confidence must be 0.0 to 1.0.")
            result = _parse(case, self.provider.call(prompt + repair,
                                             self._classification_schema()))
            if not result:
                raise InvalidModelResponse("provider returned invalid structured classification")
            return self.policy.finalize(case, result, fixed_type)

        policy = RetryPolicy(self.settings.max_retries, self.settings.retry_mode,
                             self.settings.retry_base_seconds, self.settings.retry_max_seconds,
                             self.settings.retry_jitter_seconds,
                             self.settings.message_deadline_seconds)
        try:
            result = run_with_retry(
                operation, policy,
                on_retry=lambda number, error, delay: logging.warning(
                    "retrying message %s attempt=%d category=%s delay=%.2fs",
                    case.message.message_id, number, type(error).__name__, delay))
        except Exception as error:
            raise RuntimeError("classification failed for %s: %s" %
                               (case.message.message_id, error)) from error
        if self.cache:
            self.cache.put("predictions", key, {"action": result.action,
                           "message_type": result.message_type, "reason": result.reason,
                           "confidence": result.confidence,
                           "evidence_message_ids": result.evidence_message_ids})
        return result

    def classify_type(self, case: CaseFile) -> TypeDecision:
        prompt = build_type_prompt(case)
        key = hashlib.sha256((self.name + self.settings.openai_model +
                              self.settings.ollama_model + TYPE_PROMPT_VERSION +
                              prompt).encode()).hexdigest()
        cached = self.cache.get("message_types", key) if self.cache else None
        if cached:
            return TypeDecision(str(cached["message_type"]),
                                str(cached.get("reason", "Cached semantic classification.")),
                                float(cached.get("confidence", 0.70)))

        def operation(attempt: int) -> TypeDecision:
            repair = "" if not attempt else (
                "\nA previous response was invalid. Return only message_type, reason, "
                "and confidence in the requested JSON schema.")
            kind = _parse_type(self.provider.call(prompt + repair, self._type_schema()))
            if not kind:
                raise InvalidModelResponse("provider returned invalid structured message type")
            return kind

        policy = RetryPolicy(self.settings.max_retries, self.settings.retry_mode,
                             self.settings.retry_base_seconds, self.settings.retry_max_seconds,
                             self.settings.retry_jitter_seconds,
                             self.settings.message_deadline_seconds)
        try:
            kind = run_with_retry(operation, policy)
        except Exception as error:
            raise RuntimeError("type classification failed for %s: %s" %
                               (case.message.message_id, error)) from error
        if self.cache:
            self.cache.put("message_types", key, {"message_type": kind.message_type,
                           "reason": kind.reason, "confidence": kind.confidence})
        return kind

    @staticmethod
    def _classification_schema() -> dict:
        return {"type": "object", "additionalProperties": False,
                "required": ["action", "reason", "confidence", "evidence_message_ids"],
                "properties": {"action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                               "reason": {"type": "string"},
                               "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                               "evidence_message_ids": {"type": "array", "items": {"type": "string"}}}}

    @staticmethod
    def _type_schema() -> dict:
        return {"type": "object", "additionalProperties": False,
                "required": ["message_type", "reason", "confidence"],
                "properties": {"message_type": {"type": "string", "enum": sorted(ALLOWED_MESSAGE_TYPES)},
                               "reason": {"type": "string"},
                               "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}

    def check(self) -> str:
        return self.provider.check()
