"""Cached embedding adapter for deterministic hybrid history retrieval."""
import hashlib
import logging
import math
from typing import Dict, Iterable, List, Optional

from .cache import SQLiteCache
from .config import Settings
from .reliability import RetryPolicy, run_with_retry


EMBEDDING_VERSION = "history-embedding-v1"


def _normalized(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class EmbeddingIndex:
    def __init__(self, settings: Settings, cache: SQLiteCache):
        self.settings = settings
        self.cache = cache
        self.provider = settings.resolved_embedding_provider()
        self.model = (settings.openai_embedding_model if self.provider == "openai"
                      else settings.ollama_embedding_model)

    def _key(self, text: str) -> str:
        material = "%s:%s:%s:%s" % (EMBEDDING_VERSION, self.provider, self.model, text)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def check(self) -> str:
        if self.provider == "none":
            return "embeddings disabled; lexical retrieval active"
        if self.provider == "openai":
            import os
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required when embedding provider=openai")
            return "OpenAI embeddings configured for model %s" % self.model
        from ollama import Client
        Client(host=self.settings.ollama_base_url,
               timeout=self.settings.timeout_seconds).show(self.model)
        return "Ollama embeddings configured for model %s" % self.model

    def prewarm(self, texts: Iterable[str]) -> None:
        unique = list(dict.fromkeys(text.strip() for text in texts if text and text.strip()))
        missing = [text for text in unique if self.cache.get("embeddings", self._key(text)) is None]
        for offset in range(0, len(missing), self.settings.embedding_batch_size):
            self._fetch_and_store(missing[offset:offset + self.settings.embedding_batch_size])
        logging.info("embedding index ready total=%d newly_cached=%d provider=%s model=%s",
                     len(unique), len(missing), self.provider, self.model)

    def vector(self, text: str) -> Optional[List[float]]:
        if self.provider == "none" or not text.strip():
            return None
        key = self._key(text.strip())
        cached = self.cache.get("embeddings", key)
        if cached is None:
            self._fetch_and_store([text.strip()])
            cached = self.cache.get("embeddings", key)
        return cached

    def similarity(self, left: str, right: str) -> float:
        left_vector, right_vector = self.vector(left), self.vector(right)
        if left_vector is None or right_vector is None:
            return 0.0
        # Vectors are normalized when stored, so cosine is a dot product.
        return max(-1.0, min(1.0, sum(a * b for a, b in zip(left_vector, right_vector))))

    def _fetch_and_store(self, texts: List[str]) -> None:
        if not texts or self.provider == "none":
            return

        def operation(_attempt: int) -> List[List[float]]:
            if self.provider == "openai":
                from openai import OpenAI
                client = OpenAI(timeout=self.settings.timeout_seconds, max_retries=0)
                response = client.embeddings.create(model=self.model, input=texts)
                ordered = sorted(response.data, key=lambda item: item.index)
                return [list(item.embedding) for item in ordered]
            from ollama import Client
            response = Client(host=self.settings.ollama_base_url,
                              timeout=self.settings.timeout_seconds).embed(
                                  model=self.model, input=texts)
            return [list(vector) for vector in response["embeddings"]]

        policy = RetryPolicy(self.settings.max_retries, self.settings.retry_mode,
                             self.settings.retry_base_seconds, self.settings.retry_max_seconds,
                             self.settings.retry_jitter_seconds,
                             self.settings.message_deadline_seconds)
        vectors = run_with_retry(operation, policy)
        if len(vectors) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        self.cache.put_many("embeddings", {
            self._key(text): _normalized(vector) for text, vector in zip(texts, vectors)
        })
