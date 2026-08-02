import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from code.cache import SQLiteCache
from code.embeddings import EmbeddingIndex


class FakeEmbeddingIndex(EmbeddingIndex):
    def __init__(self, settings, cache):
        super().__init__(settings, cache)
        self.fetches = []

    def _fetch_and_store(self, texts):
        self.fetches.append(list(texts))
        for text in texts:
            self.cache.put("embeddings", self._key(text), [1.0, 0.0])


class EmbeddingTests(unittest.TestCase):
    def test_prewarm_deduplicates_and_reuses_durable_vectors(self):
        settings = SimpleNamespace(
            resolved_embedding_provider=lambda: "ollama",
            openai_embedding_model="unused",
            ollama_embedding_model="fake-embed",
            embedding_batch_size=64,
        )
        with TemporaryDirectory() as directory:
            cache = SQLiteCache(Path(directory) / "router.sqlite")
            first = FakeEmbeddingIndex(settings, cache)
            first.prewarm(["alpha", "alpha", "beta", ""])
            self.assertEqual(first.fetches, [["alpha", "beta"]])
            second = FakeEmbeddingIndex(settings, cache)
            second.prewarm(["alpha", "beta"])
            self.assertEqual(second.fetches, [])
            cache.close()


if __name__ == "__main__":
    unittest.main()
