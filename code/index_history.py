"""CLI for explicitly precomputing historical media analyses and embeddings."""
import argparse
import logging

from .cache import SQLiteCache
from .config import Settings
from .data_loader import Dataset
from .embeddings import EmbeddingIndex
from .history_media import enrich_historical_media
from .media_processor import MediaProcessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the durable historical retrieval cache")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_environment(args.dataset_dir)
    cache = SQLiteCache(settings.cache_path)
    try:
        dataset = Dataset(settings.dataset_dir)
        media = MediaProcessor(settings, cache)
        embeddings = EmbeddingIndex(settings, cache)
        print("%s; %s" % (media.check(), embeddings.check()))
        enrich_historical_media(dataset, media)
        embeddings.prewarm(item.message.message_text for item in dataset.history)
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    main()
