"""One-time cached enrichment of historical image and voice messages."""
import logging

from .data_loader import Dataset
from .media_processor import MediaProcessor


def enrich_historical_media(dataset: Dataset, processor: MediaProcessor) -> int:
    enriched = 0
    total = sum(1 for item in dataset.history if item.message.media_type)
    for item in dataset.history:
        message = item.message
        if not message.media_type:
            continue
        extracted, _quality = processor.extract(message.media_type,
                                                dataset.media_path(message))
        if extracted:
            label = "historical %s extraction" % message.media_type
            message.message_text = "\n".join(
                part for part in (message.message_text, "[%s]\n%s" % (label, extracted)) if part)
            enriched += 1
    if total:
        logging.info("historical media ready enriched=%d total=%d", enriched, total)
    return enriched
