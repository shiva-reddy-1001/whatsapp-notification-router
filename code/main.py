"""CLI entry point: build case files, route, validate, and write output.csv."""
import argparse
import logging
import random
import sys
from pathlib import Path

from code.config import Settings
from code.cache import SQLiteCache
from code.data_loader import Dataset
from code.media_processor import MediaProcessor
from code.providers import Classifier
from code.retrieval import retrieve
from code.features import apply as apply_features
from code.output_writer import write


def parse_args():
    parser = argparse.ArgumentParser(description="Personalized WhatsApp notification router")
    parser.add_argument("--input", help="must resolve to <dataset-dir>/messages.csv")
    parser.add_argument("--dataset-dir", help="directory containing participant-facing CSV files")
    parser.add_argument("--output", help="destination output.csv")
    parser.add_argument("--provider", choices=["auto", "openai", "ollama"])
    parser.add_argument("--check-config", action="store_true", help="validate selected provider without routing")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    settings = Settings.from_environment(args.dataset_dir, args.output, args.provider)
    random.seed(settings.seed)
    cache = SQLiteCache(settings.cache_path)
    classifier = Classifier(settings, cache)
    if args.check_config:
        print("provider=%s; %s" % (classifier.name, classifier.check()))
        cache.close()
        return 0
    input_path = (Path(args.input) if args.input else settings.dataset_dir / "messages.csv").resolve()
    if input_path != (settings.dataset_dir / "messages.csv").resolve():
        cache.close()
        raise ValueError("--input must be the participant-facing dataset/messages.csv file")
    classifier.check()
    dataset = Dataset(settings.dataset_dir, input_path)
    media = MediaProcessor(settings, cache)
    predictions = []
    for number, message in enumerate(dataset.messages, start=1):
        extracted, quality = media.extract(message.media_type or "", dataset.media_path(message)) \
            if message.media_type else ("", 1.0)
        content = "\n".join(piece for piece in [message.message_text, extracted] if piece).strip()
        case = dataset.case_file(message, content, quality)
        apply_features(case)
        case.evidence = retrieve(case, dataset.history_by_user[message.user_id], settings.max_evidence)
        predictions.append(classifier.classify(case))
        if number % 25 == 0 or number == len(dataset.messages):
            logging.info("routed %d/%d messages", number, len(dataset.messages))
    try:
        write(settings.output_path, predictions, [item.message_id for item in dataset.messages],
              [item.message.message_id for item in dataset.history])
        logging.info("wrote %d valid predictions to %s", len(predictions), settings.output_path)
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
