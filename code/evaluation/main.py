"""Evaluate deterministic/hybrid output against solved sample messages."""
import argparse
from collections import Counter
from pathlib import Path

from code.config import Settings
from code.data_loader import Dataset, as_message, read_csv
from code.providers import Classifier
from code.retrieval import retrieve
from code.media_processor import MediaProcessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--provider", default="rules", choices=["auto", "openai", "ollama", "rules"])
    args = parser.parse_args()
    settings = Settings.from_environment(args.dataset_dir, provider=args.provider)
    dataset, classifier, media = Dataset(settings.dataset_dir), Classifier(settings), MediaProcessor(settings)
    rows = read_csv(Path(args.dataset_dir) / "sample_messages.csv")
    action_ok = type_ok = 0
    errors = Counter()
    slices = {"conversation_type": {}, "expected_type": {}}
    for row in rows:
        message = as_message(row)
        extracted, quality = media.extract(message.media_type or "", dataset.media_path(message)) \
            if message.media_type else ("", 1.0)
        content = "\n".join(piece for piece in [message.message_text, extracted] if piece)
        case = dataset.case_file(message, content, quality)
        case.evidence = retrieve(case, dataset.history_by_user[message.user_id], settings.max_evidence)
        prediction = classifier.classify(case)
        action_ok += prediction.action == row["action"]
        type_ok += prediction.message_type == row["message_type"]
        for dimension, value in (("conversation_type", message.conversation_type),
                                 ("expected_type", row["message_type"])):
            bucket = slices[dimension].setdefault(value, [0, 0, 0])
            bucket[0] += 1
            bucket[1] += prediction.action == row["action"]
            bucket[2] += prediction.message_type == row["message_type"]
        if prediction.action != row["action"] or prediction.message_type != row["message_type"]:
            errors[(row["action"], prediction.action)] += 1
    total = len(rows)
    print("samples=%d action_accuracy=%.3f type_accuracy=%.3f" %
          (total, action_ok / total, type_ok / total))
    for (expected, actual), count in errors.most_common(10):
        print("%s -> %s: %d" % (expected, actual, count))
    for dimension, values in slices.items():
        print("%s:" % dimension)
        for value, (count, actions, types) in sorted(values.items()):
            print("  %s n=%d action=%.3f type=%.3f" % (value, count, actions / count, types / count))


if __name__ == "__main__":
    main()
