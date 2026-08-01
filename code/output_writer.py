"""Submission writer and contract checks."""
import csv
import math
from pathlib import Path
from typing import Iterable, List

from .models import ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES, Prediction

COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def validate(predictions: List[Prediction], input_ids: Iterable[str], history_ids: Iterable[str]) -> None:
    expected = list(input_ids)
    actual = [prediction.message_id for prediction in predictions]
    if actual != expected:
        raise ValueError("predictions must have exactly one row in messages.csv order")
    known_history = set(history_ids)
    for prediction in predictions:
        if prediction.action not in ALLOWED_ACTIONS or prediction.message_type not in ALLOWED_MESSAGE_TYPES:
            raise ValueError("invalid label for %s" % prediction.message_id)
        if not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1:
            raise ValueError("invalid confidence for %s" % prediction.message_id)
        if not prediction.reason or not prediction.reason.strip():
            raise ValueError("empty reason for %s" % prediction.message_id)
        if len(prediction.evidence_message_ids) != len(set(prediction.evidence_message_ids)):
            raise ValueError("duplicate evidence ID for %s" % prediction.message_id)
        if any(item not in known_history for item in prediction.evidence_message_ids):
            raise ValueError("invalid evidence ID for %s" % prediction.message_id)


def write(path: Path, predictions: List[Prediction], input_ids: Iterable[str], history_ids: Iterable[str]) -> None:
    validate(predictions, input_ids, history_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for item in predictions:
            writer.writerow({"message_id": item.message_id, "action": item.action,
                             "message_type": item.message_type, "reason": item.reason,
                             "confidence": "%.2f" % item.confidence,
                             "evidence_message_ids": ";".join(item.evidence_message_ids) or "none"})
