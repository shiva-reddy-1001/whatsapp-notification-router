"""Submission writer and contract checks."""
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

from .models import ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES, Prediction

COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def validate(predictions: List[Prediction], input_ids: Iterable[str], history_ids: Iterable[str]) -> None:
    expected = list(input_ids)
    actual = [prediction.message_id for prediction in predictions]
    if actual != expected:
        mismatch = next((index for index, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                        min(len(expected), len(actual)))
        wanted = expected[mismatch] if mismatch < len(expected) else "<end>"
        got = actual[mismatch] if mismatch < len(actual) else "<missing>"
        raise ValueError("prediction identity/order mismatch at row %d: expected %s, got %s" %
                         (mismatch + 1, wanted, got))
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
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                         dir=str(path.parent), prefix=".%s." % path.name,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for item in predictions:
                writer.writerow({"message_id": item.message_id, "action": item.action,
                                 "message_type": item.message_type, "reason": item.reason,
                                 "confidence": "%.2f" % item.confidence,
                                 "evidence_message_ids": ";".join(item.evidence_message_ids) or "none"})
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
