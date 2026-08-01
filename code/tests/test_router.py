import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from code.cache import SQLiteCache
from code.models import CaseFile, Message, Prediction
from code.output_writer import validate
from code.retrieval import retrieve
from code.features import extract
from code.providers import _cached_prediction, _parse


def case(text, conversation="personal", **overrides):
    message = Message("incoming_1", "u_1", conversation, None, None, None,
                      "2026-08-01 10:00", text, None, None, 0)
    result = CaseFile(message, text, 1.0, {}, {}, {}, {}, {}, {})
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class RouterTests(unittest.TestCase):
    def test_credential_request_has_a_scam_feature(self):
        self.assertIn("credential or pressured-payment language",
                      extract(case("Reply with the OTP to keep your account active."))["risk"])

    def test_opted_out_promotion_has_a_consent_feature(self):
        facts = extract(case("Limited sale, 50% off today!", conversation="business",
                             business_history={"allows_promotions": "0"}))
        self.assertIn("promotion opt-out or no consent", facts["noise"])

    def test_output_contract_rejects_unknown_evidence(self):
        prediction = Prediction("incoming_1", "digest", "personal", "Can wait.", .6, ["missing"])
        with self.assertRaises(ValueError):
            validate([prediction], ["incoming_1"], [])

    def test_output_contract_accepts_none_evidence(self):
        prediction = Prediction("incoming_1", "digest", "personal", "Can wait.", .6, [])
        validate([prediction], ["incoming_1"], [])

    def test_unrelated_history_is_not_cited_as_evidence(self):
        historic = Message("historic_1", "u_1", "personal", None, None, None,
                           "2026-07-01", "Completely unrelated weather chat.", None, None, 0)
        from code.models import HistoryItem
        result = retrieve(case("Payment is due today."), [HistoryItem(historic)], 3)
        self.assertEqual(result, [])

    def test_sqlite_cache_persists_json_values(self):
        with TemporaryDirectory() as directory:
            cache = SQLiteCache(Path(directory) / "router.sqlite")
            cache.put("media", "key", {"text": "hello", "quality": .7})
            self.assertEqual(cache.get("media", "key"), {"text": "hello", "quality": .7})
            cache.close()

    def test_local_model_ten_point_confidence_is_normalized(self):
        result = _parse(case("Urgent account warning"),
                        '{"action":"notify","message_type":"urgent","reason":"Time-sensitive warning.","confidence":9,"evidence_message_ids":[]}')
        self.assertIsNotNone(result)
        self.assertEqual(result.confidence, .9)

    def test_cached_prediction_uses_current_message_identity(self):
        current = case("Same content on another incoming row")
        cached = {"message_id": "stale_id", "action": "digest", "message_type": "personal",
                  "reason": "Useful later.", "confidence": .7, "evidence_message_ids": []}
        self.assertEqual(_cached_prediction(current, cached).message_id, "incoming_1")


if __name__ == "__main__":
    unittest.main()
