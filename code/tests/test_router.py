import unittest

from code.models import CaseFile, Message, Prediction
from code.output_writer import validate
from code.retrieval import retrieve
from code.router import deterministic_route


def case(text, conversation="personal", **overrides):
    message = Message("incoming_1", "u_1", conversation, None, None, None,
                      "2026-08-01 10:00", text, None, None, 0)
    result = CaseFile(message, text, 1.0, {}, {}, {}, {}, {}, {})
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class RouterTests(unittest.TestCase):
    def test_credential_request_is_muted_as_scam(self):
        prediction = deterministic_route(case("Reply with the OTP to keep your account active."))
        self.assertEqual((prediction.action, prediction.message_type), ("mute", "scam"))

    def test_opted_out_promotion_is_muted(self):
        prediction = deterministic_route(case("Limited sale, 50% off today!", conversation="business",
                                               business_history={"allows_promotions": "0"}))
        self.assertEqual((prediction.action, prediction.message_type), ("mute", "promotion"))

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


if __name__ == "__main__":
    unittest.main()
