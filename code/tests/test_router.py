import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from code.cache import SQLiteCache
from code.models import (CaseFile, HistoryItem, Message, Prediction,
                         RetrievedEvidence, TypeDecision)
from code.output_writer import validate, write
from code.retrieval import retrieve
from code.features import extract
from code.providers import _authoritative_type, _cached_prediction, _parse
from code.decision_policy import DecisionPolicy


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

    def test_credential_safety_advisory_is_not_a_scam_feature(self):
        facts = extract(case("We will never ask you to share OTP or payment details."))
        self.assertNotIn("credential or pressured-payment language", facts["risk"])
        self.assertIn("credential-safety advisory, not a credential request", facts["priority"])

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
            cache.clear()
            self.assertIsNone(cache.get("media", "key"))
            cache.close()

    def test_output_is_atomically_written_with_submission_header(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "output.csv"
            prediction = Prediction("incoming_1", "digest", "personal", "Can wait.", .6, [])
            write(path, [prediction], ["incoming_1"], [])
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith(
                "message_id,action,message_type,reason,confidence,evidence_message_ids\n"))
            self.assertNotIn("\r\n", text)

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

    def test_action_stage_cannot_change_preclassified_type(self):
        result = _parse(
            case("School bus leaves early."),
            '{"action":"notify","reason":"Timing changed.","confidence":0.9,'
            '"evidence_message_ids":[],"message_type":"personal"}')
        result = _authoritative_type(result, "event")
        self.assertEqual(result.message_type, "event")

    def test_safety_invariant_overrides_urgent_notification(self):
        current = case("Reply with the OTP now or your account will be blocked.")
        facts = extract(current)
        current.risk_signals = facts["risk"]
        raw = Prediction("incoming_1", "notify", "urgent", "Urgent account issue.", .95, [])
        result = DecisionPolicy().finalize(
            current, raw, TypeDecision("urgent", "Immediate wording.", .90))
        self.assertEqual((result.action, result.message_type), ("mute", "scam"))
        self.assertIn("unsafe", result.reason)

    def test_financial_details_request_is_a_safety_fact(self):
        facts = extract(case("Claim benefits by sharing your account number today."))
        self.assertIn("sensitive financial details request", facts["risk"])

    def test_shortened_account_verification_threat_is_a_safety_fact(self):
        facts = extract(case("Open bit.ly/verify now for the final account check today."))
        self.assertIn("deceptive account or delivery verification", facts["risk"])

    def test_delivery_reattempt_fee_link_is_a_safety_fact(self):
        facts = extract(case(
            "Delivery failed. Pay a reattempt charge at parcel-delivery.in before return."))
        self.assertTrue(facts["risk"])

    def test_hinglish_otp_request_is_a_safety_fact(self):
        facts = extract(case("Account block ho jayega, OTP abhi batao. Jaldi karo."))
        self.assertIn("credential or pressured-payment language", facts["risk"])

    def test_no_payment_or_otp_required_is_not_a_scam_fact(self):
        facts = extract(case(
            "Verified delivery today. Keep ID ready; no payment or OTP is required."))
        self.assertEqual(facts["risk"], [])

    def test_do_not_use_resident_payment_links_is_not_a_scam_fact(self):
        facts = extract(case(
            "Payment due today. Please don't use any payment link shared by residents."))
        self.assertEqual(facts["risk"], [])

    def test_legitimate_payment_link_alone_does_not_force_scam(self):
        facts = extract(case("Maintenance payment is due today. Use this link and send receipt."))
        self.assertEqual(facts["risk"], [])

    def test_legitimate_credential_advisory_is_not_forced_to_scam(self):
        current = case("Safety reminder: never share your OTP or PIN with anyone.")
        current.risk_signals = extract(current)["risk"]
        raw = Prediction("incoming_1", "digest", "business_update", "Useful advisory.", .8, [])
        result = DecisionPolicy().finalize(
            current, raw, TypeDecision("business_update", "Safety advisory.", .9))
        self.assertEqual((result.action, result.message_type), ("digest", "business_update"))

    def test_explicit_no_urgency_cannot_notify(self):
        current = case("Nothing urgent. Review it tomorrow when free.")
        facts = extract(current)
        current.defer_signals = facts["defer"]
        raw = Prediction("incoming_1", "notify", "personal", "Relevant request.", .9, [])
        result = DecisionPolicy().finalize(
            current, raw, TypeDecision("personal", "Personal request.", .85))
        self.assertEqual(result.action, "digest")

    def test_short_external_wait_limit_is_not_sender_deferral(self):
        current = case("The tanker can wait 20 mins max; fill drinking water now.")
        self.assertEqual(extract(current)["defer"], [])

    def test_opted_out_promotion_cannot_interrupt(self):
        current = case("Limited travel offer. Reply STOP to unsubscribe.", conversation="business",
                       business_history={"allows_promotions": "0"})
        facts = extract(current)
        current.noise_signals = facts["noise"]
        raw = Prediction("incoming_1", "notify", "promotion", "Limited offer.", .9, [])
        result = DecisionPolicy().finalize(
            current, raw, TypeDecision("promotion", "Commercial offer.", .9))
        self.assertEqual(result.action, "mute")

    def test_stale_media_deadline_is_exposed_as_uncertainty(self):
        current = case("Applications close today.")
        current.native_text = current.message.message_text
        current.media_text = "Research internship programme 2023-24"
        facts = extract(current)
        self.assertIn("media contains a past year while caption asserts a current deadline",
                      facts["media"])

    def test_negative_evidence_supports_safety_reason(self):
        current = case("Send OTP now.")
        current.risk_signals = ["credential or pressured-payment language"]
        current.evidence = [RetrievedEvidence(
            "h1", "Send OTP.", .95, "reported", "similar prior content")]
        raw = Prediction("incoming_1", "notify", "urgent", "Urgent.", .95, ["h1"])
        result = DecisionPolicy().finalize(
            current, raw, TypeDecision("urgent", "Urgent.", .9))
        self.assertIn("negatively handled", result.reason)

    def test_same_group_alone_does_not_make_unrelated_history_evidence(self):
        current = case("Account verification link is urgent.", conversation="group")
        current.message.group_id = "g1"
        historic = Message("historic_1", "u_1", "group", "g1", None, None,
                           "2026-07-01", "Selling a denim jacket this weekend.", None, None, 0)
        self.assertEqual(retrieve(current, [HistoryItem(historic)], 3), [])

    def test_business_safety_advisory_type_and_action_remain_legitimate(self):
        current = case("We will never ask you to share OTP or payment details.",
                       conversation="business")
        facts = extract(current)
        current.risk_signals, current.priority_signals = facts["risk"], facts["priority"]
        policy = DecisionPolicy()
        kind = policy.refine_type(current, TypeDecision("scam", "Contains OTP.", .9))
        result = policy.finalize(
            current, Prediction("incoming_1", "notify", "scam", "Important warning.", .9, []),
            kind)
        self.assertEqual((result.action, result.message_type), ("digest", "business_update"))

    def test_business_feedback_is_a_business_update_not_promotion(self):
        current = case("Please rate your purchase experience in this short survey.",
                       conversation="business")
        kind = DecisionPolicy().refine_type(
            current, TypeDecision("promotion", "Outreach.", .8))
        self.assertEqual(kind.message_type, "business_update")

    def test_verified_return_pickup_is_a_business_update(self):
        current = case("Return pickup today. Keep the item packed for the courier.",
                       conversation="business", business={"verified": "1"})
        kind = DecisionPolicy().refine_type(
            current, TypeDecision("promotion", "Commercial wording.", .8))
        self.assertEqual(kind.message_type, "business_update")

    def test_verified_discount_for_first_order_remains_promotion(self):
        current = case("Get 50% off your first order. Use code TRY50.",
                       conversation="business", business={"verified": "1"})
        kind = DecisionPolicy().refine_type(
            current, TypeDecision("promotion", "Commercial discount.", .9))
        self.assertEqual(kind.message_type, "promotion")

    def test_feedback_request_cannot_interrupt(self):
        current = case("Can you fill a quick product review?", conversation="business")
        result = DecisionPolicy().finalize(
            current, Prediction("incoming_1", "notify", "business_update", "Please respond.", .9, []),
            TypeDecision("business_update", "Feedback request.", .9))
        self.assertEqual(result.action, "digest")

    def test_found_passport_with_deadline_is_urgent(self):
        current = case("Votre passeport a ete trouve; recuperer avant 18h.")
        kind = DecisionPolicy().refine_type(current, TypeDecision("scam", "Unknown.", .7))
        self.assertEqual(kind.message_type, "urgent")

    def test_substantive_plan_is_not_reduced_to_greeting(self):
        current = case("Anyone watching the match tonight? Join when free.")
        kind = DecisionPolicy().refine_type(
            current, TypeDecision("greeting", "Casual message.", .8))
        self.assertEqual(kind.message_type, "personal")

    def test_immediate_family_medical_call_is_urgent(self):
        current = case("Please call now. Dad is unwell and we are going to the clinic.")
        kind = DecisionPolicy().refine_type(
            current, TypeDecision("forward", "Unclear audio.", .6))
        self.assertEqual(kind.message_type, "urgent")

    def test_promotion_without_critical_consequence_cannot_notify(self):
        current = case("Selling a cycle helmet. Pickup this weekend.", conversation="group")
        result = DecisionPolicy().finalize(
            current, Prediction("incoming_1", "notify", "promotion", "Available now.", .9, []),
            TypeDecision("promotion", "Commercial listing.", .9))
        self.assertEqual(result.action, "digest")

    def test_verified_same_day_delivery_can_notify(self):
        current = case("Your order will reach the local hub today.", conversation="business",
                       business={"verified": "1"})
        current.priority_signals = ["same-day operational timing"]
        result = DecisionPolicy().finalize(
            current, Prediction("incoming_1", "digest", "business_update", "Order update.", .8, []),
            TypeDecision("business_update", "Order status.", .9))
        self.assertEqual(result.action, "notify")

    def test_trusted_society_payment_deadline_can_notify(self):
        current = case("Payment due today. Complete before 5 PM.", conversation="group",
                       group={"group_type": "society"},
                       membership={"group_muted_by_user": "0"})
        current.priority_signals = ["same-day operational timing"]
        result = DecisionPolicy().finalize(
            current, Prediction("incoming_1", "mute", "payment", "Suspicious.", .8, []),
            TypeDecision("payment", "Legitimate maintenance payment.", .9))
        self.assertEqual(result.action, "notify")

    def test_passport_notification_reason_matches_urgent_type(self):
        current = case("Votre passeport a ete trouve; recuperer avant 18h.")
        result = DecisionPolicy().finalize(
            current, Prediction("incoming_1", "notify", "scam", "Friendly greeting.", .8, []),
            TypeDecision("urgent", "Found document.", .9))
        self.assertIn("identity document", result.reason)

    def test_forwarded_blessing_remains_a_greeting(self):
        current = case("Good morning all. Stay positive and share blessings with everyone.")
        kind = DecisionPolicy().refine_type(
            current, TypeDecision("forward", "Asked to share.", .8))
        self.assertEqual(kind.message_type, "greeting")


if __name__ == "__main__":
    unittest.main()
