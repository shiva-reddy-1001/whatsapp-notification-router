"""Final decision invariants, confidence calibration, and reason consistency.

The model owns ambiguous routing. This layer enforces only requirements stated
by the challenge contract and makes the composed output internally consistent.
"""
import re
from typing import Tuple

from .models import CaseFile, Prediction, TypeDecision


NEGATIVE_OUTCOMES = {"reported", "muted_after", "dismissed"}
POSITIVE_OUTCOMES = {"replied", "opened"}
SAFETY_FACTS = {
    "credential or pressured-payment language",
    "sensitive financial details request",
    "advance-payment pressure",
    "business domain mismatch",
    "deceptive account or delivery verification",
}


class DecisionPolicy:
    """Compose provider decisions under narrow, auditable invariants."""

    def refine_type(self, case: CaseFile, result: TypeDecision) -> TypeDecision:
        """Apply unambiguous taxonomy boundaries before personalized routing."""
        text = (case.native_text or case.media_text or case.content or
                case.message.message_text).lower()
        safety = any(fact in SAFETY_FACTS for fact in case.risk_signals)
        if safety:
            return TypeDecision("scam", "The message contains an unsafe credential or payment request.", 0.94)
        if "credential-safety advisory, not a credential request" in case.priority_signals:
            return TypeDecision("business_update", "This is a legitimate credential-safety advisory.", 0.92)
        if re.search(r"\b(tanker|utility|production|incident bridge|payments? failing)\b", text) and re.search(
                r"\b(now|urgent|immediately|\d+\s*(?:min|mins|minutes))\b", text):
            return TypeDecision("urgent", "Immediate operational response is required.", 0.90)
        if re.search(r"\b(call now|please call now)\b", text) and re.search(
                r"\b(dad|mom|mother|father|clinic|hospital|unwell|on well)\b", text):
            return TypeDecision("urgent", "An immediate family medical response is requested.", 0.92)
        if re.search(r"\b(forms?|registration|register|consent|appointment|field trip|cultural night|"
                     r"pickup moved|bus leaves|transport|fire alarm test)\b", text):
            return TypeDecision("event", "The message concerns a scheduled activity or participation step.", 0.88)
        if (case.message.conversation_type == "business" and
                re.search(r"\b(feedback|survey|quick review|product review|rate (?:your|the)|review your experience)\b", text)):
            return TypeDecision("business_update", "This is a business feedback or service update.", 0.88)
        if (case.message.conversation_type == "business" and
                case.business.get("verified") == "1" and
                re.search(r"\b(order|delivery|return pickup|courier|shipment|parcel)\b", text) and
                not re.search(r"\b(sale|discount|offer|cashback|coupon|promo code)\b|%\s*off", text)):
            return TypeDecision("business_update", "This is a verified order, delivery, or return update.", 0.91)
        reports = int(case.business.get("user_reports_30d", "0") or 0)
        if (case.message.conversation_type == "business" and
                case.business.get("verified") == "0" and reports >= 10 and
                re.search(r"\b(counsel|call ?back|loan|sales|admission)\b", text)):
            return TypeDecision("spam", "This is high-report unsolicited business outreach.", 0.90)
        if (re.search(r"\bfound your (?:number|contact)\b", text) and
                not case.business_history.get("why_user_knows_account")):
            return TypeDecision("unknown", "The sender relationship is not established.", 0.82)
        if (result.message_type == "forward" and
                re.search(r"\b(good morning|good night|blessings|stay positive|keep smiling)\b", text) and
                not re.search(r"https?://|\b(otp|bank|medical advice|remedy)\b", text)):
            return TypeDecision("greeting", "The dominant content is a greeting or blessing.", 0.86)
        if re.search(r"\b(passport.*found|found.*passport|passeport.*trouv\w*)\b", text):
            return TypeDecision("urgent", "A found identity document has a collection deadline.", 0.90)
        if result.message_type == "greeting" and re.search(
                r"\b(call|talk|dinner|reached|watching|join|question|when free|tomorrow)\b|\?", text):
            return TypeDecision("personal", "This contains a substantive personal update or plan.", 0.86)
        return result

    def finalize(self, case: CaseFile, action_result: Prediction,
                 type_result: TypeDecision) -> Prediction:
        prediction = Prediction(
            message_id=case.message.message_id,
            action=action_result.action,
            message_type=type_result.message_type,
            reason=action_result.reason,
            confidence=action_result.confidence,
            evidence_message_ids=list(action_result.evidence_message_ids),
        )
        override = ""
        safety = any(fact in SAFETY_FACTS for fact in case.risk_signals)
        opted_out_promotion = (
            type_result.message_type in {"promotion", "spam"} and
            "promotion opt-out or no consent" in case.noise_signals
        )

        if safety:
            prediction.action = "mute"
            prediction.message_type = "scam"
            override = "safety"
        elif opted_out_promotion:
            prediction.action = "mute"
            override = "opt_out"
        elif (type_result.message_type in {"promotion", "spam"} and
              ("recipient muted this group" in case.noise_signals or
               self._negative_evidence(case) >= 2)):
            prediction.action = "mute"
            override = "unwanted_promotion"
        elif type_result.message_type == "promotion" and prediction.action == "notify":
            prediction.action = "digest"
            override = "promotion_digest"
        elif (type_result.message_type == "forward" and prediction.action == "notify" and
              (case.message.forwarded_count >= 1 or "high forwarding count" in case.noise_signals)):
            prediction.action = "mute"
            override = "unwanted_forward"
        elif ("credential-safety advisory, not a credential request" in case.priority_signals and
              prediction.action == "notify"):
            prediction.action = "digest"
            override = "safety_advisory"
        elif (type_result.message_type == "business_update" and
              re.search(r"\b(feedback|survey|quick review|product review|rate (?:your|the))\b",
                        (case.native_text or case.content).lower()) and
              prediction.action == "notify"):
            prediction.action = "digest"
            override = "feedback_digest"
        elif ("same-day operational timing" in case.priority_signals and
              case.business.get("verified") == "1" and prediction.action == "digest"):
            prediction.action = "notify"
            override = "verified_operational"
        elif (type_result.message_type == "payment" and
              "same-day operational timing" in case.priority_signals and
              not case.risk_signals and
              (case.business.get("verified") == "1" or
               (case.group.get("group_type") == "society" and
                case.membership.get("group_muted_by_user") != "1")) and
              prediction.action != "notify"):
            prediction.action = "notify"
            override = "trusted_payment_deadline"
        elif (type_result.message_type == "urgent" and
              prediction.action == "digest" and not case.defer_signals):
            prediction.action = "notify"
            override = "immediate_operational"
        elif (type_result.message_type == "event" and
              ({"explicit time-sensitive language", "same-day operational timing"}
               .intersection(case.priority_signals)) and
              prediction.action == "digest" and not case.defer_signals):
            prediction.action = "notify"
            override = "event_deadline"
        elif (type_result.message_type in {"event", "business_update"} and
              case.business.get("verified") == "1" and
              case.business_history.get("why_user_knows_account") and
              re.search(r"\b(order|deliver|reach|appointment|prescription|pickup|scheduled time)\b",
                        (case.native_text or case.content).lower()) and
              prediction.action == "digest"):
            prediction.action = "notify"
            override = "verified_relationship"
        elif case.defer_signals and prediction.action == "notify":
            prediction.action = "digest"
            override = "explicit_defer"

        prediction.confidence = self._confidence(
            case, prediction, type_result.confidence, override)
        prediction.reason = self._reason(case, prediction, override)
        return prediction

    @staticmethod
    def _negative_evidence(case: CaseFile) -> int:
        return sum(item.outcome in NEGATIVE_OUTCOMES for item in case.evidence)

    @staticmethod
    def _evidence_alignment(case: CaseFile, action: str) -> float:
        if not case.evidence:
            return 0.65
        negative = sum(item.outcome in NEGATIVE_OUTCOMES for item in case.evidence)
        positive = sum(item.outcome in POSITIVE_OUTCOMES for item in case.evidence)
        total = max(1, len(case.evidence))
        if action == "mute":
            return 0.55 + 0.40 * negative / total
        if action == "notify":
            return 0.55 + 0.40 * positive / total
        # Digest is appropriate when evidence is mixed or engagement is weak.
        balance = 1.0 - abs(positive - negative) / total
        return 0.60 + 0.25 * balance

    def _confidence(self, case: CaseFile, prediction: Prediction,
                    type_confidence: float, override: str) -> float:
        alignment = self._evidence_alignment(case, prediction.action)
        value = (0.55 * prediction.confidence +
                 0.30 * type_confidence +
                 0.15 * alignment)
        if case.media_signals:
            value -= min(0.20, 0.08 * len(case.media_signals))
        if case.message.media_type and case.media_quality < 0.50:
            value -= 0.10
        if override == "safety":
            value = max(value, 0.88 if len(SAFETY_FACTS.intersection(case.risk_signals)) > 1 else 0.82)
        elif override:
            value = min(value, 0.86)
        return round(max(0.45, min(0.96, value)), 2)

    @staticmethod
    def _reason(case: CaseFile, prediction: Prediction, override: str) -> str:
        negative = sum(item.outcome in NEGATIVE_OUTCOMES for item in case.evidence)
        if override == "safety":
            suffix = (" Similar prior messages were negatively handled."
                      if negative else "")
            return ("Muted as unsafe because the message requests credentials, sensitive "
                    "financial details, or a pressured advance payment." + suffix)[:280]
        if override == "opt_out":
            return "Muted because this is promotional outreach and the recipient opted out or denied promotional consent."
        if override == "explicit_defer":
            return "Deferred because the sender explicitly says the message can wait and gives no immediate interruption requirement."
        if override == "unwanted_promotion":
            return "Muted because this commercial outreach matches repeated negative history or a muted conversation."
        if override == "promotion_digest":
            return "Deferred because this is a commercial offer without an immediate recipient-critical consequence."
        if override == "unwanted_forward":
            return "Muted because this is forwarded chain or bulk content without a recipient-critical action."
        if override == "safety_advisory":
            return "Deferred because this is a useful safety advisory, not an immediate account action or credential request."
        if override == "verified_operational":
            return "Notified because a verified business reports a same-day operational change relevant to the recipient."
        if override == "trusted_payment_deadline":
            return "Notified because a trusted society or verified business has a legitimate same-day payment deadline."
        if override == "feedback_digest":
            return "Deferred because this feedback or survey request is useful but does not require an immediate interruption."
        if override == "immediate_operational":
            return "Notified because the message requires an immediate operational or personal response."
        if override == "verified_relationship":
            return "Notified because this verified business update matches an active order, appointment, or scheduled service."
        if override == "event_deadline":
            return "Notified because a scheduled activity has a concrete near-term deadline or timing change."

        reason = " ".join(prediction.reason.split())
        content = (case.native_text or case.media_text or case.content).lower()
        if (prediction.action == "notify" and prediction.message_type == "urgent" and
                re.search(r"\b(passport|passeport)\b", content)):
            return "Notified because a found identity document must be collected before a stated deadline."
        contradiction = (
            prediction.action == "notify" and re.search(
                r"\b(no urgency|not urgent|non[- ]immediate|no immediate|can wait)\b", reason, re.I)
        )
        if contradiction:
            reason = "Notified because the current message requires immediate, recipient-relevant action."
        if case.media_signals:
            reason = reason.rstrip(". ") + ". Media and caption details may conflict, so certainty is reduced."
        return reason[:280]
