"""Safety-first deterministic routing and structured-case helpers."""
import re
from .features import PAYMENT_PATTERNS, PROMOTION_PATTERNS, URGENT_PATTERNS, apply, matches
from .models import CaseFile, Prediction
EVENT_PATTERNS = r"\b(meeting|bus|school|event|schedule|maintenance|water|plumber|class|appointment|prescription|pickup)\b"


def _contains(pattern: str, text: str) -> bool:
    return matches(pattern, text)


def enrich_signals(case: CaseFile) -> None:
    apply(case)


def _message_type(case: CaseFile) -> str:
    text = case.content.lower()
    if case.risk_signals: return "scam"
    if re.fullmatch(r"\s*(good morning|good evening|good afternoon|hello|hi)[!. ,]*(everyone|all)?[!. ,]*", text):
        return "greeting"
    if case.message.forwarded_count >= 5: return "forward"
    if _contains(PROMOTION_PATTERNS, text): return "promotion"
    if _contains(PAYMENT_PATTERNS, text): return "payment"
    if _contains(EVENT_PATTERNS, text): return "event"
    if case.message.conversation_type == "business": return "business_update"
    if "direct mention" in case.priority_signals: return "personal"
    if _contains(URGENT_PATTERNS, text): return "urgent"
    if case.message.conversation_type == "personal": return "personal"
    if re.fullmatch(r"\s*(hi|hello|good morning|good evening)[!. ]*", text): return "greeting"
    return "unknown"


def deterministic_route(case: CaseFile) -> Prediction:
    """Route clear cases. Ambiguous cases get a conservative digest default."""
    enrich_signals(case)
    kind = _message_type(case)
    evidence_ids = [item.message_id for item in case.evidence]
    if case.risk_signals:
        return Prediction(case.message.message_id, "mute", "scam",
                          "Muted because it contains a high-risk scam or credential signal.",
                          0.91, evidence_ids)
    if "promotion opt-out or no consent" in case.noise_signals:
        return Prediction(case.message.message_id, "mute", "promotion",
                          "Muted because this is a promotion from a source the user has not opted into.",
                          0.86, evidence_ids)
    if case.message.forwarded_count >= 5 and not case.priority_signals:
        return Prediction(case.message.message_id, "mute", "forward",
                          "Muted because it is heavily forwarded without a recipient-specific need.",
                          0.78, evidence_ids)
    urgent = bool(case.priority_signals) and kind in {"urgent", "event", "payment", "personal", "business_update"}
    if urgent and "recipient muted this group" not in case.noise_signals:
        signal = case.priority_signals[0]
        return Prediction(case.message.message_id, "notify", kind,
                          "Notified because it has %s and needs prompt attention." % signal,
                          0.80, evidence_ids)
    if kind == "promotion":
        return Prediction(case.message.message_id, "digest", kind,
                          "A legitimate promotion can wait for a later digest.", 0.67, evidence_ids)
    if "recipient muted this group" in case.noise_signals:
        return Prediction(case.message.message_id, "digest", kind,
                          "The group is muted and this update has no clear immediate interruption need.",
                          0.72, evidence_ids)
    if kind == "greeting":
        return Prediction(case.message.message_id, "mute", kind,
                          "Muted as a low-priority greeting without an action request.", 0.66, evidence_ids)
    return Prediction(case.message.message_id, "digest", kind,
                      "Useful but not clearly time-sensitive, so it can be shown in a digest.",
                      0.58 if case.media_quality < 0.3 else 0.67, evidence_ids)


def case_file_prompt(case: CaseFile) -> str:
    """Compact prompt shared by provider adapters; never contains a secret."""
    history = "\n".join("- %s (%s): %s" % (item.message_id, item.outcome, item.text)
                        for item in case.evidence) or "- none"
    return """Classify this WhatsApp notification case. Safety beats engagement. Notify only for an
immediate, recipient-relevant interruption. Return JSON only with action, message_type,
reason, confidence, and evidence_message_ids. action is notify/digest/mute; message_type is
personal/urgent/event/payment/business_update/promotion/greeting/forward/spam/scam/unknown.
Only cite supplied evidence IDs.\n\nMessage: {content}\nConversation: {conversation}\nRisk: {risk}\nPriority: {priority}\nNoise: {noise}\nHistory:\n{history}""".format(
        content=case.content[:4000], conversation=case.message.conversation_type,
        risk=", ".join(case.risk_signals) or "none",
        priority=", ".join(case.priority_signals) or "none",
        noise=", ".join(case.noise_signals) or "none", history=history)
