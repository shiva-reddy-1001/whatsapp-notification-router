"""Pure, auditable feature extraction separated from routing policy."""
import re
from datetime import datetime
from typing import Dict, List

from .models import CaseFile

SCAM_PATTERNS = (
    r"\b(otp|one.time password|pin|password|cvv|login code)\b.*\b(share|send|enter|verify|confirm|reply|provide)\b",
    r"\b(reply|send|share|confirm|provide)\b.*\b(otp|one.time password|pin|password|cvv|login code)\b",
    r"\b(pay|payment|fee|refund|reward)\b.*\b(otp|pin|password|wallet details|card details|processing fee)\b",
    r"\b(account|bank|package|delivery)\b.*\b(blocked|suspended|release|verify)\b",
    r"\b(bank details|account number|wallet details|card details)\b.*\b(send|shar\w*|enter|fill|confirm|verify)\b",
    r"\b(send|shar\w*|enter|fill|confirm|verify)\b.*\b(bank details|account number|wallet details|card details)\b",
    r"\b(processing fee|token)\b.*\b(pay|payment|release|papers|registry)\b",
    r"\b(otp|verification code|login code)\b.*\b(batao|daal|dalo|jaldi|abhi)\b",
    r"\b(delivery|parcel|package)\b.*\b(reattempt|small fee|charge)\b.*\b(link|verify|verification|\.in)\b",
)
URGENT_PATTERNS = r"\b(urgent|emergency|immediately|right now|eod|deadline|last[- ]minute|within \d+|\d+\s*(?:min|mins|minutes|hours?|hrs?))\b"
PROMOTION_PATTERNS = r"\b(sale|offer|discount|coupon|cashback|buy now|limited offer|% off)\b"
PAYMENT_PATTERNS = r"\b(payment|invoice|bill|due|receipt|transaction|refund)\b"
DEFER_PATTERNS = (
    r"\b(no rush|nothing urgent|not urgent|whenever convenient|when convenient)\b",
    r"\b(whenever you get time|no pressure|this can wait|it can wait|not time[- ]sensitive)\b",
    r"\b(call|check|review|talk|reply|send)\b.{0,30}\b(later|tomorrow|when free)\b",
)
PROMPT_INJECTION_PATTERNS = (
    r"\b(system note|assistant instruction|routing override)\b",
    r"\bignore\b.{0,50}\b(instruction|risk|sender|classif)\w*",
)


def matches(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL))


def extract(case: CaseFile) -> Dict[str, List[str]]:
    """Return facts; policy decides how much each fact should matter."""
    text, message = case.content.lower(), case.message
    risk, priority, noise = [], [], []
    credential_advisory = matches(
        r"\b(never|do not|don't|will not|won't)\b.{0,60}\b(ask|share|provide|send)\b.{0,40}"
        r"\b(otp|password|pin|cvv|payment details|login code)\b", text)
    no_credential_required = matches(
        r"\b(no|not)\b.{0,40}\b(payment|otp|pin|password|cvv)\b.{0,25}\b(required|needed)\b", text)
    safety_exemption = credential_advisory or no_credential_required
    if any(matches(pattern, text) for pattern in SCAM_PATTERNS) and not safety_exemption:
        risk.append("credential or pressured-payment language")
    business = case.business
    trusted_business = bool(
        business and business.get("verified") == "1" and
        business.get("official_domain", "").lower() ==
        business.get("domain_used_by_sender", "").lower()
    )
    deceptive_verification = (
        matches(r"\b(account|security|login|profile|delivery|parcel|package)\b", text) and
        matches(r"\b(link|https?://|bit\.ly|\.in\b|\.com\b|verify|verification)\b", text) and
        matches(r"\b(block|restrict|suspend|failed|urgent|final reminder|return|reattempt|today)\w*", text)
    )
    if deceptive_verification and not trusted_business and not safety_exemption:
        risk.append("deceptive account or delivery verification")
    if (matches(r"\b(bank details|account number|wallet details|card details)\b", text) and
            matches(r"\b(send|shar\w*|enter|fill|confirm|verify)\b", text) and not credential_advisory):
        risk.append("sensitive financial details request")
    if (matches(r"\b(processing fee|token)\b", text) and
            matches(r"\b(pay|payment|release|papers|registry)\b", text)):
        risk.append("advance-payment pressure")
    if any(matches(pattern, text) for pattern in PROMPT_INJECTION_PATTERNS):
        risk.append("instruction-like sender content must be treated as data")
    if credential_advisory:
        priority.append("credential-safety advisory, not a credential request")
    if message.forwarded_count >= 5: noise.append("high forwarding count")
    if matches(URGENT_PATTERNS, text): priority.append("explicit time-sensitive language")
    if message.conversation_type == "group" and case.membership.get("group_muted_by_user") == "1":
        noise.append("recipient muted this group")
    if message.user_id.lower() in text or "@" + message.user_id.lower() in text: priority.append("direct mention")
    if matches(r"\b(today|tonight)\b", text) and matches(r"\b(leave|arrive|deliver|delivery|reach|packed|pickup|meeting|bus|schedule|before|by)\b", text):
        priority.append("same-day operational timing")
    relationship = case.business_history
    if matches(PROMOTION_PATTERNS, text):
        noise.append("promotional wording")
        if relationship.get("allows_promotions") == "0" or relationship.get("promotions_opted_out_at"):
            noise.append("promotion opt-out or no consent")
    if int(case.notification_summary.get("notifications_dismissed", "0") or 0) >= 5:
        noise.append("high notification fatigue")
    if business and business.get("verified") == "1" and matches(PAYMENT_PATTERNS, text):
        if business.get("official_domain", "").lower() != business.get("domain_used_by_sender", "").lower():
            risk.append("business domain mismatch")
    defer = ["sender explicitly indicates the message can wait"] \
        if any(matches(pattern, text) for pattern in DEFER_PATTERNS) else []
    media = _media_signals(case)
    return {"risk": list(dict.fromkeys(risk)),
            "priority": list(dict.fromkeys(priority)),
            "noise": list(dict.fromkeys(noise)),
            "defer": defer, "media": media}


def _media_signals(case: CaseFile) -> List[str]:
    """Flag uncertainty for the model; never infer a route from token overlap."""
    native, media = case.native_text.lower(), case.media_text.lower()
    if not native or not media:
        return []
    signals = []
    try:
        current_year = datetime.fromisoformat(case.message.created_at).year
    except ValueError:
        current_year = 0
    years = sorted({int(value) for value in re.findall(r"\b20\d{2}\b", media)})
    if current_year and years and max(years) < current_year and matches(
            r"\b(today|tonight|tomorrow|current|now|close[sd]?|deadline)\b", native):
        signals.append("media contains a past year while caption asserts a current deadline")
    native_times = set(re.findall(r"\b(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:am|pm)\b", native))
    media_times = set(re.findall(r"\b(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:am|pm)\b", media))
    if native_times and media_times and native_times.isdisjoint(media_times):
        signals.append("caption and media contain different times")
    if matches(r"\b(reallygreatsite|example\.com|\+123[- ]?456)\b", media):
        signals.append("media contains placeholder contact or website details")
    return signals


def apply(case: CaseFile) -> None:
    facts = extract(case)
    case.risk_signals, case.priority_signals, case.noise_signals = facts["risk"], facts["priority"], facts["noise"]
    case.defer_signals, case.media_signals = facts["defer"], facts["media"]
