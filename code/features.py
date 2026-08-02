"""Pure, auditable feature extraction separated from routing policy."""
import re
from datetime import datetime
from typing import Dict, List

from .models import CaseFile

SCAM_PATTERNS = (
    r"\b(otp|one.time password|pin|password|cvv|login code)\b.*\b(share|send|enter|verify|confirm|reply|provide)\b",
    r"\b(reply|send|share|confirm|provide)\b.*\b(otp|one.time password|pin|password|cvv|login code)\b",
    r"\b(pay|payment|fee|refund|reward)\b.*\b(otp|link|verify|urgent)\b",
    r"\b(account|bank|package|delivery)\b.*\b(blocked|suspended|release|verify)\b",
)
URGENT_PATTERNS = r"\b(urgent|emergency|immediately|right now|eod|deadline|last[- ]minute|within \d+|\d+\s*(?:min|mins|minutes|hours?|hrs?))\b"
PROMOTION_PATTERNS = r"\b(sale|offer|discount|coupon|cashback|buy now|limited offer|% off)\b"
PAYMENT_PATTERNS = r"\b(payment|invoice|bill|due|receipt|transaction|refund)\b"


def matches(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL))


def extract(case: CaseFile) -> Dict[str, List[str]]:
    """Return facts; policy decides how much each fact should matter."""
    text, message = case.content.lower(), case.message
    risk, priority, noise = [], [], []
    credential_advisory = matches(
        r"\b(never|do not|don't|will not|won't)\b.{0,60}\b(ask|share|provide|send)\b.{0,40}"
        r"\b(otp|password|pin|cvv|payment details|login code)\b", text)
    if any(matches(pattern, text) for pattern in SCAM_PATTERNS) and not credential_advisory:
        risk.append("credential or pressured-payment language")
    if credential_advisory:
        priority.append("credential-safety advisory, not a credential request")
    if message.forwarded_count >= 5: noise.append("high forwarding count")
    if matches(URGENT_PATTERNS, text): priority.append("explicit time-sensitive language")
    if message.conversation_type == "group" and case.membership.get("group_muted_by_user") == "1":
        noise.append("recipient muted this group")
    if message.user_id.lower() in text or "@" + message.user_id.lower() in text: priority.append("direct mention")
    if matches(r"\b(today|tonight)\b", text) and matches(r"\b(leave|arrive|deliver|pickup|meeting|bus|schedule|before|by)\b", text):
        priority.append("same-day operational timing")
    relationship = case.business_history
    if matches(PROMOTION_PATTERNS, text):
        noise.append("promotional wording")
        if relationship.get("allows_promotions") == "0" or relationship.get("promotions_opted_out_at"):
            noise.append("promotion opt-out or no consent")
    if int(case.notification_summary.get("notifications_dismissed", "0") or 0) >= 5:
        noise.append("high notification fatigue")
    business = case.business
    if business and business.get("verified") == "1" and matches(PAYMENT_PATTERNS, text):
        if business.get("official_domain", "").lower() != business.get("domain_used_by_sender", "").lower():
            risk.append("business domain mismatch")
    return {"risk": risk, "priority": priority, "noise": noise}


def apply(case: CaseFile) -> None:
    facts = extract(case)
    case.risk_signals, case.priority_signals, case.noise_signals = facts["risk"], facts["priority"], facts["noise"]
