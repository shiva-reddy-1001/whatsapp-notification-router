"""Small, deterministic history retrieval with evidence provenance."""
from difflib import SequenceMatcher
from datetime import datetime
import re
from typing import List, Optional

try:
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:
    token_set_ratio = None

from .models import CaseFile, HistoryItem, RetrievedEvidence
from .embeddings import EmbeddingIndex


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if token_set_ratio:
        return token_set_ratio(left, right) / 100.0
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _outcome(event: dict) -> str:
    if event.get("message_reported") == "1": return "reported"
    if event.get("muted_after_message") == "1": return "muted_after"
    if event.get("message_replied") == "1": return "replied"
    if event.get("notification_dismissed") == "1": return "dismissed"
    if event.get("message_opened") == "1": return "opened"
    return "no recorded reaction"


def _risk_profile(text: str) -> bool:
    return bool(re.search(
        r"\b(otp|password|pin|login code|bank details|account number|card details|"
        r"processing fee|verif\w*|account blocked|suspended)\b", text, re.I))


def _recency(current: str, historic: str) -> float:
    try:
        days = max(0, (datetime.fromisoformat(current) - datetime.fromisoformat(historic)).days)
        return max(0.0, 1.0 - days / 365.0)
    except ValueError:
        return 0.0


def retrieve(case: CaseFile, candidates: List[HistoryItem], limit: int,
             embeddings: Optional[EmbeddingIndex] = None) -> List[RetrievedEvidence]:
    ranked = []
    current = case.message
    for item in candidates:
        historic = item.message
        if historic.created_at >= current.created_at:
            continue
        same_sender = bool(historic.sender_user_id and historic.sender_user_id == current.sender_user_id)
        same_group = bool(historic.group_id and historic.group_id == current.group_id)
        same_business = bool(historic.business_id and historic.business_id == current.business_id)
        relation = 1.0 if same_sender or same_group or same_business else 0.0
        lexical = _similarity(case.content, historic.message_text)
        semantic = embeddings.similarity(case.content, historic.message_text) if embeddings else 0.0
        similarity = max(lexical, semantic)
        current_risk = _risk_profile(case.content)
        historic_risk = _risk_profile(historic.message_text)
        # Relationship is a useful candidate generator, not sufficient evidence
        # by itself. This prevents an urgent link from citing an unrelated sale
        # merely because both appeared in the same group or business thread.
        if relation and similarity < 0.30:
            continue
        if not relation and similarity < 0.55:
            continue
        if current_risk != historic_risk and similarity < 0.55:
            continue
        event_outcome = _outcome(item.event)
        outcome = 1.0 if event_outcome in {"reported", "muted_after", "dismissed", "replied"} else 0.0
        recency = _recency(current.created_at, historic.created_at)
        risk_match = current_risk == historic_risk
        if embeddings and embeddings.provider != "none":
            score = (0.40 * semantic + 0.24 * lexical + 0.18 * relation +
                     0.08 * outcome + 0.05 * recency + 0.05 * risk_match)
        else:
            score = (0.58 * lexical + 0.24 * relation + 0.08 * outcome +
                     0.05 * recency + 0.05 * risk_match)
        # Do not manufacture evidence: a loosely related message from the same
        # user is not useful support for a routing decision.
        if score < 0.50:
            continue
        rationale = ("same conversation context" if relation else "similar prior content")
        if event_outcome != "no recorded reaction":
            rationale += "; prior outcome=%s" % event_outcome
        ranked.append(RetrievedEvidence(historic.message_id, historic.message_text[:240],
                                        min(score, 1.0), event_outcome, rationale))
    return sorted(ranked, key=lambda row: (-row.score, row.message_id))[:limit]
