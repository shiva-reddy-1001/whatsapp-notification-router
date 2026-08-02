"""Small, deterministic history retrieval with evidence provenance."""
from difflib import SequenceMatcher
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
        event_outcome = _outcome(item.event)
        outcome = 1.0 if event_outcome in {"reported", "muted_after", "dismissed", "replied"} else 0.0
        if embeddings and embeddings.provider != "none":
            score = 0.45 * semantic + 0.25 * lexical + 0.22 * relation + 0.08 * outcome
        else:
            score = 0.65 * lexical + 0.27 * relation + 0.08 * outcome
        # Do not manufacture evidence: a loosely related message from the same
        # user is not useful support for a routing decision.
        if score < 0.50 or (not relation and max(lexical, semantic) < 0.55):
            continue
        rationale = "same conversation context" if relation else "similar prior content"
        ranked.append(RetrievedEvidence(historic.message_id, historic.message_text[:240],
                                        min(score, 1.0), event_outcome, rationale))
    return sorted(ranked, key=lambda row: (-row.score, row.message_id))[:limit]
