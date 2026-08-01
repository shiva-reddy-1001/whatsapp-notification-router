"""Typed records passed between router stages."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}


@dataclass
class Message:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: str
    message_text: str
    media_type: Optional[str]
    media_id: Optional[str]
    forwarded_count: int


@dataclass
class HistoryItem:
    message: Message
    event: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievedEvidence:
    message_id: str
    text: str
    score: float
    outcome: str
    rationale: str


@dataclass
class CaseFile:
    message: Message
    content: str
    media_quality: float
    user: Dict[str, str]
    group: Dict[str, str]
    membership: Dict[str, str]
    business: Dict[str, str]
    business_history: Dict[str, str]
    notification_summary: Dict[str, str]
    risk_signals: List[str] = field(default_factory=list)
    priority_signals: List[str] = field(default_factory=list)
    noise_signals: List[str] = field(default_factory=list)
    evidence: List[RetrievedEvidence] = field(default_factory=list)


@dataclass
class Prediction:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: List[str] = field(default_factory=list)

