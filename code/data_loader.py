"""CSV loading, lookups, and contextual case-file construction."""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import CaseFile, HistoryItem, Message


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _none(value: Optional[str]) -> Optional[str]:
    return value.strip() if value and value.strip() else None


def as_message(row: Dict[str, str]) -> Message:
    return Message(
        message_id=row["message_id"], user_id=row["user_id"],
        conversation_type=row["conversation_type"], group_id=_none(row.get("group_id")),
        business_id=_none(row.get("business_id")), sender_user_id=_none(row.get("sender_user_id")),
        created_at=row.get("created_at", ""), message_text=row.get("message_text", "").strip(),
        media_type=_none(row.get("media_type")), media_id=_none(row.get("media_id")),
        forwarded_count=int(row.get("forwarded_count") or 0),
    )


class Dataset:
    def __init__(self, root: Path, messages_path: Optional[Path] = None):
        self.root = root
        self.messages_path = messages_path or root / "messages.csv"
        self.messages = [as_message(row) for row in read_csv(self.messages_path)]
        self.users = self._by_key("users.csv", "user_id")
        self.groups = self._by_key("groups.csv", "group_id")
        self.memberships = self._by_key("group_members.csv", ("group_id", "user_id"))
        self.businesses = self._by_key("business_accounts.csv", "business_id")
        self.business_history = self._by_key("user_business_history.csv", ("user_id", "business_id"))
        self.images = self._by_key("images.csv", "image_id")
        self.voice_notes = self._by_key("voice_notes.csv", "voice_note_id")
        self.daily = defaultdict(list)
        for row in read_csv(root / "daily_notification_summary.csv"):
            self.daily[row["user_id"]].append(row)
        self.events = self._by_key("message_events.csv", "message_id")
        self.history = self._history()
        self.history_by_user = defaultdict(list)
        for item in self.history:
            self.history_by_user[item.message.user_id].append(item)

    def _by_key(self, filename: str, key):
        result = {}
        for row in read_csv(self.root / filename):
            if isinstance(key, tuple):
                result[tuple(row[column] for column in key)] = row
            else:
                result[row[key]] = row
        return result

    def _history(self) -> List[HistoryItem]:
        return [HistoryItem(as_message(row), self.events.get(row["message_id"], {}))
                for row in read_csv(self.root / "message_history.csv")]

    def media_path(self, message: Message) -> Optional[Path]:
        if not message.media_id:
            return None
        table = self.images if message.media_type == "image" else self.voice_notes
        key = "image_id" if message.media_type == "image" else "voice_note_id"
        row = table.get(message.media_id)
        return self.root / row["file_path"] if row and row.get(key) else None

    def case_file(self, message: Message, content: str, media_quality: float,
                  media_text: str = "") -> CaseFile:
        daily = self.daily.get(message.user_id, [])
        latest = max(daily, key=lambda row: row.get("date", ""), default={})
        return CaseFile(
            message=message, content=content,
            media_quality=media_quality,
            user=self.users.get(message.user_id, {}),
            group=self.groups.get(message.group_id or "", {}),
            membership=self.memberships.get((message.group_id, message.user_id), {}),
            business=self.businesses.get(message.business_id or "", {}),
            business_history=self.business_history.get((message.user_id, message.business_id), {}),
            notification_summary=latest,
            native_text=message.message_text,
            media_text=media_text,
        )
