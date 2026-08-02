"""Small durable cache for media, case features, and provider responses.

The cache is intentionally SQLite rather than external infrastructure: it is
portable, inspectable, resumable, and suitable for this small challenge data.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class SQLiteCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("""CREATE TABLE IF NOT EXISTS cache_entries (
            namespace TEXT NOT NULL, cache_key TEXT NOT NULL, value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL, PRIMARY KEY(namespace, cache_key))""")
        self.connection.commit()

    def get(self, namespace: str, cache_key: str) -> Optional[Any]:
        row = self.connection.execute("SELECT value_json FROM cache_entries WHERE namespace=? AND cache_key=?",
                                      (namespace, cache_key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, namespace: str, cache_key: str, value: Any) -> None:
        self.put_many(namespace, {cache_key: value})

    def put_many(self, namespace: str, entries: dict) -> None:
        """Persist a model batch in one transaction instead of one commit per vector."""
        updated_at = datetime.now(timezone.utc).isoformat()
        self.connection.executemany("""INSERT INTO cache_entries(namespace, cache_key, value_json, updated_at)
            VALUES (?, ?, ?, ?) ON CONFLICT(namespace, cache_key) DO UPDATE SET
            value_json=excluded.value_json, updated_at=excluded.updated_at""",
                                [(namespace, key, json.dumps(value, sort_keys=True), updated_at)
                                 for key, value in entries.items()])
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
