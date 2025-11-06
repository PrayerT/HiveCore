# -*- coding: utf-8 -*-
"""Persistent storage helpers for AA user memory."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


def _default_store_path() -> Path:
    base = os.environ.get("AGENTSCOPE_HOME") or os.path.join(Path.home(), ".agentscope")
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(base) / "aa_memory.json"


@dataclass
class AAMemoryRecord:
    """Representation of long-term AA memory for a single user."""

    user_id: str
    prompt: str = ""
    knowledge_base: list[str] = field(default_factory=list)
    conversation_log: list[dict[str, str]] = field(default_factory=list)

    def append_message(self, role: str, content: str) -> None:
        self.conversation_log.append({"role": role, "content": content})

    def add_knowledge(self, entry: str) -> None:
        if entry not in self.knowledge_base:
            self.knowledge_base.append(entry)


class AAMemoryStore:
    """JSON-file backed store for AA memory records."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _default_store_path()
        self._lock = Lock()
        self._cache: dict[str, AAMemoryRecord] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        for user_id, payload in data.items():
            record = AAMemoryRecord(
                user_id=user_id,
                prompt=payload.get("prompt", ""),
                knowledge_base=list(payload.get("knowledge_base", [])),
                conversation_log=list(payload.get("conversation_log", [])),
            )
            self._cache[user_id] = record

    def _flush(self) -> None:
        serialized: dict[str, Any] = {
            user_id: asdict(record) for user_id, record in self._cache.items()
        }
        self.path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, user_id: str) -> AAMemoryRecord:
        with self._lock:
            if user_id not in self._cache:
                self._cache[user_id] = AAMemoryRecord(user_id=user_id)
                self._flush()
            return self._cache[user_id]

    def save(self, record: AAMemoryRecord) -> None:
        with self._lock:
            self._cache[record.user_id] = record
            self._flush()

    def append(self, user_id: str, role: str, content: str) -> None:
        record = self.load(user_id)
        record.append_message(role, content)
        self.save(record)

    def update_prompt(self, user_id: str, prompt: str) -> None:
        record = self.load(user_id)
        record.prompt = prompt
        self.save(record)

    def add_knowledge(self, user_id: str, entry: str) -> None:
        record = self.load(user_id)
        record.add_knowledge(entry)
        self.save(record)
