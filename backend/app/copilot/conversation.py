"""Short conversation context (9J).

Deliberately NOT a long-term agent memory: an in-memory store holding the
last N turns plus tool-call summaries per conversation, with a maximum
number of conversations and TTL. Enough to resolve "那 Station 03 呢？"
against the previous Line A context, nothing more.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_TURNS = 10          # messages kept per conversation
MAX_CONVERSATIONS = 200
TTL_SECONDS = 3600 * 6  # 6h


@dataclass
class Turn:
    role: str                 # user | assistant
    content: str
    tool_summary: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: list[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [
                {"role": t.role, "content": t.content, "tools": t.tool_summary}
                for t in self.turns
            ],
        }


class ConversationStore:
    def __init__(self) -> None:
        self._store: dict[str, Conversation] = {}

    def _evict(self) -> None:
        now = time.time()
        stale = [k for k, v in self._store.items() if now - v.updated_at > TTL_SECONDS]
        for k in stale:
            del self._store[k]
        if len(self._store) > MAX_CONVERSATIONS:
            for k in sorted(self._store, key=lambda k: self._store[k].updated_at)[: len(self._store) - MAX_CONVERSATIONS]:
                del self._store[k]

    def get_or_create(self, conversation_id: str | None) -> Conversation:
        self._evict()
        cid = conversation_id or uuid.uuid4().hex
        conv = self._store.get(cid)
        if conv is None:
            conv = Conversation(id=cid)
            self._store[cid] = conv
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        return self._store.get(conversation_id)

    def reset(self) -> None:
        self._store.clear()


conversation_store = ConversationStore()
