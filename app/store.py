from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    content: str
    embedding: np.ndarray
    created_at: str
    user_id: str
    session_id: str
    timestamp_ms: int | None
    row_index: int


@dataclass(frozen=True, slots=True)
class WindowRecord:
    window_id: str
    user_id: str
    session_id: str
    memory_ids: tuple[str, ...]
    content: str
    embedding: np.ndarray


class SQLiteMemoryStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=30000")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp_ms INTEGER,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES add_requests(request_id)
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user_id
                    ON memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_memories_request_id
                    ON memories(request_id);
                CREATE INDEX IF NOT EXISTS idx_memories_user_session_row
                    ON memories(user_id, session_id, id);

                CREATE TABLE IF NOT EXISTS memory_windows (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    memory_ids TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_windows_user
                    ON memory_windows(user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_windows_user_session
                    ON memory_windows(user_id, session_id);
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        memory_id UNINDEXED,
                        user_id UNINDEXED,
                        content,
                        tokenize='unicode61'
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO memories_fts(memory_id, user_id, content)
                    SELECT m.id, m.user_id, m.content FROM memories AS m
                    WHERE NOT EXISTS (
                        SELECT 1 FROM memories_fts AS f WHERE f.memory_id = m.id
                    )
                    """
                )
            except sqlite3.OperationalError:
                # Some SQLite builds omit FTS5. Dense retrieval remains available.
                pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def payload_hash(
        user_id: str,
        session_id: str,
        messages: list[dict[str, object]],
    ) -> str:
        canonical_messages = [
            {
                "role": message["role"],
                "timestamp": message.get("timestamp"),
                "content": message.get("raw_content", message["content"]),
            }
            for message in messages
        ]
        serialized = json.dumps(
            {"user_id": user_id, "session_id": session_id, "messages": canonical_messages},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def memory_id(request_id: str, message_index: int) -> str:
        digest = hashlib.sha256(f"{request_id}:{message_index}".encode()).hexdigest()
        return f"mem_{digest[:24]}"

    def add(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[dict[str, object]],
        embeddings: np.ndarray,
    ) -> None:
        if len(messages) != len(embeddings):
            raise ValueError("message and embedding counts differ")

        payload_hash = self.payload_hash(user_id, session_id, messages)
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_hash FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload_hash:
                    raise ValueError("request_id was already used with a different payload")
                return

            connection.execute(
                """
                INSERT INTO add_requests
                    (request_id, user_id, session_id, payload_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, user_id, session_id, payload_hash, now),
            )

            rows = []
            for index, (message, embedding) in enumerate(zip(messages, embeddings, strict=True)):
                timestamp = message.get("timestamp")
                created_at = (
                    datetime.fromtimestamp(int(timestamp) / 1000, tz=UTC)
                    .isoformat()
                    .replace("+00:00", "Z")
                    if timestamp is not None
                    else now
                )
                vector = np.asarray(embedding, dtype=np.float32)
                rows.append(
                    (
                        self.memory_id(request_id, index),
                        request_id,
                        user_id,
                        session_id,
                        str(message["role"]),
                        timestamp,
                        str(message["content"]),
                        vector.tobytes(),
                        vector.size,
                        created_at,
                    )
                )

            connection.executemany(
                """
                INSERT INTO memories (
                    id, request_id, user_id, session_id, role, timestamp_ms,
                    content, embedding, embedding_dim, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            with suppress(sqlite3.OperationalError):
                connection.executemany(
                    "INSERT INTO memories_fts(memory_id, user_id, content) VALUES (?, ?, ?)",
                    [(row[0], user_id, row[6]) for row in rows],
                )

    def get_by_user(self, user_id: str) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT rowid, id, content, embedding, embedding_dim, created_at,
                       user_id, session_id, timestamp_ms
                FROM memories
                WHERE user_id = ?
                ORDER BY rowid ASC
                """,
                (user_id,),
            ).fetchall()

        records = []
        for (
            row_index,
            memory_id,
            content,
            embedding_blob,
            dimension,
            created_at,
            record_user_id,
            session_id,
            timestamp_ms,
        ) in rows:
            vector = np.frombuffer(embedding_blob, dtype=np.float32, count=dimension).copy()
            records.append(
                MemoryRecord(
                    memory_id,
                    content,
                    vector,
                    created_at,
                    record_user_id,
                    session_id,
                    timestamp_ms,
                    row_index,
                )
            )
        return records

    def search_lexical(self, user_id: str, query: str, limit: int) -> list[str]:
        terms = [term for term in query.split() if term.strip()]
        if not terms:
            return []
        escaped_terms = [term.replace('"', '""') for term in terms]
        match_query = " OR ".join(f'"{term}"' for term in escaped_terms)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT f.memory_id
                    FROM memories_fts AS f
                    WHERE f.user_id = ? AND f MATCH ?
                    ORDER BY bm25(memories_fts)
                    LIMIT ?
                    """,
                    (user_id, match_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    def replace_windows(
        self,
        user_id: str,
        session_id: str,
        windows: list[tuple[str, list[str], str, np.ndarray]],
    ) -> None:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            window_ids = [window[0] for window in windows]
            if window_ids:
                placeholders = ",".join("?" for _ in window_ids)
                connection.execute(
                    f"DELETE FROM memory_windows WHERE id IN ({placeholders})",
                    window_ids,
                )
            connection.executemany(
                """
                INSERT INTO memory_windows (
                    id, user_id, session_id, memory_ids, content,
                    embedding, embedding_dim, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        window_id,
                        user_id,
                        session_id,
                        json.dumps(memory_ids),
                        content,
                        np.asarray(embedding, dtype=np.float32).tobytes(),
                        int(np.asarray(embedding).size),
                        now,
                    )
                    for window_id, memory_ids, content, embedding in windows
                ],
            )

    def get_windows_by_user(self, user_id: str) -> list[WindowRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, session_id, memory_ids, content,
                       embedding, embedding_dim
                FROM memory_windows
                WHERE user_id = ?
                ORDER BY rowid ASC
                """,
                (user_id,),
            ).fetchall()
        return [
            WindowRecord(
                window_id=str(row[0]),
                user_id=str(row[1]),
                session_id=str(row[2]),
                memory_ids=tuple(json.loads(row[3])),
                content=str(row[4]),
                embedding=np.frombuffer(row[5], dtype=np.float32, count=row[6]).copy(),
            )
            for row in rows
        ]

    def get_neighbors(
        self, user_id: str, session_id: str, row_index: int, radius: int
    ) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT rowid, id, content, embedding, embedding_dim, created_at,
                       user_id, session_id, timestamp_ms
                FROM memories
                WHERE user_id = ? AND session_id = ?
                  AND rowid BETWEEN ? AND ?
                ORDER BY rowid ASC
                """,
                (user_id, session_id, row_index - radius, row_index + radius),
            ).fetchall()
        records = []
        for (
            index,
            memory_id,
            content,
            blob,
            dimension,
            created_at,
            record_user_id,
            session_id,
            timestamp_ms,
        ) in rows:
            records.append(
                MemoryRecord(
                    memory_id,
                    content,
                    np.frombuffer(blob, dtype=np.float32, count=dimension).copy(),
                    created_at,
                    record_user_id,
                    session_id,
                    timestamp_ms,
                    index,
                )
            )
        return records

    async def add_async(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[dict[str, object]],
        embeddings: np.ndarray,
    ) -> None:
        await asyncio.to_thread(
            self.add,
            request_id,
            user_id,
            session_id,
            messages,
            embeddings,
        )

    async def get_by_user_async(self, user_id: str) -> list[MemoryRecord]:
        return await asyncio.to_thread(self.get_by_user, user_id)

    async def replace_windows_async(
        self,
        user_id: str,
        session_id: str,
        windows: list[tuple[str, list[str], str, np.ndarray]],
    ) -> None:
        await asyncio.to_thread(self.replace_windows, user_id, session_id, windows)

    async def get_windows_by_user_async(self, user_id: str) -> list[WindowRecord]:
        return await asyncio.to_thread(self.get_windows_by_user, user_id)
