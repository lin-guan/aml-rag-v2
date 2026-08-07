from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import numpy as np

from app.embedder import Embedder
from app.schemas import AddRequest, Message, SearchRequest, SearchResult
from app.store import MemoryRecord, SQLiteMemoryStore

_CODE_TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)"
    r"|(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"
    r"|(?:[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z0-9_.-]+)+)"
    r"|(?:[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+)"
    r"|(?:[a-z]+[A-Z][A-Za-z0-9]*)"
    r"|(?:[A-Z][A-Z0-9_]{2,})"
    r"|(?:[A-Za-z0-9_.-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|cpp|c|h|cs|rb|php|sql|yaml|yml|json|toml|md))"
)
_CODE_MARKERS = (
    "```",
    "traceback",
    "exception",
    "error:",
    "stack trace",
    "npm ",
    "pip ",
    "pytest",
    "function ",
    "class ",
    "import ",
    "select ",
)


def code_tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _CODE_TOKEN_PATTERN.finditer(text)}


def is_code_query(text: str) -> bool:
    lowered = text.casefold()
    return bool(code_tokens(text)) or any(marker in lowered for marker in _CODE_MARKERS)


_RELATIVE_TIME_PATTERNS = (
    (re.compile(r"\b(day before yesterday)\b", re.IGNORECASE), -2, "day"),
    (re.compile(r"\b(yesterday)\b", re.IGNORECASE), -1, "day"),
    (re.compile(r"\b(today)\b", re.IGNORECASE), 0, "day"),
    (re.compile(r"\b(tomorrow)\b", re.IGNORECASE), 1, "day"),
    (
        re.compile(
            r"\b(last|previous) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.IGNORECASE,
        ),
        -1,
        "weekday",
    ),
    (
        re.compile(
            r"\b(this) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
        ),
        0,
        "weekday",
    ),
    (
        re.compile(
            r"\b(next) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
        ),
        1,
        "weekday",
    ),
    (re.compile(r"\b(last|previous) month\b", re.IGNORECASE), -1, "month"),
    (re.compile(r"\b(this month)\b", re.IGNORECASE), 0, "month"),
    (re.compile(r"\b(next month)\b", re.IGNORECASE), 1, "month"),
    (re.compile(r"\b(last|previous) year\b", re.IGNORECASE), -1, "year"),
    (re.compile(r"\b(this year)\b", re.IGNORECASE), 0, "year"),
    (re.compile(r"\b(next year)\b", re.IGNORECASE), 1, "year"),
)


def _shift_month(value: datetime, offset: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + offset
    return value.replace(year=month_index // 12, month=month_index % 12 + 1, day=1)


def _resolve_weekday(anchor: datetime, weekday_name: str, direction: int) -> datetime:
    target = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ).index(weekday_name.casefold())
    if direction < 0:
        days = (anchor.weekday() - target) % 7 or 7
        return anchor - timedelta(days=days)
    if direction > 0:
        days = (target - anchor.weekday()) % 7 or 7
        return anchor + timedelta(days=days)
    return anchor + timedelta(days=target - anchor.weekday())


def temporal_annotations(content: str, timestamp_ms: int | None) -> list[str]:
    if timestamp_ms is None:
        return []
    anchor = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    annotations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for pattern, offset, granularity in _RELATIVE_TIME_PATTERNS:
        for match in pattern.finditer(content):
            if granularity == "day":
                resolved = (anchor + timedelta(days=offset)).strftime("%d %B %Y").lstrip("0")
            elif granularity == "weekday":
                resolved = _resolve_weekday(anchor, match.group(2), offset)
                resolved = resolved.strftime("%A, %d %B %Y").replace(" 0", " ")
            elif granularity == "month":
                resolved = _shift_month(anchor, offset).strftime("%B %Y")
            else:
                resolved = str(anchor.year + offset)
            key = (match.group(0).lower(), resolved)
            if key not in seen:
                annotations.append(
                    f'Resolved time: "{match.group(0)}" = {resolved} ({granularity} granularity)'
                )
                seen.add(key)
    return annotations


def render_memory(message: Message) -> str:
    prefix_parts = []
    if message.timestamp is not None:
        timestamp = (
            datetime.fromtimestamp(message.timestamp / 1000, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
        prefix_parts.append(timestamp)
    prefix_parts.append(message.role)
    rendered = f"[{' | '.join(prefix_parts)}] {message.content}"
    annotations = temporal_annotations(message.content, message.timestamp)
    annotation_text = "\n".join(annotations)
    return f"{rendered}\n{annotation_text}" if annotations else rendered


class MemoryService:
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedder: Embedder,
        include_options_in_query: bool,
        max_top_k: int,
        enable_hybrid_retrieval: bool = True,
        lexical_candidate_k: int = 100,
        dense_weight: float = 1.0,
        lexical_weight: float = 0.9,
        neighborhood_radius: int = 1,
        context_embedding_radius: int = 2,
        context_embedding_weight: float = 0.3,
        neighbor_result_ratio: float = 0.2,
        index_window_enabled: bool = True,
        index_window_size: int = 6,
        index_window_overlap: int = 2,
        window_retrieval_weight: float = 0.7,
        code_retrieval_enabled: bool = True,
        code_exact_match_weight: float = 0.08,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._include_options_in_query = include_options_in_query
        self._max_top_k = max_top_k
        self._enable_hybrid_retrieval = enable_hybrid_retrieval
        self._lexical_candidate_k = lexical_candidate_k
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight
        self._neighborhood_radius = neighborhood_radius
        self._context_embedding_radius = context_embedding_radius
        self._context_embedding_weight = context_embedding_weight
        self._neighbor_result_ratio = neighbor_result_ratio
        self._index_window_enabled = index_window_enabled
        self._index_window_size = index_window_size
        self._index_window_overlap = index_window_overlap
        self._window_retrieval_weight = window_retrieval_weight
        self._code_retrieval_enabled = code_retrieval_enabled
        self._code_exact_match_weight = code_exact_match_weight

    def _embedding_texts(self, memory_texts: list[str]) -> list[str]:
        if self._context_embedding_radius == 0:
            return memory_texts
        texts: list[str] = []
        for index, target in enumerate(memory_texts):
            start = max(0, index - self._context_embedding_radius)
            end = min(len(memory_texts), index + self._context_embedding_radius + 1)
            context = "\n".join(memory_texts[start:index] + memory_texts[index + 1 : end])
            texts.append(
                f"Target memory:\n{target}\nNearby conversation context:\n{context}"
                if context
                else target
            )
        return texts

    async def add(self, request: AddRequest) -> None:
        messages = [message.model_dump() for message in request.messages]
        memory_texts = [render_memory(message) for message in request.messages]
        target_vectors = await self._embedder.encode(memory_texts)
        if self._context_embedding_weight > 0 and self._context_embedding_radius > 0:
            context_vectors = await self._embedder.encode(self._embedding_texts(memory_texts))
            vectors = (
                1 - self._context_embedding_weight
            ) * target_vectors + self._context_embedding_weight * context_vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        else:
            vectors = target_vectors
        for message, memory_text in zip(messages, memory_texts, strict=True):
            message["content"] = memory_text
        await self._store.add_async(
            request.request_id,
            request.user_id,
            request.session_id,
            messages,
            vectors,
        )
        if not self._index_window_enabled:
            return
        memory_ids = [
            self._store.memory_id(request.request_id, index) for index in range(len(memory_texts))
        ]
        step = max(1, self._index_window_size - self._index_window_overlap)
        starts = [
            start
            for start in range(0, len(memory_texts), step)
            if start == 0
            or len(memory_texts[start : start + self._index_window_size])
            > self._index_window_overlap
        ]
        window_groups = [memory_ids[start : start + self._index_window_size] for start in starts]
        window_texts = [
            "Conversation window:\n"
            + "\n".join(memory_texts[start : start + self._index_window_size])
            for start in starts
        ]
        window_vectors = np.vstack(
            [np.mean(vectors[start : start + self._index_window_size], axis=0) for start in starts]
        )
        window_norms = np.linalg.norm(window_vectors, axis=1, keepdims=True)
        window_vectors = window_vectors / np.maximum(window_norms, 1e-12)
        windows = [
            (
                f"window_{request.request_id}_{start}",
                window_records,
                window_text,
                window_vector,
            )
            for start, (window_records, window_text, window_vector) in enumerate(
                zip(window_groups, window_texts, window_vectors, strict=True)
            )
        ]
        await self._store.replace_windows_async(request.user_id, request.session_id, windows)

    def _search_text(self, request: SearchRequest) -> str:
        if not self._include_options_in_query or not request.options:
            return request.query
        options = "\n".join(request.options)
        return f"{request.query}\n{options}"

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        records = await self._store.get_by_user_async(request.user_id)
        if not records:
            return []

        search_text = self._search_text(request)
        query_vector = (await self._embedder.encode([search_text]))[0]
        matrix = np.vstack([record.embedding for record in records])
        message_scores = matrix @ query_vector
        window_records = (
            await self._store.get_windows_by_user_async(request.user_id)
            if self._index_window_enabled
            else []
        )
        window_rank_by_memory: dict[str, int] = {}
        if window_records:
            window_matrix = np.vstack([record.embedding for record in window_records])
            window_scores = window_matrix @ query_vector
            window_order = np.argsort(window_scores)[::-1]
            for rank, index in enumerate(window_order, start=1):
                for memory_id in window_records[index].memory_ids:
                    window_rank_by_memory[memory_id] = min(
                        window_rank_by_memory.get(memory_id, rank),
                        rank,
                    )
        scores = message_scores
        limit = min(request.top_k, self._max_top_k, len(records))
        lexical_text = re.sub(r"[^\w]+", " ", search_text, flags=re.UNICODE).strip()
        query_terms = {term.casefold() for term in lexical_text.split() if len(term) > 1}
        code_query = self._code_retrieval_enabled and is_code_query(search_text)
        query_code_tokens = code_tokens(search_text) if code_query else set()
        question_lower = request.query.casefold()
        temporal_query = any(
            token in question_lower
            for token in (
                "when",
                "date",
                "year",
                "month",
                "day",
                "week",
                "how long",
                "before",
                "after",
                "first",
                "last",
                "earlier",
                "later",
                "order",
                "sequence",
                "chronological",
            )
        )

        dense_order = np.argsort(scores)[::-1]
        dense_rank = {
            records[index].memory_id: rank for rank, index in enumerate(dense_order, start=1)
        }
        lexical_ids = (
            self._store.search_lexical(request.user_id, lexical_text, self._lexical_candidate_k)
            if self._enable_hybrid_retrieval
            else []
        )
        lexical_rank = {memory_id: rank for rank, memory_id in enumerate(lexical_ids, start=1)}
        rrf_k = 60.0
        fused = {
            record.memory_id: self._dense_weight / (rrf_k + dense_rank[record.memory_id])
            + self._lexical_weight
            / (rrf_k + lexical_rank.get(record.memory_id, rrf_k + self._lexical_candidate_k))
            + (
                self._window_retrieval_weight / (rrf_k + window_rank_by_memory[record.memory_id])
                if record.memory_id in window_rank_by_memory
                else 0.0
            )
            for record in records
            if (
                record.memory_id in lexical_rank
                or record.memory_id in window_rank_by_memory
                or dense_rank[record.memory_id] <= self._max_top_k
            )
        }
        by_id = {record.memory_id: record for record in records}
        term_bonus = {
            record.memory_id: min(
                0.15,
                0.02 * sum(term in record.content.casefold() for term in query_terms),
            )
            for record in records
        }
        temporal_bonus = {
            record.memory_id: (0.08 if temporal_query and record.timestamp_ms is not None else 0.0)
            for record in records
        }
        code_bonus = {
            record.memory_id: min(
                self._code_exact_match_weight * 3,
                self._code_exact_match_weight
                * len(query_code_tokens & code_tokens(record.content)),
            )
            for record in records
        }
        record_index = {record.memory_id: index for index, record in enumerate(records)}
        ordered_ids = sorted(
            fused,
            key=lambda memory_id: (
                fused[memory_id]
                + term_bonus[memory_id]
                + temporal_bonus[memory_id]
                + code_bonus[memory_id],
                scores[record_index[memory_id]],
            ),
            reverse=True,
        )[:limit]
        score_by_id = {
            record.memory_id: float(fused.get(record.memory_id, scores[index]))
            for index, record in enumerate(records)
        }

        selected: list[MemoryRecord] = []
        selected_ids: set[str] = set()
        anchor_limit = max(1, limit - round(limit * self._neighbor_result_ratio))
        for memory_id in ordered_ids[:anchor_limit]:
            record = by_id[memory_id]
            selected.append(record)
            selected_ids.add(memory_id)

        if self._neighborhood_radius > 0:
            neighbor_candidates: list[MemoryRecord] = []
            for memory_id in ordered_ids[:anchor_limit]:
                record = by_id[memory_id]
                neighbor_candidates.extend(
                    self._store.get_neighbors(
                        request.user_id,
                        record.session_id,
                        record.row_index,
                        self._neighborhood_radius,
                    )
                )
            for neighbor in neighbor_candidates:
                if neighbor.memory_id not in selected_ids:
                    selected.append(neighbor)
                    selected_ids.add(neighbor.memory_id)
                if len(selected) >= limit:
                    break

        return [
            SearchResult(
                id=record.memory_id,
                content=record.content,
                score=score_by_id[record.memory_id],
                created_at=record.created_at,
            )
            for record in selected
        ]
