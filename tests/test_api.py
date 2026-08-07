from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import Message
from app.service import render_memory


class FakeEmbedder:
    ready = True

    @staticmethod
    async def encode(texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.lower().encode()).digest()
            vector = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


def make_client(tmp_path: Path, auth_mode: str = "bearer") -> TestClient:
    settings = Settings(
        auth_mode=auth_mode,
        api_key=None if auth_mode == "none" else "test-secret",
        database_path=tmp_path / "memory.db",
    )
    app = create_app(settings)
    app.state.embedder = FakeEmbedder()
    app.state.service._embedder = app.state.embedder
    return TestClient(app)


def add_payload() -> dict[str, object]:
    return {
        "request_id": "eval:run:test:chunk-0",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "Alice adopted a cat named Luna.",
            },
            {
                "role": "assistant",
                "content": "Luna likes sleeping by the window.",
            },
        ],
        "user_id": "eval:run:test:user-0",
        "session_id": "eval:run:test:session-0",
    }


def test_health_requires_no_authentication(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_and_search_contract(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer test-secret"}
    payload = add_payload()

    with make_client(tmp_path) as client:
        add_response = client.post("/add", json=payload, headers=headers)
        search_response = client.post(
            "/search",
            json={
                "query": "What is Alice's cat called?",
                "options": ["A. Luna", "B. Milo"],
                "user_id": payload["user_id"],
                "top_k": 100,
            },
            headers=headers,
        )

    assert add_response.status_code == 200
    assert add_response.json() == {
        "success": True,
        "request_id": payload["request_id"],
        "user_id": payload["user_id"],
        "session_id": payload["session_id"],
    }
    assert search_response.status_code == 200
    results = search_response.json()["data"]
    assert len(results) == 2
    assert all(set(item) == {"id", "content", "score", "created_at"} for item in results)


def test_add_is_idempotent(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer test-secret"}
    payload = add_payload()

    with make_client(tmp_path) as client:
        assert client.post("/add", json=payload, headers=headers).status_code == 200
        assert client.post("/add", json=payload, headers=headers).status_code == 200
        response = client.post(
            "/search",
            json={"query": "cat", "user_id": payload["user_id"], "top_k": 100},
            headers=headers,
        )

    assert len(response.json()["data"]) == 2


def test_request_id_payload_conflict(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer test-secret"}
    payload = add_payload()

    with make_client(tmp_path) as client:
        assert client.post("/add", json=payload, headers=headers).status_code == 200
        payload["messages"][0]["content"] = "A different memory."
        conflict = client.post("/add", json=payload, headers=headers)

    assert conflict.status_code == 409


def test_user_id_isolation_and_authentication(tmp_path: Path) -> None:
    payload = add_payload()

    with make_client(tmp_path) as client:
        assert client.post("/add", json=payload).status_code == 401
        assert (
            client.post(
                "/add",
                json=payload,
                headers={"Authorization": "Bearer test-secret"},
            ).status_code
            == 200
        )
        response = client.post(
            "/search",
            json={"query": "cat", "user_id": "another-user", "top_k": 100},
            headers={"Authorization": "Bearer test-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_none_auth_mode_for_smoke(tmp_path: Path) -> None:
    payload = add_payload()
    with make_client(tmp_path, auth_mode="none") as client:
        response = client.post("/add", json=payload)
    assert response.status_code == 200


def test_temporal_annotations_stay_inside_rendered_memory() -> None:
    rendered = render_memory(
        Message(
            role="user",
            timestamp=1683763200000,
            content="I went hiking yesterday and visited Luna last Saturday.",
        )
    )

    assert "[2023-05-11T00:00:00Z | user]" in rendered
    assert 'Resolved time: "yesterday" = 10 May 2023' in rendered
    assert 'Resolved time: "last Saturday" = Saturday, 6 May 2023' in rendered


def test_window_index_maps_context_back_to_original_messages(tmp_path: Path) -> None:
    payload = add_payload()
    payload["messages"].extend(
        [
            {"role": "user", "content": "Alice moved to Stockholm."},
            {"role": "assistant", "content": "She enjoys the long summer days there."},
            {"role": "user", "content": "Her favorite park is Hagaparken."},
            {"role": "assistant", "content": "Luna likes that park too."},
        ]
    )
    headers = {"Authorization": "Bearer test-secret"}
    with make_client(tmp_path) as client:
        assert client.post("/add", json=payload, headers=headers).status_code == 200
        windows = client.app.state.service._store.get_windows_by_user(payload["user_id"])
        response = client.post(
            "/search",
            json={
                "query": "Where does Luna enjoy going?",
                "user_id": payload["user_id"],
                "top_k": 100,
            },
            headers=headers,
        )

    assert len(windows) == 1
    assert all(window.memory_ids for window in windows)
    results = response.json()["data"]
    assert len(results) == len(payload["messages"])
    assert all(not result["id"].startswith("window_") for result in results)


def test_internal_windows_preserve_public_message_results(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer test-secret"}
    payload = add_payload()
    payload["messages"].extend(
        [
            {"role": "user", "content": "Alice bought Luna a blue collar."},
            {"role": "assistant", "content": "The collar has a silver bell."},
            {"role": "user", "content": "Luna sleeps beside Alice's desk."},
            {"role": "assistant", "content": "Alice keeps cat treats in the drawer."},
        ]
    )

    with make_client(tmp_path) as client:
        assert client.post("/add", json=payload, headers=headers).status_code == 200
        service = client.app.state.service
        windows = service._store.get_windows_by_user(payload["user_id"])
        response = client.post(
            "/search",
            json={
                "query": "What is on Luna's collar?",
                "user_id": payload["user_id"],
                "top_k": 100,
            },
            headers=headers,
        )

    assert len(windows) == 1
    assert len(windows[0].memory_ids) == 6
    results = response.json()["data"]
    assert len(results) == 6
    assert all(result["id"].startswith("mem_") for result in results)
    assert all(not result["content"].startswith("Conversation window:") for result in results)
