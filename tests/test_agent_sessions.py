from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from services.agent_api.app import main
from services.agent_api.app.sessions import SessionManager, derive_session_title
from shared.schemas import ChatDebugInfo, ChatResponse


def test_derive_session_title_normalizes_whitespace_and_trims() -> None:
    title = derive_session_title("  任隊長   第一次   被提到是在哪裡？  ")
    assert title == "任隊長 第一次 被提到是在哪裡？"

    long_title = derive_session_title("a" * 40)
    assert long_title == "a" * 24


def test_session_manager_create_get_delete() -> None:
    manager = SessionManager()
    created = manager.create_session()

    detail = manager.get_session_detail(created.session_id)
    assert detail.session_id == created.session_id
    assert detail.messages == []

    manager.delete_session(created.session_id)
    with pytest.raises(Exception) as exc_info:
        manager.get_session_detail(created.session_id)

    assert "Session not found" in str(exc_info.value)


def test_session_manager_expires_idle_sessions() -> None:
    manager = SessionManager(idle_ttl_minutes=45)
    created = manager.create_session()
    session = manager._sessions[created.session_id]
    session.updated_at = datetime.now(UTC) - timedelta(minutes=46)

    assert manager.list_sessions() == []
    with pytest.raises(Exception) as exc_info:
        manager.get_session_detail(created.session_id)

    assert "Session not found" in str(exc_info.value)


def test_session_manager_evicts_least_recently_updated_session() -> None:
    manager = SessionManager(max_sessions=2)
    first = manager.create_session()
    second = manager.create_session()
    manager.append_user_message(first.session_id, "first")
    third = manager.create_session()

    listed_ids = [session.session_id for session in manager.list_sessions()]
    assert len(listed_ids) == 2
    assert first.session_id in listed_ids
    assert third.session_id in listed_ids
    assert second.session_id not in listed_ids


def test_session_manager_truncates_messages_and_history_window() -> None:
    manager = SessionManager(max_messages=4, history_window=3)
    created = manager.create_session()

    for index in range(6):
        manager.append_user_message(created.session_id, f"user-{index}")
        manager.append_assistant_message(
            created.session_id,
            f"assistant-{index}",
            ChatDebugInfo(provider="ollama", model="demo"),
        )

    detail = manager.get_session_detail(created.session_id)
    assert [message.content for message in detail.messages] == [
        "user-4",
        "assistant-4",
        "user-5",
        "assistant-5",
    ]
    assert [message.content for message in manager.get_recent_history(created.session_id)] == [
        "assistant-4",
        "user-5",
        "assistant-5",
    ]


def test_session_manager_sets_title_from_first_message() -> None:
    manager = SessionManager()
    created = manager.create_session()
    manager.append_user_message(created.session_id, "  任隊長   第一次   被提到是在哪裡？  ")

    detail = manager.get_session_detail(created.session_id)
    assert detail.title == "任隊長 第一次 被提到是在哪裡？"


def test_post_chat_route_remains_unchanged(monkeypatch) -> None:
    expected = ChatResponse(
        answer="ok",
        citations=[],
        debug=ChatDebugInfo(provider="ollama", model="demo"),
    )
    monkeypatch.setattr(main, "run_chat", lambda request: expected)
    monkeypatch.setattr(main, "start_agent_warmup", lambda: False)

    with TestClient(main.app) as client:
        response = client.post("/chat", json={"message": "hello", "history": []})

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")


def test_session_endpoints_and_history_usage(monkeypatch) -> None:
    manager = SessionManager()
    captured_histories: list[list[str]] = []
    captured_messages: list[str] = []

    def fake_run_chat_turn(*, message, history, include_timing):
        captured_messages.append(message)
        captured_histories.append([f"{item.role}:{item.content}" for item in history])
        return ChatResponse(
            answer=f"answer:{message}",
            citations=[],
            debug=ChatDebugInfo(provider="ollama", model="demo"),
        )

    monkeypatch.setattr(main, "get_session_manager", lambda: manager)
    monkeypatch.setattr(main, "run_chat_turn", fake_run_chat_turn)
    monkeypatch.setattr(main, "start_agent_warmup", lambda: False)

    with TestClient(main.app) as client:
        created = client.post("/sessions")
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        listed = client.get("/sessions")
        assert listed.status_code == 200
        assert listed.json()["sessions"][0]["session_id"] == session_id

        first = client.post(f"/sessions/{session_id}/chat", json={"message": "first turn", "include_timing": True})
        assert first.status_code == 200
        assert first.json()["answer"] == "answer:first turn"
        assert first.json()["session_id"] == session_id
        assert first.json()["turn_index"] == 1

        second = client.post(f"/sessions/{session_id}/chat", json={"message": "follow up"})
        assert second.status_code == 200
        assert second.json()["answer"] == "answer:follow up"
        assert second.json()["turn_index"] == 2

        detail = client.get(f"/sessions/{session_id}")
        assert detail.status_code == 200
        assert [message["content"] for message in detail.json()["messages"]] == [
            "first turn",
            "answer:first turn",
            "follow up",
            "answer:follow up",
        ]
        assert detail.json()["last_debug"]["provider"] == "ollama"

        deleted = client.delete(f"/sessions/{session_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"status": "deleted", "session_id": session_id}

    assert captured_messages == ["first turn", "follow up"]
    assert captured_histories == [[], ["user:first turn", "assistant:answer:first turn"]]


def test_session_endpoints_return_404_for_missing_session(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_session_manager", lambda: SessionManager())
    monkeypatch.setattr(main, "start_agent_warmup", lambda: False)

    with TestClient(main.app) as client:
        response = client.post("/sessions/missing/chat", json={"message": "hello"})

    assert response.status_code == 404


def test_root_and_static_assets_are_served(monkeypatch) -> None:
    monkeypatch.setattr(main, "start_agent_warmup", lambda: False)

    with TestClient(main.app) as client:
        root = client.get("/")
        asset = client.get("/static/app.js")

    assert root.status_code == 200
    assert "<!doctype html>" in root.text.lower()
    assert asset.status_code == 200
    assert "ensureSession" in asset.text
