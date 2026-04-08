from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from uuid import uuid4

from fastapi import HTTPException

from shared.schemas import ChatDebugInfo, SessionCreateResponse, SessionDetailResponse, SessionMessage, SessionSummary

DEFAULT_IDLE_TTL_MINUTES = 45
DEFAULT_MAX_SESSIONS = 100
DEFAULT_MAX_MESSAGES = 20
DEFAULT_HISTORY_WINDOW = 8
DEFAULT_TITLE_LENGTH = 24


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def derive_session_title(message: str, *, max_length: int = DEFAULT_TITLE_LENGTH) -> str:
    normalized = _normalize_whitespace(message).strip()
    if not normalized:
        return "New Session"
    return normalized[:max_length]


@dataclass
class SessionRecord:
    session_id: str
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    messages: list[SessionMessage] = field(default_factory=list)
    last_debug: ChatDebugInfo | None = None

    def summary(self) -> SessionSummary:
        return SessionSummary(
            session_id=self.session_id,
            title=self.title,
            created_at=_isoformat(self.created_at),
            updated_at=_isoformat(self.updated_at),
            message_count=len(self.messages),
        )

    def detail(self) -> SessionDetailResponse:
        return SessionDetailResponse(
            session_id=self.session_id,
            title=self.title,
            created_at=_isoformat(self.created_at),
            updated_at=_isoformat(self.updated_at),
            message_count=len(self.messages),
            messages=list(self.messages),
            last_debug=self.last_debug,
        )

    def create_response(self) -> SessionCreateResponse:
        return SessionCreateResponse(
            session_id=self.session_id,
            title=self.title,
            created_at=_isoformat(self.created_at),
            updated_at=_isoformat(self.updated_at),
            message_count=len(self.messages),
        )


class SessionManager:
    def __init__(
        self,
        *,
        idle_ttl_minutes: int = DEFAULT_IDLE_TTL_MINUTES,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        history_window: int = DEFAULT_HISTORY_WINDOW,
    ) -> None:
        self.idle_ttl = timedelta(minutes=idle_ttl_minutes)
        self.max_sessions = max_sessions
        self.max_messages = max_messages
        self.history_window = history_window
        self._sessions: OrderedDict[str, SessionRecord] = OrderedDict()
        self._lock = RLock()

    def _evict_expired_sessions(self, now: datetime) -> None:
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.updated_at > self.idle_ttl
        ]
        for session_id in expired_ids:
            self._sessions.pop(session_id, None)

    def _evict_over_capacity(self) -> None:
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def _touch(self, session: SessionRecord) -> None:
        self._sessions.move_to_end(session.session_id)

    def _get_active_session(self, session_id: str, *, now: datetime) -> SessionRecord:
        self._evict_expired_sessions(now)
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return session

    def create_session(self) -> SessionCreateResponse:
        now = _utc_now()
        with self._lock:
            self._evict_expired_sessions(now)
            session = SessionRecord(session_id=str(uuid4()), created_at=now, updated_at=now)
            self._sessions[session.session_id] = session
            self._touch(session)
            self._evict_over_capacity()
            return session.create_response()

    def list_sessions(self) -> list[SessionSummary]:
        now = _utc_now()
        with self._lock:
            self._evict_expired_sessions(now)
            ordered_sessions = sorted(self._sessions.values(), key=lambda session: session.updated_at, reverse=True)
            return [session.summary() for session in ordered_sessions]

    def get_session_detail(self, session_id: str) -> SessionDetailResponse:
        now = _utc_now()
        with self._lock:
            session = self._get_active_session(session_id, now=now)
            return session.detail()

    def append_user_message(self, session_id: str, content: str) -> SessionRecord:
        now = _utc_now()
        with self._lock:
            session = self._get_active_session(session_id, now=now)
            if session.title is None:
                session.title = derive_session_title(content)
            session.messages.append(SessionMessage(role="user", content=content, created_at=_isoformat(now)))
            session.messages = session.messages[-self.max_messages :]
            session.updated_at = now
            self._touch(session)
            return session

    def append_assistant_message(self, session_id: str, content: str, debug: ChatDebugInfo) -> SessionRecord:
        now = _utc_now()
        with self._lock:
            session = self._get_active_session(session_id, now=now)
            session.messages.append(SessionMessage(role="assistant", content=content, created_at=_isoformat(now)))
            session.messages = session.messages[-self.max_messages :]
            session.last_debug = debug
            session.updated_at = now
            self._touch(session)
            return session

    def delete_session(self, session_id: str) -> None:
        now = _utc_now()
        with self._lock:
            self._evict_expired_sessions(now)
            if session_id not in self._sessions:
                raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
            self._sessions.pop(session_id, None)

    def get_recent_history(self, session_id: str) -> list[SessionMessage]:
        now = _utc_now()
        with self._lock:
            session = self._get_active_session(session_id, now=now)
            return list(session.messages[-self.history_window :])

    def assistant_turn_index(self, session_id: str) -> int:
        now = _utc_now()
        with self._lock:
            session = self._get_active_session(session_id, now=now)
            return sum(1 for message in session.messages if message.role == "assistant")


_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager
