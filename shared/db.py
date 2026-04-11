from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from shared.config import get_settings

Base = declarative_base()

@lru_cache(maxsize=32)
def get_engine(work: str | None = None):
    settings = get_settings()
    return create_engine(settings.database_url_for_work(work), pool_pre_ping=True)


@lru_cache(maxsize=32)
def get_session_factory(work: str | None = None):
    return sessionmaker(bind=get_engine(work), autoflush=False, autocommit=False, expire_on_commit=False)


SessionLocal = get_session_factory()


def create_session(work: str | None = None) -> Session:
    return get_session_factory(work)()


def get_db(work: str | None = None) -> Generator[Session, None, None]:
    session = create_session(work)
    try:
        yield session
    finally:
        session.close()
