import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db import Base

EMBEDDING_DIMENSION = 1024


class SummaryChunk(Base):
    __tablename__ = "summary_chunks"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_summary_chunks_external_id"),
        Index("ix_summary_chunks_chapter_id", "chapter_id"),
        Index("ix_summary_chunks_timeline_layer", "timeline_layer"),
        Index("ix_summary_chunks_source_hash", "source_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    chapter_id: Mapped[str] = mapped_column(String(120), nullable=False)
    paragraph_id: Mapped[int] = mapped_column(Integer, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    timeline_layer: Mapped[str] = mapped_column(String(120), nullable=False)
    scene: Mapped[str] = mapped_column(String(255), nullable=False)
    characters: Mapped[list[str]] = mapped_column(ARRAY(String()), nullable=False, default=list)
    mentioned_characters: Mapped[list[str]] = mapped_column(ARRAY(String()), nullable=False, default=list)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String()), nullable=False, default=list)
    key_events: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    plot: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(String(500), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)

    raw_chunks: Mapped[list["RawChunk"]] = relationship(back_populates="summary_chunk")


class RawChunk(Base):
    __tablename__ = "raw_chunks"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_raw_chunks_external_id"),
        Index("ix_raw_chunks_chapter_id", "chapter_id"),
        Index("ix_raw_chunks_source_hash", "source_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    chapter_id: Mapped[str] = mapped_column(String(120), nullable=False)
    paragraph_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(String(500), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    linked_summary_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("summary_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )

    summary_chunk: Mapped[SummaryChunk | None] = relationship(back_populates="raw_chunks")
