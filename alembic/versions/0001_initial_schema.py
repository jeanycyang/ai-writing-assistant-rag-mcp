"""initial schema"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "summary_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("chapter_id", sa.String(length=120), nullable=False),
        sa.Column("paragraph_id", sa.Integer(), nullable=False),
        sa.Column("priority_score", sa.Float(), nullable=False),
        sa.Column("timeline_layer", sa.String(length=120), nullable=False),
        sa.Column("scene", sa.String(length=255), nullable=False),
        sa.Column("characters", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("mentioned_characters", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("key_events", sa.JSON(), nullable=False),
        sa.Column("plot", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("source_path", sa.String(length=500), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(dim=1024), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id", name="uq_summary_chunks_external_id"),
    )
    op.create_index("ix_summary_chunks_chapter_id", "summary_chunks", ["chapter_id"])
    op.create_index("ix_summary_chunks_timeline_layer", "summary_chunks", ["timeline_layer"])
    op.create_index("ix_summary_chunks_source_hash", "summary_chunks", ["source_hash"])

    op.create_table(
        "raw_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("chapter_id", sa.String(length=120), nullable=False),
        sa.Column("paragraph_id", sa.Integer(), nullable=True),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("original_text", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("source_path", sa.String(length=500), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(dim=1024), nullable=False),
        sa.Column("linked_summary_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["linked_summary_id"], ["summary_chunks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id", name="uq_raw_chunks_external_id"),
    )
    op.create_index("ix_raw_chunks_chapter_id", "raw_chunks", ["chapter_id"])
    op.create_index("ix_raw_chunks_source_hash", "raw_chunks", ["source_hash"])


def downgrade() -> None:
    op.drop_index("ix_raw_chunks_source_hash", table_name="raw_chunks")
    op.drop_index("ix_raw_chunks_chapter_id", table_name="raw_chunks")
    op.drop_table("raw_chunks")
    op.drop_index("ix_summary_chunks_source_hash", table_name="summary_chunks")
    op.drop_index("ix_summary_chunks_timeline_layer", table_name="summary_chunks")
    op.drop_index("ix_summary_chunks_chapter_id", table_name="summary_chunks")
    op.drop_table("summary_chunks")
