from pathlib import Path

from shared.config import Settings
from shared.ingestion import ingest_directories
from shared.parsing import ParsedRawChunkRecord, ParsedSummaryRecord
from shared.repository import RagRepository


class FakeSession:
    def __init__(self) -> None:
        self.execute_calls = 0
        self.commit_calls = 0

    def execute(self, stmt):
        self.execute_calls += 1

        class EmptyScalarResult:
            def scalars(self):
                return self

            def all(self):
                return []

        return EmptyScalarResult()

    def flush(self):
        return None

    def commit(self):
        self.commit_calls += 1


def _summary_record() -> ParsedSummaryRecord:
    return ParsedSummaryRecord(
        external_id="Chapter_1:1",
        chapter_id="Chapter_1",
        paragraph_id=1,
        priority_score=1.0,
        timeline_layer="present",
        scene="scene",
        characters=["A"],
        mentioned_characters=[],
        tags=["tag"],
        key_events=["event"],
        plot="plot",
        embedding_text="embedding_text",
        source_path="summary.md",
        source_hash="hash",
    )


def _raw_record(index: int) -> ParsedRawChunkRecord:
    return ParsedRawChunkRecord(
        external_id=f"Chapter_1:1:{index}",
        chapter_id="Chapter_1",
        paragraph_id=1,
        chunk_id=index,
        original_text=f"text-{index}",
        embedding_text=f"embedding-{index}",
        source_path="raw.md",
        source_hash="hash",
    )


def test_upsert_raw_chunks_batches_large_insert() -> None:
    session = FakeSession()
    repo = RagRepository(session)
    records = [_raw_record(index) for index in range(65)]
    embeddings = [[0.1] * 4 for _ in records]

    repo.upsert_raw_chunks(records, embeddings, batch_size=32)

    assert session.execute_calls == 4


def test_ingest_directories_reuses_disk_cache(monkeypatch, tmp_path: Path) -> None:
    summary_dir = tmp_path / "summaries"
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    summary_dir.mkdir()
    raw_dir.mkdir()
    (summary_dir / "Chapter_1_summary.md").write_text("## 1\npriority_score: 1\ntimeline_layer: present\nscene: x\ncharacters: A\nmentioned_characters:\ntags: t\nkey_events: e\nplot: p\n", encoding="utf-8")
    (raw_dir / "Chapter_1.md").write_text("## 1\nhello\n", encoding="utf-8")

    counters = {
        "parse_summary": 0,
        "parse_raw": 0,
        "embed_texts": 0,
        "upsert_summary": 0,
        "upsert_raw": 0,
    }

    class FakeEmbeddingProvider:
        def embed_texts(self, texts):
            counters["embed_texts"] += 1
            return [[0.1, 0.2] for _ in texts]

    class FakeRepo:
        def __init__(self, session):
            self.session = session

        def upsert_summary_chunks(self, records, embeddings, *, batch_size=32):
            counters["upsert_summary"] += 1

        def upsert_raw_chunks(self, records, embeddings, *, batch_size=32):
            counters["upsert_raw"] += 1

    def fake_parse_summary_file(path):
        counters["parse_summary"] += 1
        return [_summary_record()]

    def fake_parse_raw_file(path, chunk_size, overlap):
        counters["parse_raw"] += 1
        return [_raw_record(0)]

    monkeypatch.setattr(
        "shared.ingestion.get_settings",
        lambda: Settings(embedding_model="demo-model", raw_chunk_size=10, raw_chunk_overlap=2),
    )
    monkeypatch.setattr("shared.ingestion.get_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr("shared.ingestion.RagRepository", FakeRepo)
    monkeypatch.setattr("shared.ingestion.parse_summary_file", fake_parse_summary_file)
    monkeypatch.setattr("shared.ingestion.parse_raw_file", fake_parse_raw_file)

    first_session = FakeSession()
    second_session = FakeSession()

    first = ingest_directories(
        first_session,
        summary_dir=str(summary_dir),
        raw_dir=str(raw_dir),
        cache_dir=str(cache_dir),
    )
    second = ingest_directories(
        second_session,
        summary_dir=str(summary_dir),
        raw_dir=str(raw_dir),
        cache_dir=str(cache_dir),
    )

    assert first == second == {
        "summary_files": 1,
        "summary_records": 1,
        "raw_files": 1,
        "raw_chunks": 1,
    }
    assert counters["parse_summary"] == 1
    assert counters["parse_raw"] == 1
    assert counters["embed_texts"] == 2
    assert counters["upsert_summary"] == 2
    assert counters["upsert_raw"] == 2
    assert first_session.commit_calls == 2
    assert second_session.commit_calls == 2
    assert (next(cache_dir.iterdir()) / "summary_embeddings.pkl").exists()
    assert (next(cache_dir.iterdir()) / "raw_embeddings.pkl").exists()
