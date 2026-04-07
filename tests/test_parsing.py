from pathlib import Path

import pytest

from shared.parsing import normalize_list, parse_raw_file, parse_summary_file


def test_parse_summary_file_reads_structured_fields() -> None:
    records = parse_summary_file(Path("data/sample/summaries/episode_01.md"))
    assert len(records) == 2
    first = records[0]
    assert first.chapter_id == "episode_01"
    assert first.paragraph_id == 1
    assert first.characters == ["林妍", "梅子"]
    assert "走廊傳聞" in first.plot


def test_parse_summary_file_fails_on_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("## 1\nscene: hallway\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_summary_file(path)


def test_parse_raw_file_chunks_text() -> None:
    records = parse_raw_file(Path("data/sample/raw/episode_01.md"), chunk_size=120, overlap=20)
    assert len(records) >= 3
    assert records[0].chapter_id == "episode_01"
    assert records[0].paragraph_id == 1


def test_normalize_list_supports_traditional_chinese_delimiters() -> None:
    assert normalize_list("林妍、梅子，任隊長；委員會") == ["林妍", "梅子", "任隊長", "委員會"]
