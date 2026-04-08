from pathlib import Path

import pytest

from shared.parsing import normalize_chapter_id, normalize_list, parse_raw_file, parse_summary_file


def test_parse_summary_file_reads_structured_fields() -> None:
    records = parse_summary_file(Path("data/sample/summaries/episode_01.md"))
    assert len(records) == 2
    first = records[0]
    assert first.chapter_id == "episode_01"
    assert first.paragraph_id == 1
    assert first.characters == ["林妍", "梅子"]
    assert "走廊傳聞" in first.plot


def test_parse_summary_file_supports_multiline_bullet_fields(tmp_path: Path) -> None:
    path = tmp_path / "Chapter_10_summary.md"
    path.write_text(
        "\n".join(
            [
                "## 1",
                "",
                "priority_score: 3",
                "timeline_layer: present",
                "",
                "scene:",
                "- 青清一水澈／深夜",
                "",
                "characters:",
                "- 挹天癒",
                "- 百重泉",
                "",
                "mentioned_characters:",
                "",
                "tags:",
                "- 青清一水澈",
                "- 子夜",
                "",
                "key_events:",
                "- 百重泉請求挹天癒醫治其父。",
                "- 挹天癒直接入內。",
                "",
                "plot:",
                "青清一水澈近子夜時分，百重泉仍在等候。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    records = parse_summary_file(path)

    assert len(records) == 1
    first = records[0]
    assert first.chapter_id == "Chapter_10"
    assert first.scene == "青清一水澈／深夜"
    assert first.characters == ["挹天癒", "百重泉"]
    assert first.mentioned_characters == []
    assert first.tags == ["青清一水澈", "子夜"]
    assert first.key_events == ["百重泉請求挹天癒醫治其父。", "挹天癒直接入內。"]
    assert first.plot == "青清一水澈近子夜時分，百重泉仍在等候。"


def test_parse_summary_file_ignores_file_title_before_first_block(tmp_path: Path) -> None:
    path = tmp_path / "Chapter_14_summary.md"
    path.write_text(
        "\n".join(
            [
                "# Chapter_14_summary",
                "",
                "## 1",
                "",
                "priority_score: 0",
                "timeline_layer: present",
                "scene:",
                "- 玉川仙境天下坪",
                "characters:",
                "- 月無缺",
                "mentioned_characters:",
                "tags:",
                "- 玉川仙境",
                "key_events:",
                "- 月無缺壓制僰君之魂。",
                "plot:",
                "月無缺壓制僰君之魂。",
            ]
        ),
        encoding="utf-8",
    )

    records = parse_summary_file(path)

    assert len(records) == 1
    assert records[0].chapter_id == "Chapter_14"
    assert records[0].paragraph_id == 1


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


def test_normalize_chapter_id_strips_summary_suffix() -> None:
    assert normalize_chapter_id("Chapter_10_summary.md") == "Chapter_10"
    assert normalize_chapter_id("Chapter_10.md") == "Chapter_10"
